"""Pretrained residual nnU-Net with centreline and local instance-affinity heads.

Backbone and decoder stages: MIC-DKFZ dynamic-network-architectures 0.4.4.
Auxiliary supervision leaves the released four-class inference head intact.
"""
import importlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'dependencies'))
import torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent/'dependencies'))
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class FiberDetector(nn.Module):
    def __init__(self, plans):
        super().__init__()
        config=plans['configurations']['3d_fullres'];args=dict(config['architecture']['arch_kwargs'])
        for key in config['architecture']['_kw_requires_import']:
            if args[key] is not None:
                mod,name=args[key].rsplit('.',1);args[key]=getattr(importlib.import_module(mod),name)
        self.backbone=ResidualEncoderUNet(input_channels=1,num_classes=4,deep_supervision=False,**args)
        channels=args['features_per_stage'][0]
        self.centre=nn.Conv3d(channels,1,1)
        self.affinity=nn.Conv3d(channels,6,1)
        nn.init.normal_(self.centre.weight,std=.01);nn.init.constant_(self.centre.bias,-2)
        nn.init.normal_(self.affinity.weight,std=.01);nn.init.zeros_(self.affinity.bias)

    def initialize(self,path):
        # Published user-requested checkpoint with separately verified SHA256.
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        return self.backbone.load_state_dict(ckpt['network_weights'],strict=True)

    def forward(self,x):
        skips=self.backbone.encoder(x);decoder=self.backbone.decoder
        x=skips[-1];deep=[]
        for i,(up,stage) in enumerate(zip(decoder.transpconvs,decoder.stages)):
            x=stage(torch.cat((up(x),skips[-i-2]),dim=1))
            if i>=len(decoder.stages)-3:deep.append(decoder.seg_layers[i](x))
        return {'seg':deep[-1],'deep':deep[-2::-1],'centre':self.centre(x),'affinity':self.affinity(x)}

def normalize(ct,mean=None,std=None):
    ct=ct.float()
    if mean is None:mean=ct.mean((2,3,4),keepdim=True)
    else:mean=mean[:,None,None,None,None]
    if std is None:std=ct.std((2,3,4),keepdim=True)
    else:std=std[:,None,None,None,None]
    return (ct-mean)/std.clamp_min(10.)

def balanced_bce(logits,target,valid):
    loss=F.binary_cross_entropy_with_logits(logits.float(),target.float(),reduction='none')
    target=target.float();valid=valid.float()
    pos=valid*target;neg=valid*(1-target)
    terms=[]
    # Normalize present classes independently. Unknown examples contribute nothing.
    for weight in (pos,neg):
        terms.append((loss*weight).sum()/weight.sum().clamp_min(1))
    return .5*sum(terms)

def semantic_loss(logits,target):
    valid=target!=255
    ce=F.cross_entropy(logits.float(),target.long(),ignore_index=255,reduction='none')
    # Foreground and known background receive equal aggregate weight if present.
    fg=valid&(target>0);bg=valid&(target==0)
    ce=.5*((ce*fg).sum()/fg.sum().clamp_min(1)+(ce*bg).sum()/bg.sum().clamp_min(1))
    p=logits.float().softmax(1)
    dice=[]
    for cls in (1,2):
        truth=(target==cls).float();pred=p[:,cls]*valid
        dice.append(1-(2*(pred*truth).sum()+1)/(pred.sum()+truth.sum()+1))
    return ce+.5*sum(dice)

def affinity_targets(instance,semantic):
    targets=[];valids=[]
    # Channel order: dz1,dy1,dx1,dz4,dy4,dx4. No wraparound edges.
    for step in (1,4):
        for axis in (1,2,3):
            a=[slice(None)]*4;b=a.copy();a[axis]=slice(None,-step);b[axis]=slice(step,None)
            a=tuple(a);b=tuple(b)
            left,right=instance[a],instance[b]
            valid=(semantic[a]!=255)&(semantic[b]!=255)&((left>0)|(right>0))
            target=(left==right)&(left>0)
            output=torch.zeros_like(instance,dtype=torch.float32);mask=output.clone()
            output[a]=target;mask[a]=valid
            targets.append(output);valids.append(mask)
    return torch.stack(targets,1),torch.stack(valids,1)

def loss_for(output,batch):
    sem=batch['semantic'];core=batch['centre']>0
    seg=semantic_loss(output['seg'],sem)
    deep=seg*0
    for weight,logits in zip((.25,.125),output['deep']):
        # Only supervise a coarse cell if all contributing fine cells are known.
        factor=sem.shape[-1]//logits.shape[-1]
        target=F.interpolate(sem[:,None].float(),size=logits.shape[2:],mode='nearest')[:,0].long()
        unknown=F.max_pool3d((sem==255)[:,None].float(),factor,stride=factor)[:,0]>0
        target[unknown]=255
        deep=deep+weight*semantic_loss(logits,target)
    center=balanced_bce(output['centre'][:,0],core,sem!=255)
    p=output['seg'].float().softmax(1)
    fg=p[:,1]+p[:,2]+p[:,3]
    recall=((1-fg)*core).sum()/core.sum().clamp_min(1)
    targets,valid=affinity_targets(batch['instances'],sem)
    affinity=balanced_bce(output['affinity'],targets,valid)
    total=seg+deep+.25*center+.25*recall+.3*affinity
    return total,{'semantic':seg.detach(),'deep':deep.detach(),'centre':center.detach(),
                  'skeleton_recall_loss':recall.detach(),'affinity':affinity.detach()}
