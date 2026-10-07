"""Tracked, restartable residual nnU-Net detector training on a frozen CT corpus."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,copy,hashlib,json,math,time,traceback,sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'dependencies'))
import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.utils.data import DataLoader
from network import FiberDetector,loss_for,normalize,semantic_loss,affinity_targets
from data import TrainingDataset
from reliable_io import replace_with_retry
ROOT=Path(__file__).resolve().parent

def atomic_json(path,data):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');replace_with_retry(temp,path)

def save_checkpoint(path,data):
    temp=path.with_suffix('.partial.pt');torch.save(data,temp);replace_with_retry(temp,path)

def sha(path):
    with path.open('rb') as f:return hashlib.sha256(f.read()).hexdigest()

def panel(batch,pred,path):
    ct=batch['ct'][0,0].cpu().numpy();sem=batch['semantic'][0].cpu().numpy()
    prob=pred[0].cpu().numpy();n=ct.shape[0]
    canvas=Image.new('RGB',(3*n,3*n+40),(18,21,26));draw=ImageDraw.Draw(canvas)
    draw.text((4,3),'CT / partial pseudo-labels / predicted HZ+VT',fill='white')
    colors=np.array([[0,0,0],[245,140,60],[65,155,245],[180,160,210]])
    for axis in range(3):
        grey=np.take(ct,n//2,axis=axis).clip(0,255).astype('u1');rgb=np.repeat(grey[...,None],3,axis=2)
        truth=np.take(sem,n//2,axis=axis);label=rgb.astype('f4');mask=(truth>0)&(truth<4)
        label[mask]=.45*label[mask]+.55*colors[truth[mask]]
        p=np.take(prob,n//2,axis=axis+1);pcol=np.moveaxis(p[1:4],0,-1)@colors[1:4]
        overlay=.5*rgb+.5*pcol
        for row,arr in enumerate((rgb,label.astype('u1'),overlay.clip(0,255).astype('u1'))):
            canvas.paste(Image.fromarray(arr),(axis*n,20+row*n))
    canvas.resize((6*n,6*n+80)).save(path)

@torch.inference_mode()
def validate(model,out,step):
    model.eval();groups={'real':[],'compressed':[]};cases=json.loads((out/'validation.json').read_text());cohorts={}
    previews={'real':0,'compressed':0}
    for case in cases:
        with np.load(out/'validation'/case['file']) as z:
            batch={k:torch.as_tensor(z[k])[None].cuda(non_blocking=True) for k in z.files}
        with torch.autocast('cuda',dtype=torch.bfloat16):output=model(normalize(batch['ct'],batch['mean'],batch['std']))
        loss,parts=loss_for(output,batch);p=output['seg'].float().softmax(1)
        truth=batch['semantic'];known=truth!=255;fgtruth=(truth>0)&known;fg=p[:,1:].sum(1)
        core=(batch['centre']>0)&known
        prediction=fg>=.5
        dice=float((2*(prediction&fgtruth).sum()+1)/((prediction&known).sum()+fgtruth.sum()+1))
        covered=int((prediction&core).sum());core_count=int(core.sum())
        targets,valid=affinity_targets(batch['instances'],truth)
        affinity=output['affinity'].sigmoid();fp=int(((affinity>.5)&(targets==0)&(valid>0)).sum());neg=int(((targets==0)&(valid>0)).sum())
        row={'loss':float(loss),'semanticLoss':float(parts['semantic']),'foregroundDice':dice,
             'skeletonCovered':covered,'skeletonCount':core_count,'affinityFalsePositivePairs':fp,'affinityNegativePairs':neg}
        background=truth==0
        row['backgroundFalsePositiveVoxels']=int((prediction&background).sum())
        row['backgroundVoxels']=int(background.sum())
        group='compressed' if case['augmented'] else 'real';groups[group].append(row)
        cohorts.setdefault(case.get('cohort','legacy')+'/'+group,[]).append(row)
        if previews[group]<2:
            dest=out/'examples';dest.mkdir(exist_ok=True)
            panel(batch,p,dest/f'{group}-{previews[group]}-latest.png')
            if step==0:panel(batch,p,dest/f'{group}-{previews[group]}-baseline.png')
            previews[group]+=1
    summary={}
    for group,rows in {**groups,**cohorts}.items():
        count=sum(r['skeletonCount'] for r in rows);neg=sum(r['affinityNegativePairs'] for r in rows)
        summary[group]={'cases':len(rows),'loss':float(np.mean([r['loss'] for r in rows])),
            'semanticLoss':float(np.mean([r['semanticLoss'] for r in rows])),
            'foregroundDice':float(np.mean([r['foregroundDice'] for r in rows])),
            'skeletonCoverage':sum(r['skeletonCovered'] for r in rows)/count if count else None,
            'skeletonVoxels':count,'affinityFalsePositiveRate':sum(r['affinityFalsePositivePairs'] for r in rows)/neg if neg else None,
            'affinityNegativePairs':neg,
            'backgroundFalsePositiveRate':sum(r['backgroundFalsePositiveVoxels'] for r in rows)/max(1,sum(r['backgroundVoxels'] for r in rows))}
    # Real CT has twice the semantic weight of constructed compressed samples.
    score=summary['real']['semanticLoss']+.5*summary['compressed']['semanticLoss']
    summary['selectionScore']=score;summary['step']=step
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--resume',action='store_true');p.add_argument('--initialize-from',type=Path)
    p.add_argument('--control',action='store_true');p.add_argument('--workers',type=int,default=6)
    p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--learning-rate',type=float,default=2e-5)
    p.add_argument('--checkpoint-every',type=int,default=50)
    p.add_argument('--patience',type=int,default=20);args=p.parse_args()
    out=ROOT/'runs'/args.name;out.mkdir(exist_ok=True)
    snapshot=out/'snapshot.json';snapshot_sha=sha(snapshot);torch.manual_seed(3582026);np.random.seed(3582026)
    manifest=json.loads(snapshot.read_text())
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    config={'name':args.name,'maxSteps':args.steps,'batchSize':2,'accumulation':4,'seed':3582026,
        'learningRate':args.learning_rate,'warmupSteps':100,'validationEvery':args.validate_every,'workers':args.workers,
        'snapshotSHA256':snapshot_sha,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'precision':'bf16','augmentationEnabled':not args.control,'materialThresholdU8Range':[50,80],
        'sourceLabelStatus':'Long teacher pseudo-labels; not human ground truth','model':'ResidualEncoderUNet + centre + six affinities',
        'selection':'real semantic loss + 0.5 compressed semantic loss','earlyStoppingPatience':args.patience,'earlyStoppingMinSteps':1000,
        'codeSHA256':{n:sha(ROOT/n) for n in ('network.py','data.py','material.py','train.py','reliable_io.py')},
        'teacherSHA256':manifest['teacherCheckpoints'],
        'sampleOffset':10000000,'distillation':False,'parentStep':8000,'checkpointEvery':args.checkpoint_every,
        'sampling':'uniform scroll, uniform family, arclength within group; compaction donors from same scroll',
        'initializationSHA256':sha(args.initialize_from) if args.initialize_from else None,
        'limitations':['Geometric material compaction, not calibrated mechanical simulation.',
          'Validation combines the unchanged PHerc0813 cases and disjoint PHerc1447 regions; targets are pseudo-labels, not independently verified fibre accuracy.',
          'Intersection-dominated targets are ignored; intersection performance is not established.']}
    plans=json.loads((ROOT/'weights/plans.json').read_text());model=FiberDetector(plans)
    if not args.resume and args.initialize_from is None:raise ValueError('Full v5 must initialize explicitly from checkpoint 8000')
    if args.initialize_from:
        init=torch.load(args.initialize_from,map_location='cpu',weights_only=False);model.load_state_dict(init['model'],strict=True)
    model.cuda();ema=copy.deepcopy(model).eval()
    for param in ema.parameters():param.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learningRate'],weight_decay=1e-4)
    step=0;best=float('inf');stale=0
    if args.resume:
        previous=json.loads((out/'config.json').read_text());assert previous['snapshotSHA256']==snapshot_sha
        config['initializationSHA256']=previous['initializationSHA256']
        assert previous['codeSHA256']==config['codeSHA256'],'Code changed; use a new experiment rather than silent resume'
        ckpt=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(ckpt['model']);ema.load_state_dict(ckpt['ema']);optimizer.load_state_dict(ckpt['optimizer'])
        step=ckpt['step'];best=ckpt['best'];stale=ckpt['stale'];torch.set_rng_state(ckpt['rngCPU'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in ckpt['rngCUDA']])
        if (out/'metrics.jsonl').exists():
            lines=[l for l in (out/'metrics.jsonl').read_text().splitlines() if json.loads(l)['step']<=step]
            (out/'metrics.jsonl').write_text('\n'.join(lines)+'\n')
    elif (out/'last.pt').exists() or (out/'metrics.jsonl').exists():raise FileExistsError('Existing run; use --resume or a new snapshot name')
    atomic_json(out/'config.json',config)
    def status(stage,**extra):atomic_json(out/'status.json',{'stage':stage,'step':step,'maxSteps':args.steps,'bestScore':best if np.isfinite(best) else None,'updatedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra})
    def record(row):
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    status('baseline validation' if step==0 else 'resuming')
    if step==0:
        baseline=validate(ema,out,0);best=baseline['selectionScore'];atomic_json(out/'baseline.json',baseline)
        record({'kind':'validation','step':0,**baseline})
        save_checkpoint(out/'best.pt',{'model':ema.state_dict(),'step':0,'config':config,'validation':baseline})
    dataset=TrainingDataset(ROOT,snapshot,'train',offset=config['sampleOffset']+step*8,compressed=not args.control,length=(args.steps-step)*8)
    loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=args.workers,pin_memory=True,
        persistent_workers=args.workers>0,**({'prefetch_factor':2} if args.workers else {}))
    iterator=iter(loader);started=time.monotonic();start_step=step
    model.train();status('training')
    while step<args.steps:
        if (out/'STOP').exists():break
        begin=time.monotonic();ratio=min(1.,(step+1)/100)*.5*(1+math.cos(math.pi*max(0,step-100)/max(1,args.steps-100)))
        lr=config['learningRate']*max(.02,ratio)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);parts_sum={};loss_sum=0.;aug=0.;data_seconds=0.
        for _ in range(4):
            wait_started=time.monotonic();host_batch=next(iterator);data_seconds+=time.monotonic()-wait_started
            batch={k:v.cuda(non_blocking=True) for k,v in host_batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(normalize(batch['ct'],batch['mean'],batch['std']));loss,parts=loss_for(output,batch)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            (loss/4).backward();loss_sum+=float(loss)/4;aug+=float(batch['augmented'].mean())/4
            for key,v in parts.items():parts_sum[key]=parts_sum.get(key,0)+float(v)/4
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();step+=1
        with torch.no_grad():
            decay=min(.999,1-1/(step+1))
            for dst,src in zip(ema.parameters(),model.parameters()):dst.lerp_(src,1-decay)
        elapsed=time.monotonic()-started
        row={'kind':'train','step':step,'loss':loss_sum,**parts_sum,'lr':lr,'augmentedFraction':aug,
             'seconds':time.monotonic()-begin,'dataWaitSeconds':data_seconds,'gradientNorm':float(norm),'elapsedSeconds':elapsed}
        record(row);status('training',loss=loss_sum,secondsPerStep=elapsed/(step-start_step),estimatedRemainingSeconds=elapsed/(step-start_step)*(args.steps-step))
        if step%10==0:print(json.dumps(row),flush=True)
        if step%args.validate_every==0 or step==args.steps:
            status('validation');result=validate(ema,out,step);record({'kind':'validation',**result})
            improved=result['selectionScore']<best
            if improved:
                best=result['selectionScore'];stale=0
                save_checkpoint(out/'best.pt',{'model':ema.state_dict(),'step':step,'config':config,'validation':result})
            else:stale+=1
            atomic_json(out/'validation-latest.json',result)
            save_checkpoint(out/'last.pt',{'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                'step':step,'best':best,'stale':stale,'rngCPU':torch.get_rng_state(),'rngCUDA':torch.cuda.get_rng_state_all(),'config':config})
            model.train()
            if stale>=args.patience and step>=1000:break
        elif step==10 or step%args.checkpoint_every==0:
            save_checkpoint(out/'last.pt',{'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                'step':step,'best':best,'stale':stale,'rngCPU':torch.get_rng_state(),'rngCUDA':torch.cuda.get_rng_state_all(),'config':config})
        if step%500==0:
            checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
            save_checkpoint(checkpoints/f'FIBER_FULL_V5_{step:05d}.pt',{'model':ema.state_dict(),'step':step,'config':config})
    save_checkpoint(out/'last.pt',{'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
        'step':step,'best':best,'stale':stale,'rngCPU':torch.get_rng_state(),'rngCUDA':torch.cuda.get_rng_state_all(),'config':config})
    status('stopped' if (out/'STOP').exists() else 'complete',earlyStopped=step<args.steps)

if __name__=='__main__':
    try:main()
    except BaseException:
        import sys
        out=ROOT/'runs'/sys.argv[1];out.mkdir(exist_ok=True)
        atomic_json(out/'failure.json',{'error':traceback.format_exc(),'updatedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});raise
