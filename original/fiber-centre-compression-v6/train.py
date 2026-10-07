"""Checkpointed continuation from v5 +874 with wide centres and void-first compaction."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,hashlib,json,math,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt,maximum_filter
from PIL import Image,ImageDraw
from settings import ROOT,OUT,SOURCE,SOURCE_RUN,PHASE_START,INITIAL_SHA
from reliable_io import atomic_json,replace_with_retry
from network import FiberDetector,normalize,loss_for,affinity_targets
from data import TrainingDataset


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    tmp=path.with_suffix('.partial.pt');torch.save(value,tmp);replace_with_retry(tmp,path)


def load_case(case):
    with np.load(OUT/'validation'/case['file']) as z:d={k:np.array(z[k]) for k in z.files}
    if 'target' not in d:
        core=d['centre']>0
        distance=distance_transform_edt(~core).astype('f4') if core.any() else np.full(core.shape,128,'f4')
        d['distance']=distance;d['valid']=((d['semantic']==0)|((d['semantic']<255)&(d['instances']>0)&(distance<=6))).astype('u1')
        d['target']=np.exp(-.5*(distance/1.5)**2).astype('f4');d['target'][(d['semantic']==0)|(d['valid']==0)]=0
    return {k:torch.as_tensor(v.astype('i8') if k in ('semantic','instances','centre') else v.astype('f4'))[None].cuda()
            for k,v in d.items()}


def preview(batch,output,path):
    ct=batch['ct'][0,0].cpu().numpy();truth=batch['target'][0].cpu().numpy()
    pred=output['centre'][0,0].float().sigmoid().cpu().numpy();n=ct.shape[0]
    canvas=Image.new('RGB',(3*n,3*n+24),(18,21,26));draw=ImageDraw.Draw(canvas)
    draw.text((4,4),'CT / spline centre target / predicted centre score',fill='white')
    color=np.array([60,165,250],dtype='f4')
    for axis in range(3):
        grey=np.take(ct,n//2,axis).clip(0,255);rgb=np.repeat(grey[...,None],3,2)
        for row,p in enumerate((None,np.take(truth,n//2,axis),np.take(pred,n//2,axis))):
            pixels=rgb if p is None else rgb*(1-.8*p[...,None])+color*.8*p[...,None]
            canvas.paste(Image.fromarray(pixels.clip(0,255).astype('u1')),(axis*n,24+row*n))
    canvas.resize((6*n,6*n+48)).save(path)


@torch.inference_mode()
def validate(model,step):
    model.eval();groups={};rows=[];seen={}
    coordinates=np.indices((5,)*3)-2;sphere=(coordinates*coordinates).sum(0)<=4
    sl=(slice(8,-8),)*3
    for case in json.loads((OUT/'validation.json').read_text()):
        batch=load_case(case)
        with torch.autocast('cuda',dtype=torch.bfloat16):output=model(normalize(batch['ct'],batch['mean'],batch['std']))
        loss,parts=loss_for(output,batch)
        p=output['seg'][0].float().softmax(0).cpu().numpy();c=output['centre'][0,0].float().sigmoid().cpu().numpy()
        gt=batch['semantic'][0].cpu().numpy()[sl];known=gt<255;fg=(gt>0)&known
        detected=((1-p[0][sl])>=.5)&known
        core=(batch['centre'][0].cpu().numpy()[sl]>0)&known
        valid=batch['valid'][0].cpu().numpy()[sl]>0;dist=batch['distance'][0].cpu().numpy()[sl]
        maximum=maximum_filter(c,footprint=sphere,mode='constant')[sl];c=c[sl]
        target,pairs=affinity_targets(batch['instances'],batch['semantic']);aff=output['affinity'].sigmoid()
        neg=(target==0)&(pairs>0)
        row=dict(file=case['file'],semanticLoss=float(parts['semantic']),centreLoss=float(parts['centre']),loss=float(loss),
                 foregroundDice=float((2*(detected&fg).sum()+1)/(detected.sum()+fg.sum()+1)),
                 coreCount=int(core.sum()),semanticCovered=int((detected&core).sum()),
                 affinityNegativePairs=int(neg.sum()),affinityFalsePositivePairs=int(((aff>.5)&neg).sum()),
                 backgroundCount=int((gt==0).sum()),backgroundFP=int((detected&(gt==0)).sum()),curve=[])
        for threshold in (.5,.75,.9):
            mask=(c>=threshold)&valid
            row['curve'].append(dict(threshold=threshold,positive=int(mask.sum()),near=int((mask&(dist<=2)&fg).sum()),
                                     covered=int(((maximum>=threshold)&core).sum()),backgroundFP=int(((c>=threshold)&(gt==0)).sum())))
        names=[case['group'],case.get('cohort','unknown')+'/'+case['group']]
        if not case['legacy'] and case['severity']!='real':names.append(case['severity'])
        for name in names:groups.setdefault(name,[]).append(row)
        rows.append(row)
        if not case['legacy']:
            for group in [case['group']]+([case['severity']] if case['severity']!='real' else []):
                index=seen.get(group,0)
                if index<2:
                    folder=OUT/'examples';folder.mkdir(exist_ok=True)
                    preview(batch,output,folder/f'{group}-{index}-latest.png')
                    if step==PHASE_START:preview(batch,output,folder/f'{group}-{index}-baseline.png')
                    seen[group]=index+1
    summary={}
    for name,items in groups.items():
        cores=sum(x['coreCount'] for x in items);bg=sum(x['backgroundCount'] for x in items);neg=sum(x['affinityNegativePairs'] for x in items)
        value={key:float(np.mean([x[key] for x in items])) for key in ('loss','semanticLoss','centreLoss','foregroundDice')}
        value.update(cases=len(items),skeletonVoxels=cores,
                     skeletonCoverage=sum(x['semanticCovered'] for x in items)/max(1,cores),
                     affinityFalsePositiveRate=sum(x['affinityFalsePositivePairs'] for x in items)/max(1,neg),
                     backgroundFalsePositiveRate=sum(x['backgroundFP'] for x in items)/max(1,bg),centreCurve=[])
        for i,threshold in enumerate((.5,.75,.9)):
            positives=sum(x['curve'][i]['positive'] for x in items)
            value['centreCurve'].append(dict(threshold=threshold,
                  coverage=sum(x['curve'][i]['covered'] for x in items)/max(1,cores),
                  precision=sum(x['curve'][i]['near'] for x in items)/max(1,positives),
                  voxelsPerReferencePoint=positives/max(1,cores),
                  backgroundFPR=sum(x['curve'][i]['backgroundFP'] for x in items)/max(1,bg)))
        value.update(centreCoverage=value['centreCurve'][0]['coverage'],centrePrecision=value['centreCurve'][0]['precision'])
        summary[name]=value
    summary.update(step=step,selectionScore=summary['real']['semanticLoss']+.5*summary['compressed']['semanticLoss']+
                   .5*summary['real']['centreLoss']+.5*summary['hard']['centreLoss'])
    atomic_json(OUT/f'validation-cases-{step}.json',rows)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--stop-after',type=int)
    ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(3582026);np.random.seed(3582026)
    torch.backends.cudnn.benchmark=False
    code={n:digest(ROOT/n) for n in ('settings.py','geometry.py','data.py','network.py','train.py','reliable_io.py')}
    prepared=json.loads((OUT/'prepared.json').read_text())
    assert digest(OUT/'snapshot.json')==prepared['snapshot'] and digest(OUT/'validation.json')==prepared['validation']
    config=dict(name='multiscroll-centre-v6',phaseStartStep=PHASE_START,maxSteps=12000,batchSize=2,accumulation=4,
                learningRate=2e-5,warmupSteps=50,workers=args.workers,codeSHA256=code,
                sourceSnapshotSHA256=prepared['snapshot'],validationSHA256=prepared['validation'],
                initializationSHA256=INITIAL_SHA,initializationState='v5 +874 model weights',optimizerReset=True,
                targetSigmaVoxels=1.5,centreLossWeight=2.,precision='bf16',device=torch.cuda.get_device_name(0),
                compression='single-region smooth monotone void-first compaction',nativeProposalFraction=.5,
                curriculum='hard fraction of augmented samples ramps from 0 at phase step 100 to 0.4 at 750',
                compressedMixAfter750={'mild':.2,'moderate':.4,'hard':.4},
                materialFactors={'mild':[1,1],'moderate':[.85,1],'hard':[.55,.8]},
                voidFactors={'mild':[.3,.6],'moderate':[.08,.25],'hard':[.02,.1]},
                checkpointEvery=25,validationEvery=250,distillation=False,productionPromotion=False,
                limitations=['Automatic partial targets, not independent human ground truth',
                             'Geometric compaction, not calibrated material mechanics',
                             'Synthetic hard-case success does not establish performance on real compressed fibres'])
    model=FiberDetector(json.loads((SOURCE/'weights/plans.json').read_text())).cuda()
    ema=copy.deepcopy(model).eval();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)
    if args.resume:
        previous=json.loads((OUT/'config.json').read_text());assert previous['codeSHA256']==code
        state=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False)
        assert state['config']['validationSHA256']==config['validationSHA256']
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);optimizer.load_state_dict(state['optimizer'])
        step=state['step'];best=state['best'];stale=state['stale']
        torch.set_rng_state(state['rngCPU']);torch.cuda.set_rng_state_all(state['rngCUDA'])
        if (OUT/'metrics.jsonl').exists():
            lines=[l for l in (OUT/'metrics.jsonl').read_text().splitlines() if json.loads(l)['step']<=step]
            (OUT/'metrics.jsonl').write_text('\n'.join(lines)+'\n')
    else:
        assert not (OUT/'last.pt').exists()
        assert digest(SOURCE_RUN/'last.pt')==INITIAL_SHA
        state=torch.load(SOURCE_RUN/'last.pt',map_location='cpu',weights_only=False);assert state['step']==PHASE_START
        model.load_state_dict(state['model']);ema.load_state_dict(state['model']);step=PHASE_START;best=float('inf');stale=0
    del state
    for parameter in ema.parameters():parameter.requires_grad_(False)
    atomic_json(OUT/'config.json',config)
    def status(stage,**kw):
        atomic_json(OUT/'status.json',dict(stage=stage,step=step,phaseStep=step-PHASE_START,maxSteps=12000,
                                         updatedAt=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
    def record(value):
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    def checkpoint():
        save(OUT/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,
                               best=best,stale=stale,rngCPU=torch.get_rng_state(),rngCUDA=torch.cuda.get_rng_state_all(),config=config))
        atomic_json(OUT/'checkpoint-present.json',dict(step=step,updated=time.time()))
    if not args.resume:
        status('baseline validation');baseline=validate(ema,step);best=baseline['selectionScore']
        atomic_json(OUT/'baseline.json',baseline);record(dict(kind='validation',**baseline))
        save(OUT/'best.pt',dict(model=ema.state_dict(),step=step,config=config,validation=baseline))
    baseline=json.loads((OUT/'baseline.json').read_text())
    status('loading training data')
    dataset=TrainingDataset(offset=(step-PHASE_START)*8,length=(12000-step)*8)
    loader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False,num_workers=args.workers,pin_memory=True,
                                     persistent_workers=args.workers>0,**({'prefetch_factor':2} if args.workers else {}))
    iterator=iter(loader);model.train();begin=time.monotonic();start_step=step
    while step<12000 and not (OUT/'STOP').exists():
        tick=time.monotonic();phase=step-PHASE_START
        lr=2e-5*min(1.,(phase+1)/50)*max(.02,.5*(1+math.cos(math.pi*max(0,phase-50)/(12000-PHASE_START-50))))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);values={};loss_value=0.;wait=0.;counts={'native':0,'mild':0,'moderate':0,'hard':0,'rejected':0}
        for _ in range(4):
            stamp=time.monotonic();batch=next(iterator);wait+=time.monotonic()-stamp
            aug=batch['augmented'].numpy()>0;f=batch['factor'].numpy();v=batch['void_factor'].numpy()
            counts['native']+=int((~aug).sum());counts['mild']+=int((aug&(v>=.3)).sum())
            counts['hard']+=int((aug&(f<.85)).sum());counts['moderate']+=int((aug&(f>=.85)&(v<.3)).sum())
            counts['rejected']+=int(batch['rejected'].sum())
            batch={k:value.cuda(non_blocking=True) for k,value in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(normalize(batch['ct'],batch['mean'],batch['std']));loss,parts=loss_for(output,batch)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            (loss/4).backward();loss_value+=float(loss)/4
            for k,value in parts.items():values[k]=values.get(k,0)+float(value)/4
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();step+=1
        with torch.no_grad():
            decay=min(.999,1-1/(step-PHASE_START+1))
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)
        row=dict(kind='train',step=step,phaseStep=step-PHASE_START,loss=loss_value,**values,lr=lr,
                 seconds=time.monotonic()-tick,dataWaitSeconds=wait,gradientNorm=float(norm),samples=counts,
                 augmentedFraction=1-counts['native']/8)
        record(row);rate=(time.monotonic()-begin)/(step-start_step)
        status('training',loss=loss_value,secondsPerStep=rate,estimatedRemainingSeconds=rate*(12000-step),samples=counts)
        if (step-PHASE_START)%10==0:print(json.dumps(row),flush=True)
        if (step-PHASE_START)%250==0 or step-PHASE_START==100 or step==12000:
            status('validation');result=validate(ema,step)
            guard=(result['real']['foregroundDice']>=baseline['real']['foregroundDice']-.03 and
                   result['real']['centreCoverage']>=baseline['real']['centreCoverage']-.03)
            result['retentionGuardPassed']=guard
            if result['selectionScore']<best and guard:
                best=result['selectionScore'];stale=0;save(OUT/'best.pt',dict(model=ema.state_dict(),step=step,config=config,validation=result))
            else:stale+=1
            atomic_json(OUT/'validation-latest.json',result);record(dict(kind='validation',**result));checkpoint();model.train()
            if stale>=20 and step-PHASE_START>=1000:break
        elif (step-PHASE_START)%25==0:checkpoint()
        if (step-PHASE_START)%500==0:
            folder=OUT/'checkpoints';folder.mkdir(exist_ok=True)
            save(folder/f'FIBER_CENTRE_V6_{step:05d}.pt',dict(model=ema.state_dict(),step=step,config=config))
        if args.stop_after and step-start_step>=args.stop_after:break
    checkpoint()
    stage='startup-complete' if args.stop_after else 'stopped' if (OUT/'STOP').exists() else 'complete'
    status(stage)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    try:main()
    except BaseException:
        atomic_json(OUT/'failure.json',dict(error=traceback.format_exc(),updatedAt=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        raise
