"""Freeze paired compaction validation and preserve all legacy cases unchanged."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,shutil,time
import numpy as np
from settings import ROOT,OUT,SOURCE_RUN
from reliable_io import atomic_json
from data import Samples
from geometry import sheet_normal,Compaction


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def geometry_checks():
    # Two material slabs separated by a real empty gap. Closing void must reduce
    # their spacing more than material thickness; source/target transforms agree.
    profile=np.full(512,110.,dtype='f4');profile[236:276]=0
    ct=np.broadcast_to(profile[:,None,None],(512,512,512))
    warp=Compaction(ct,[255.5]*3,[1.,0,0],1.,.05)
    source=np.array([[220.,255.5,255.5],[228.,255.5,255.5],
                     [283.,255.5,255.5],[291.,255.5,255.5]],dtype='f4')
    output=warp.output_points(source)
    reconstructed=warp.source_coordinates(output.T).T
    assert np.max(np.abs(reconstructed-source))<.35
    gap_before=float(source[2,0]-source[1,0]);gap_after=float(output[2,0]-output[1,0])
    thickness_before=float(source[1,0]-source[0,0]);thickness_after=float(output[1,0]-output[0,0])
    assert gap_after/gap_before<.5,(gap_before,gap_after)
    assert thickness_after/thickness_before>.85,(thickness_before,thickness_after)
    # Oblique transform round trip, including smoothly varying inverse interpolation.
    oblique=Compaction(ct,[255.5]*3,[.8,.5,.3],.75,.08)
    x=np.random.default_rng(42).uniform(-50,50,(300,3)).astype('f4')
    input_points=oblique.source_coordinates(x.T).T
    recovered=oblique.output_points(input_points)
    assert recovered.shape==x.shape and np.max(np.abs(x-recovered))<.04
    return dict(voidGapBefore=gap_before,voidGapAfter=gap_after,materialThicknessBefore=thickness_before,
                materialThicknessAfter=thickness_after,maxObliqueRoundTripError=float(np.max(np.abs(x-recovered))))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'prepared.json').exists():return
    atomic_json(ROOT/'geometry-verified.json',geometry_checks())
    shutil.copy2(SOURCE_RUN/'snapshot.json',OUT/'snapshot.json')
    folder=OUT/'validation';folder.mkdir(exist_ok=True)
    cases=[];samples=Samples('validation')
    variants=[('real',1.,1.),('mild',1.,.45),('moderate',.9,.15),('hard',.65,.04)]
    for index in range(16):
        rng=np.random.default_rng(94000000+index)
        scroll=samples.scrolls[index%len(samples.scrolls)]
        family=('vertical','horizontal')[(index//len(samples.scrolls))%2]
        row=samples.choose(rng,scroll=scroll,family=family)
        center=np.full(3,255.5,dtype='f4')
        normal,confidence=sheet_normal(samples.region(row['region'])['ct'],center)
        for severity,material,void in variants:
            result=samples.generate(94000000+index,forced_factors=(material,void),row=row,center=center,normal=normal)
            name=f'paired-{index:02d}-{severity}.npz'
            payload={k:np.ascontiguousarray(v,dtype='f4' if k in ('ct','target','distance','mean','std','augmented','factor','void_factor','rejected') else 'u2' if k=='instances' else 'u1') for k,v in result.items() if k!='meta'}
            # Scalars must remain scalars when stacked by the validation reader.
            for key in ('mean','std','augmented','factor','void_factor','rejected'):payload[key]=np.asarray(result[key],dtype='f4')
            np.savez_compressed(folder/name,**payload)
            cohort='retention-0813' if scroll=='PHerc0813' else 'new-1447'
            cases.append(dict(file=name,sha256=digest(folder/name),group='real' if severity=='real' else 'compressed',
                              severity=severity,cohort=cohort,legacy=False,augmented=severity!='real',
                              normalConfidence=confidence,**result['meta']))
            atomic_json(ROOT/'preparation-status.json',dict(stage='freezing paired validation',completed=len(cases),total=64,updated=time.time()))
    # Keep source files byte-for-byte, with explicit legacy groups in reports.
    for source in json.loads((SOURCE_RUN/'validation.json').read_text()):
        name='legacy-'+source['file'];src=SOURCE_RUN/'validation'/source['file'];dst=folder/name
        assert digest(src)==source['sha256']
        if not dst.exists():os.link(src,dst)
        cases.append(dict(**{k:v for k,v in source.items() if k not in ('file','group')},file=name,
                          group='legacy-compressed' if source['augmented'] else 'legacy-real',legacy=True,
                          severity='legacy-compressed' if source['augmented'] else 'legacy-real'))
    atomic_json(OUT/'validation.json',cases)
    atomic_json(OUT/'phase.json',dict(parentRun='multiscroll-full-v5',parentStep=874,baseModelStep=8000,
                                    optimizerReset=True,reason='Centre objective and compaction distribution changed'))
    atomic_json(OUT/'prepared.json',dict(snapshot=digest(OUT/'snapshot.json'),validation=digest(OUT/'validation.json'),cases=len(cases)))
    atomic_json(ROOT/'preparation-status.json',dict(stage='ready',cases=len(cases),updated=time.time()))


if __name__=='__main__':main()
