"""Verify frozen files and prepare fixed validation cases before GPU training."""
from pathlib import Path
import argparse,hashlib,json,os,time,shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as f:return hashlib.sha256(f.read()).hexdigest()

def unpack_region(folder):
    """Publish complete mmap arrays only; reusable during progressive staging."""
    with np.load(folder/'teacher-labels.npz') as z:
        for key in ('semantic','instances','centre'):
            destination=folder/(key+'.npy')
            if not destination.exists():
                partial=folder/(key+'.partial.npy');np.save(partial,z[key]);partial.replace(destination)

def validation_case(args):
    name,index=args
    from data import Samples
    samples=Samples(ROOT,ROOT/'runs'/name/'snapshot.json','validation')
    result=samples.generate(100000+index,force_compressed=bool(index%2))
    path=ROOT/'runs'/name/'validation'/f'{index:03d}.npz'
    np.savez_compressed(path,**{k:v for k,v in result.items() if k!='meta'})
    path.with_suffix('.json').write_text(json.dumps(result['meta'],indent=2))
    return {'file':path.name,'sha256':sha(path),'augmented':bool(result['augmented']),**result['meta']}

def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--cases',type=int,default=48);p.add_argument('--validation-from');args=p.parse_args()
    out=ROOT/'runs'/args.name;manifest=json.loads((out/'snapshot.json').read_text())
    (out/'validation').mkdir(exist_ok=True)
    for i,row in enumerate(manifest['regions']):
        folder=ROOT/'tiles'/row['id']
        for name,expected in row['filesSHA256'].items():assert sha(folder/name)==expected,(folder,name)
        unpack_region(folder)
        print('verified/unpacked',i+1,len(manifest['regions']),row['id'],flush=True)
    if args.validation_from and not (out/'validation.json').exists():
        previous=ROOT/'runs'/args.validation_from
        cases=json.loads((previous/'validation.json').read_text())
        allowed={r['id'] for r in manifest['regions'] if r['split']=='validation'}
        assert all(set(case['sourceRegions'])<=allowed for case in cases),'Benchmark donor moved outside validation partition'
        for case in cases:
            source=previous/'validation'/case['file'];assert sha(source)==case['sha256']
            shutil.copy2(source,out/'validation'/case['file'])
            shutil.copy2(source.with_suffix('.json'),(out/'validation'/case['file']).with_suffix('.json'))
        shutil.copy2(previous/'validation.json',out/'validation.json')
    if not (out/'validation.json').exists():
        with ProcessPoolExecutor(max_workers=6) as pool:cases=list(pool.map(validation_case,[(args.name,i) for i in range(args.cases)]))
        assert sum(c['augmented'] for c in cases)>=args.cases//4,'Too few valid compressed cases; inspect generator'
        (out/'validation.json').write_text(json.dumps(cases,indent=2))
    print('PREPARED',args.name,flush=True)
if __name__=='__main__':main()
