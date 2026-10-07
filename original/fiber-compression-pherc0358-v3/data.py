"""Deterministic samples from a frozen corpus; fresh augmentation on every index.

Validation uses fixed indices and source regions distinct from all train donors.
Both fibre families are sampled equally, proportional to source arclength within
family. Only original sequences with >=30 points enter the donor lists.
"""
import json
from pathlib import Path
from collections import OrderedDict
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'dependencies'))
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from material import compact,deform

class Samples:
    def __init__(self,root,snapshot,split,compressed=True,size=128):
        self.root=Path(root);self.split=split;self.compressed=compressed;self.size=size;self.cache=OrderedDict()
        manifest=json.loads(Path(snapshot).read_text());self.rows=[];self.norms={}
        for region in manifest['regions']:
            if region['split']!=split:continue
            folder=self.root/'tiles'/region['id'];corpus=json.loads((folder/'corpus.json').read_text())
            self.norms[region['id']]=json.loads((folder/'hzvt-classes.json').read_text())['normalization']
            for trace in corpus['traces']:
                assert len(trace['pointsLocalXYZ'])>=30
                self.rows.append({'region':region['id'],'trace':trace})
        self.families={f:[r for r in self.rows if r['trace']['class']==f] for f in ('vertical','horizontal')}
        assert all(self.families.values()),'Both families required in each experiment partition'
        self.probs={f:np.array([r['trace']['sourceLengthMeters'] for r in rows]) for f,rows in self.families.items()}
        for f in self.probs:self.probs[f]/=self.probs[f].sum()
        self.grid=np.indices((size,)*3,dtype='f4').reshape(3,-1)-(size-1)/2

    def region(self,region):
        if region in self.cache:self.cache.move_to_end(region);return self.cache[region]
        folder=self.root/'tiles'/region;fields={'ct':np.memmap(folder/'volume.u8',mode='r',dtype='u1',shape=(512,)*3)}
        if (folder/'semantic.npy').exists():
            for key in ('semantic','instances','centre'):fields[key]=np.load(folder/(key+'.npy'),mmap_mode='r')
        else:
            with np.load(folder/'teacher-labels.npz') as z:
                fields.update({k:z[k] for k in ('semantic','instances','centre')})
        self.cache[region]=fields
        while len(self.cache)>2:self.cache.popitem(last=False)
        return fields

    def choose(self,rng,family=None):
        family=family or ('vertical','horizontal')[int(rng.integers(2))]
        rows=self.families[family]
        return rows[int(rng.choice(len(rows),p=self.probs[family]))]

    def source(self,row,rng,oriented):
        data=self.region(row['region']);points=np.asarray(row['trace']['pointsLocalXYZ'],dtype='f4')[:,::-1]
        center=points[int(rng.integers(len(points)))].copy()
        n=self.size
        if not oriented:
            start=np.clip(np.rint(center-n/2).astype(int),0,512-n)
            sel=tuple(slice(int(s),int(s+n)) for s in start)
            result={k:np.array(v[sel],dtype='i4' if k=='instances' else v.dtype) for k,v in data.items()}
        else:
            # A constant local sheet frame estimated from real CT gradients.
            ci=np.clip(np.rint(center).astype(int),12,499);sel=tuple(slice(int(s-12),int(s+13)) for s in ci)
            patch=np.asarray(data['ct'][sel],dtype='f4');grad=np.array(np.gradient(gaussian_filter(patch,1.)))
            g=grad[:,patch>=65]
            if g.shape[1]<20:g=grad.reshape(3,-1)
            _,vectors=np.linalg.eigh(g@g.T/max(1,g.shape[1]));normal=vectors[:,-1]
            if normal[np.argmax(np.abs(normal))]<0:normal=-normal
            vertical=np.array([1.,0.,0.]);vertical-=normal*np.dot(normal,vertical)
            if np.linalg.norm(vertical)<.15:
                vertical=np.array([0.,1.,0.]);vertical-=normal*np.dot(normal,vertical)
            vertical/=np.linalg.norm(vertical);horizontal=np.cross(normal,vertical)
            frame=np.stack([normal,vertical,horizontal],axis=1)
            coords=(center[:,None]+frame@self.grid).reshape((3,n,n,n))
            result={k:map_coordinates(v,coords,order=1 if k=='ct' else 0,mode='constant',cval=255 if k=='semantic' else 0,
                                     prefilter=False,output='f4' if k=='ct' else 'i4' if k=='instances' else 'u1') for k,v in data.items()}
        result['centre'][result['semantic']==255]=0
        result['meta']={'donors':[row['trace']['id']],'sourceRegions':[row['region']],'compressed':False}
        return result

    def generate(self,index,force_compressed=None):
        seed=int(index)+(3582026 if self.split=='train' else 800000000)
        rng=np.random.default_rng(seed);row=self.choose(rng)
        compressed=bool(rng.random()<.5) if force_compressed is None else force_compressed
        compressed=compressed and self.compressed
        if compressed:
            factor=float(rng.uniform(.15,.65));count=int(np.ceil(1.8/factor))
            rows=[row]+[self.choose(rng) for _ in range(count-1)]
            slabs=[self.source(r,rng,True) for r in rows]
            threshold=int(rng.integers(50,81))
            result=compact(slabs,factor,threshold=threshold)
            if result['meta']['coverage']>=.97 and np.count_nonzero(result['instances'])>=32:
                result=deform(result,rng)
                result['meta'].update(donors=[r['trace']['id'] for r in rows],sourceRegions=[r['region'] for r in rows],compressed=True)
                mean=float(result['ct'].mean());std=float(result['ct'].std())
            else:
                failure=result['meta'];result=self.source(row,rng,False);result['meta']['compressionRejected']=failure
                norm=self.norms[row['region']];mean,std=norm['mean'],norm['std']
        else:
            result=self.source(row,rng,False);norm=self.norms[row['region']];mean,std=norm['mean'],norm['std']
        result['ct']=np.ascontiguousarray(result['ct'][None],dtype='f4')
        for k in ('semantic','instances','centre'):result[k]=np.ascontiguousarray(result[k],dtype='i8')
        result['mean']=np.float32(mean);result['std']=np.float32(std);result['augmented']=np.float32(result['meta']['compressed'])
        result['meta']['seed']=seed
        return result

class TrainingDataset:
    def __init__(self,root,snapshot,split,offset=0,compressed=True,length=1000000):
        self.samples=Samples(root,snapshot,split,compressed);self.offset=offset;self.length=length
    def __len__(self):return self.length
    def __getitem__(self,i):
        result=self.samples.generate(i+self.offset)
        return {k:v for k,v in result.items() if k!='meta'}
