"""Ordered material compaction of real CT slabs, with conservative partial labels.

CT below the sampled threshold contributes neither thickness nor intensity.
This is a controlled geometric stress augmentation, not calibrated mechanics.
Subvoxel mixtures of distinct IDs are ignored for instance/centre supervision.
"""
import numpy as np
from scipy.ndimage import map_coordinates,gaussian_filter

def compact(slabs, factor, threshold=65, samples=3):
    n=slabs[0]['ct'].shape[0]
    assert 0<factor<=1 and all(s['ct'].shape==(n,n,n) for s in slabs)
    fields={k:np.concatenate([s[k] for s in slabs],axis=0) for k in ('ct','semantic','instances','centre')}
    for k,slab in enumerate(slabs):
        ids=fields['instances'][k*n:(k+1)*n]
        ids[ids>0]+=k*65536
        # Artificial joins are censored, not taught as fibre boundaries.
        fields['semantic'][k*n:k*n+2]=255;fields['semantic'][(k+1)*n-2:(k+1)*n]=255
    material=fields['ct']>=threshold
    counts=material.sum(0)
    order=np.argsort(~material,axis=0,kind='stable')
    # Neighbouring columns share a smooth compaction displacement. Independent
    # integer ranks would corrugate a fibre whenever one dark voxel appears.
    density=gaussian_filter(material.astype('f4'),sigma=(0,4,4),mode='reflect')
    cumulative=np.cumsum(density,axis=0)-.5*density
    smooth_counts=density.sum(0)
    packed_position=np.take_along_axis(cumulative,order,axis=0)
    packed_position[np.arange(len(order))[:,None,None]>=counts[None]]=np.inf
    del material,density,cumulative
    zz,yy,xx=np.indices((n,n,n),dtype=np.float32)
    yi=yy.astype(np.int32);xi=xx.astype(np.int32)
    centres=smooth_counts[None]/2+(zz-(n-1)/2)/factor
    ct=np.zeros((n,n,n),'f4');semantic=None;identity=None;core=np.zeros((n,n,n),bool)
    covered=np.ones((n,n,n),bool);same_id=np.ones((n,n,n),bool)
    for sample in range(samples):
        rank=centres+((sample+.5)/samples-.5)/factor
        first=packed_position[0][None]
        last=packed_position[np.maximum(counts-1,0),np.arange(n)[:,None],np.arange(n)[None,:]][None]
        covered&=(rank>=first)&(rank<=last)&(counts[None]>0)
        lower=np.zeros(rank.shape,'i4');upper=np.broadcast_to(counts,rank.shape).astype('i4').copy()
        for _ in range(int(np.ceil(np.log2(len(order)+1)))+1):
            mid=(lower+upper)//2
            value=packed_position[np.minimum(mid,len(order)-1),yi,xi]
            left=value<rank
            lower=np.where(left,np.minimum(mid+1,upper),lower)
            upper=np.where(left,upper,mid)
        high=np.minimum(lower,np.maximum(0,counts[None]-1)).astype('i4')
        low=np.maximum(high-1,0)
        ua=packed_position[low,yi,xi];ub=packed_position[high,yi,xi]
        alpha=np.nan_to_num(np.clip((rank-ua)/np.maximum(ub-ua,1e-6),0,1),nan=0.)
        ia=order[low,yi,xi];ib=order[high,yi,xi]
        va=fields['ct'][ia,yi,xi].astype('f4');vb=fields['ct'][ib,yi,xi].astype('f4')
        ct+=((1-alpha)*va+alpha*vb)/samples
        for idx in (ia,ib):
            sem=fields['semantic'][idx,yi,xi];ids=fields['instances'][idx,yi,xi]
            if semantic is None:semantic=sem.copy();identity=ids.copy()
            else:semantic[semantic!=sem]=255;same_id&=identity==ids
            core|=fields['centre'][idx,yi,xi]>0
    # A nearest material sample is used only to fill outside-support appearance;
    # those voxels never contribute labels or a measured compression claim.
    semantic[~covered]=255
    identity[~same_id|~covered|(semantic==255)]=0
    core&=(identity>0)&(semantic!=255)
    return {'ct':ct,'semantic':semantic,'instances':identity,'centre':core.astype('u1'),
        'meta':{'factor':float(factor),'materialThreshold':threshold,'coverage':float(covered.mean()),
                'voidFractionRemoved':float(1-counts.sum()/fields['ct'].size),
                'mixedInstanceFraction':float((~same_id&covered).mean()),'slabs':len(slabs),'compactionFieldSmoothingVoxels':4,
                'preImagingMinimumCoveredCT':float(ct[covered].min()) if covered.any() else None}}

def deform(data,rng):
    n=data['ct'].shape[0];z,y,x=np.indices((n,n,n),dtype='f4');mid=(n-1)/2
    phase=float(rng.uniform(0,2*np.pi));a=float(rng.uniform(1,5));b=float(rng.uniform(-.15,.15))
    lateral=float(rng.uniform(.75,1.05))
    # Triangular inverse warp: dx/dx=1, dy/dy=1/lateral, dz/dz=1.
    sy=mid+(y-mid)/lateral-b*(x-mid)-a*np.sin(2*np.pi*x/n+phase)
    sz=z-a*np.sin(2*np.pi*x/n+phase+.5)-.5*a*np.cos(2*np.pi*sy/n+phase)
    coords=np.array([sz,sy,x]);inside=((coords>=1)&(coords<=n-2)).all(0)
    result={}
    for key in ('ct','semantic','instances','centre'):
        result[key]=map_coordinates(data[key],coords,order=1 if key=='ct' else 0,mode='nearest',prefilter=False)
    result['semantic'][~inside]=255;result['instances'][~inside]=0;result['centre'][~inside]=0
    result['meta']={**data['meta'],'forwardWarpJacobian':lateral,'bendingAmplitudeVoxels':a}
    # Small acquisition variation follows compaction. No synthetic background.
    result['ct']=np.clip(result['ct']+rng.normal(0,rng.uniform(0,2),result['ct'].shape),0,255).astype('f4')
    return result
