"""Fresh native and increasingly difficult compressed samples from the frozen v5 corpus."""
import importlib.util
from collections import OrderedDict
import numpy as np
from settings import SOURCE, SOURCE_RUN
from geometry import sheet_normal,sample

spec=importlib.util.spec_from_file_location('v5_sampling',SOURCE/'data.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


class Samples(legacy.Samples):
    def __init__(self,split):
        super().__init__(SOURCE,SOURCE_RUN/'snapshot.json',split,compressed=False)
        # Keep worker spawning and memory bounded; all sampling lists refer to rows.
        for row in self.rows:
            t=row['trace'];assert not t.get('bridgePositive',False)
            row['trace']={k:t[k] for k in ('id','class','sourceLengthMeters')}
            for k in ('pointsLocalXYZ','lineLocalXYZ'):row['trace'][k]=np.asarray(t[k],dtype='f4')
        self.region_rows={}
        for row in self.rows:self.region_rows.setdefault(row['region'],[]).append(row)
        self.line_cache=OrderedDict()

    def lines(self,region):
        if region not in self.line_cache:
            self.line_cache[region]=np.concatenate([r['trace']['lineLocalXYZ'][:,::-1] for r in self.region_rows[region]])
            while len(self.line_cache)>4:self.line_cache.popitem(last=False)
        self.line_cache.move_to_end(region)
        return self.line_cache[region]

    def generate(self,index,phase_step=0,forced_factors=None,row=None,center=None,normal=None):
        rng=np.random.default_rng(int(index)+(3582026 if self.split=='train' else 800000000))
        row=row or self.choose(rng)
        fields=self.region(row['region'])
        if center is None:
            points=row['trace']['pointsLocalXYZ'][:,::-1]
            center=points[int(rng.integers(len(points)))].copy()
        if forced_factors is not None:material,void=map(float,forced_factors)
        elif rng.random()>=.5:material,void=1.,1.
        else:
            # 50% real CT throughout. By step 750, augmented cases comprise
            # 20% mild / 40% medium / 40% hard; hard cases are introduced gradually.
            hard=.4*float(np.clip((phase_step-100)/650,0,1))
            pick=rng.random()
            if pick<hard:material,void=float(rng.uniform(.55,.8)),float(rng.uniform(.02,.1))
            elif pick<hard+.4:material,void=float(rng.uniform(.85,1.)),float(rng.uniform(.08,.25))
            else:material,void=1.,float(rng.uniform(.3,.6))
        if material==1 and void==1 and forced_factors is None:
            start=np.clip(np.rint(center-64).astype(int),0,384);center=start.astype('f4')+63.5
        elif forced_factors is None:center=np.clip(center,208,303)
        if normal is None:normal,_=sheet_normal(fields['ct'],center)
        result=sample(fields,self.lines(row['region']),center,normal,material,void)
        rejected=None
        if void<1 and forced_factors is None and (result['centre'].sum()<16 or result['meta']['coverage']<.97):
            rejected=result['meta'];start=np.clip(np.rint(center-64).astype(int),0,384)
            result=sample(fields,self.lines(row['region']),start.astype('f4')+63.5,normal,1.,1.)
            material,void=1.,1.
        norm=self.norms[row['region']]
        result['ct']=np.ascontiguousarray(result['ct'][None],dtype='f4')
        for k in ('semantic','instances','centre'):result[k]=np.ascontiguousarray(result[k],dtype='i8')
        result.update(mean=np.float32(norm['mean']),std=np.float32(norm['std']),
                      augmented=np.float32(void<1),factor=np.float32(material),void_factor=np.float32(void),
                      rejected=np.float32(rejected is not None))
        result['meta'].update(seed=int(index),sourceRegions=[row['region']],donors=[row['trace']['id']],scroll=row['scroll'])
        if rejected:result['meta']['compressionRejected']=rejected
        return result


class TrainingDataset:
    def __init__(self,offset,length):self.samples=Samples('train');self.offset=offset;self.length=length
    def __len__(self):return self.length
    def __getitem__(self,index):
        global_index=index+self.offset
        result=self.samples.generate(30000000+global_index,phase_step=global_index//8)
        return {k:v for k,v in result.items() if k!='meta'}
