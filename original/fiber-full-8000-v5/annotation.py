"""Partial fibre-instance labels owned only by accepted >=30-point source traces.

All skeleton branches (including rejected fragments) compete for ownership, so
an accepted trace cannot label an entire connected bundle by region growing.
Teacher-derived support remains pseudo ground truth, not measured fibre anatomy.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

def build(predictions, traces):
    shape=predictions.shape[:3]
    semantic=np.full(shape,255,'u1')
    semantic[predictions[...,3]>=242]=0
    instances=np.zeros(shape,'u2')
    centre=np.zeros(shape,'u1')
    if len(traces)>=65535:raise ValueError('Region exceeds uint16 instance capacity')
    for channel,family in enumerate(('vertical','horizontal')):
        selected=[(i+1,t) for i,t in enumerate(traces) if t['class']==family]
        if not selected:continue
        probability=predictions[...,channel].astype('f4')+.5*predictions[...,2]
        support=probability>=153
        skeleton=skeletonize(support,method='lee')
        coords=np.argwhere(skeleton)
        del probability
        dense=[];ids=[]
        for identity,trace in selected:
            points=np.asarray(trace['lineLocalXYZ'])[:,::-1]
            dense.append(points);ids.extend([identity]*len(points))
        distance,nearest=cKDTree(np.concatenate(dense)).query(coords,workers=2)
        seed_ids=np.zeros(shape,'u2')
        accepted=distance<=1.5
        seed_ids[tuple(coords[accepted].T)]=np.asarray(ids,dtype='u2')[nearest[accepted]]
        # The nearest of ALL skeleton points owns a voxel, not just accepted seeds.
        indices=distance_transform_edt(~skeleton,return_distances=False,return_indices=True)
        owner=seed_ids[tuple(indices)]
        owned=support&(owner>0)
        del indices,seed_ids,skeleton
        overlaps=owned&(instances>0)
        semantic[owned]=channel+1
        semantic[overlaps]=255
        instances[owned]=owner[owned]
        instances[overlaps]=0
        centre[tuple(coords[accepted].T)]=channel+1
        del owner,support
    # Intersection ownership is not an instance match. Exclude from instance loss.
    uncertain_intersection=predictions[...,2]>=128
    semantic[uncertain_intersection]=255
    instances[uncertain_intersection]=0
    centre[uncertain_intersection]=0
    return {'semantic':semantic,'instances':instances,'centre':centre}
