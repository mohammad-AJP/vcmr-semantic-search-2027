"""Local conditional temporal-surrogate evidence; a heuristic, not a p-value."""
import numpy as np
from .association import soft_conjunction, interval_means


def local_null_signals(similarities, holistic, shifts, tau=.5):
    s=np.asarray(similarities,dtype=np.float64)
    if s.ndim!=2 or s.shape[0]<2: raise ValueError('Need at least two predicates')
    h=np.asarray(holistic,dtype=np.float64)
    if h.shape!=(s.shape[1],): raise ValueError('Holistic time length differs')
    z=(s-s.mean(axis=1,keepdims=True))/np.maximum(s.std(axis=1,keepdims=True),.01)
    h=h-h.mean(); h=h/max(h.std(),.01)
    fitted=(z@h/(h@h+1e-12))[:,None]*h[None,:]
    residual=z-fitted
    observed=soft_conjunction(z,tau)
    unconditional=[]; conditional=[]
    for row in shifts:
        unconditional.append(soft_conjunction(np.stack([np.roll(t,int(k)) for t,k in zip(z,row)]),tau))
        conditional.append(soft_conjunction(fitted+np.stack([np.roll(t,int(k)) for t,k in zip(residual,row)]),tau))
    return {'unconditional':observed-np.mean(unconditional,axis=0),
            'conditional':observed-np.mean(conditional,axis=0),'holistic_z':h,
            'fitted':fitted,'residual':residual,'observed':observed,
            'shared_slope':observed-fitted.mean(axis=0)-np.mean([soft_conjunction(np.stack([np.roll(t,int(k)) for t,k in zip(residual,row)]),tau) for row in shifts])}


def flank_contrast(signal,starts,ends):
    """Established inside/outside contrast used only as a control."""
    s=np.asarray(signal);a=np.asarray(starts);b=np.asarray(ends);length=b-a
    left=np.maximum(0,a-length);right=np.minimum(len(s),b+length)
    prefix=np.r_[0.,np.cumsum(s)]
    outside_n=(a-left)+(right-b)
    outside=(prefix[a]-prefix[left]+prefix[right]-prefix[b])/np.maximum(outside_n,1)
    return np.where(outside_n>0,interval_means(s,a,b)-outside,0.)
