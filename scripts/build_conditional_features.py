#!/usr/bin/env python3
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,os,time
import numpy as np
import h5py
from vcmr.conditional import local_null_signals,flank_contrast
from vcmr.association import interval_means
from vcmr.ranking import read_prediction_rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix('.partial.h5');start=time.time()
    cols=['base_log','whole_query','conjunction','surplus','whole_local_contrast','unconditional_local_excess','conditional_local_excess','shared_slope_control']
    stats={'queries':0,'eligible_queries':0,'eligible_video_pairs':0,'columns':cols,'surrogates':16,'tau':.5}
    with h5py.File(root/'data/tvr/clip_video.h5','r') as vf,h5py.File(root/f'data/tvr/phrases_{a.split}.h5','r') as qf,h5py.File(a.input,'r') as old,h5py.File(tmp,'w') as out:
        out.attrs['columns']=json.dumps(cols);out.attrs['feature_version']='conditional_v2';out.attrs['baseline_sha256']=hashlib.sha256(a.baseline.read_bytes()).hexdigest()
        @lru_cache(maxsize=5000)
        def video(name):
            x=vf[name][()].astype(np.float32);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
        for qid,pred in read_prediction_rows(a.baseline):
            ds=old[qid];indices=json.loads(ds.attrs['phrase_indices']);gate=bool(ds.attrs['eligible'])
            f=np.zeros((len(pred),len(cols)),np.float32);f[:,:4]=ds[()]
            if gate:
                emb=qf[qid][()];groups={}
                for i,p in enumerate(pred): groups.setdefault(p[0],[]).append(i)
                for name,loc in groups.items():
                    if name not in vf: continue
                    x=video(name);scores=emb@x.T
                    starts=np.floor(np.asarray([pred[i][1] for i in loc])/1.5+1e-6).astype(int)
                    ends=np.ceil(np.asarray([pred[i][2] for i in loc])/1.5-1e-6).astype(int)
                    coverage=(np.minimum(ends,len(x))-starts)/np.maximum(ends-starts,1)
                    valid=(starts>=0)&(starts<len(x))&(ends>starts)&(coverage>=.8)
                    if not valid.any() or not np.all(scores[indices].std(axis=1)>.001):continue
                    use=np.asarray(loc)[valid];aa=starts[valid];bb=np.minimum(ends[valid],len(x))
                    seed=int.from_bytes(hashlib.sha256(f'2027:{qid}:{name}'.encode()).digest()[:8],'little')
                    shifts=np.random.default_rng(seed).integers(0,len(x),size=(16,len(indices)))
                    sig=local_null_signals(scores[indices],scores[0],shifts)
                    f[use,4]=np.clip(flank_contrast(sig['holistic_z'],aa,bb),-3,3)
                    f[use,5]=np.clip(interval_means(sig['unconditional'],aa,bb),-3,3)
                    f[use,6]=np.clip(interval_means(sig['conditional'],aa,bb),-3,3)
                    f[use,7]=np.clip(interval_means(sig['shared_slope'],aa,bb),-3,3)
                    stats['eligible_video_pairs']+=1
            o=out.create_dataset(qid,data=f,compression='lzf');o.attrs['eligible']=gate;o.attrs['phrase_indices']=json.dumps(indices)
            stats['queries']+=1;stats['eligible_queries']+=int(gate)
            if stats['queries']%500==0:print(json.dumps({'queries':stats['queries'],'seconds':time.time()-start}),flush=True)
        stats['seconds']=time.time()-start;out.attrs['stats']=json.dumps(stats)
    os.replace(tmp,a.output);a.output.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
