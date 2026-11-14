#!/usr/bin/env python3
"""Extract fixed, label-free reranking features for the first research attempt."""
from pathlib import Path
import argparse, hashlib, json, os, time
from functools import lru_cache
import h5py
import numpy as np
from vcmr.association import association_features, select_phrase_indices, eligible, interval_means
from vcmr.ranking import read_prediction_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--split", required=True, choices=["dev_select","dev_confirm","official_val"])
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--surrogates", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite completed feature cache")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parses = {str(x["desc_id"]):x for x in map(json.loads, (args.root/f"data/tvr/parses_{args.split}.jsonl").open())}
    video_file = h5py.File(args.root/"data/tvr/clip_video.h5", "r")
    query_file = h5py.File(args.root/f"data/tvr/phrases_{args.split}.h5", "r")
    @lru_cache(maxsize=5000)
    def video(name):
        x = video_file[name][()].astype(np.float32)
        x /= np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
        return x
    tmp = args.output.with_suffix(".partial.h5")
    start = time.time()
    stats = {"queries":0,"eligible_queries":0,"eligible_video_pairs":0,"unsupported_video_pairs":0,
             "rows":0,"surrogates":args.surrogates,"tau":args.tau,
             "timing":"nominal 1.5 seconds per CLIP frame; no guessed actor tracks",
             "columns":["base_log","whole_query","conjunction","surplus"],
             "gate":"simultaneous positive query; at least two deduplicated nonholistic phrases; nonflat traces",
             "null":"independent circular shifts; same-length all-window maximum; mean across surrogates"}
    with h5py.File(tmp,"w") as out:
        out.attrs["feature_version"]="association_v1"
        out.attrs["split"]=args.split
        out.attrs["baseline_path"]=str(args.baseline.resolve())
        out.attrs["baseline_sha256"]=hashlib.sha256(args.baseline.read_bytes()).hexdigest()
        out.attrs["columns"]=json.dumps(stats["columns"])
        for qid,predictions in read_prediction_rows(args.baseline):
            if args.limit is not None and stats["queries"]>=args.limit:
                break
            parse,emb = parses[qid],query_file[qid][()]
            indices=select_phrase_indices(parse,emb)
            gate=eligible(parse,indices)
            f=np.zeros((len(predictions),4),np.float32)
            f[:,0]=np.log(np.maximum([p[3] for p in predictions],1e-30))
            groups={}
            for i,p in enumerate(predictions):
                groups.setdefault(str(p[0]),[]).append(i)
            for name,loc in groups.items():
                if name not in video_file:
                    stats["unsupported_video_pairs"]+=1
                    continue
                x=video(name)
                scores=emb @ x.T
                # Official moments use [start index*1.5,(end index+1)*1.5].
                starts=np.floor(np.asarray([predictions[i][1] for i in loc])/1.5+1e-6).astype(int)
                ends=np.ceil(np.asarray([predictions[i][2] for i in loc])/1.5-1e-6).astype(int)
                coverage=(np.minimum(ends,len(x))-starts)/np.maximum(ends-starts,1)
                valid=(starts>=0)&(starts<len(x))&(ends>starts)&(coverage>=0.8)
                if not np.any(valid):
                    stats["unsupported_video_pairs"]+=1
                    continue
                use=np.asarray(loc)[valid]
                a=starts[valid]; b=np.minimum(ends[valid],len(x))
                f[use,1]=interval_means(scores[0],a,b)/0.05
                if gate and np.all(scores[indices].std(axis=1)>0.001):
                    seed=int.from_bytes(hashlib.sha256(f"2027:{qid}:{name}".encode()).digest()[:8],"little")
                    rng=np.random.default_rng(seed)
                    shifts=rng.integers(0,len(x),size=(args.surrogates,len(indices)))
                    feats=association_features(scores[indices],a,b,shifts,args.tau)
                    f[use,2]=np.clip(feats["conjunction"],-3,3)
                    f[use,3]=np.clip(feats["surplus"],-3,3)
                    stats["eligible_video_pairs"]+=1
            ds=out.create_dataset(qid,data=f,compression="lzf")
            ds.attrs["phrase_indices"]=json.dumps(indices)
            ds.attrs["eligible"]=gate
            stats["queries"]+=1
            stats["eligible_queries"]+=int(gate)
            stats["rows"]+=len(predictions)
            if stats["queries"]%200==0:
                print(json.dumps({"queries":stats["queries"],"seconds":round(time.time()-start,1)}),flush=True)
        stats["seconds"]=time.time()-start
        stats["cache"]=str(args.output)
        out.attrs["stats"]=json.dumps(stats)
    os.replace(tmp,args.output)
    args.output.with_suffix(".json").write_text(json.dumps(stats,indent=2)+"\n")
    print(json.dumps(stats,indent=2),flush=True)

if __name__=="__main__":
    main()
