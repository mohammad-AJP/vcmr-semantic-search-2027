from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_attempt import load_arrays
from vcmr.data import load_records

def main():
    out={}
    for split in ['dev_select','dev_confirm']:
        qids,rows,x,labels,names,eligible=load_arrays(ROOT/f'runs/demul_{split}/raw_predictions.jsonl',ROOT/f'runs/attempt1/features_{split}.h5',load_records(ROOT/f'data/tvr/splits/demul_{split}.jsonl'))
        base=x[:,:,0]+.25*x[:,:,1]
        corr=[]; changes=0; samegroupstd=[]; within_best=[]
        for n,(_,p) in enumerate(rows):
            if np.any(x[n,:,2]):
                corr.append(float(np.corrcoef(x[n,:,2],x[n,:,3])[0,1]))
                changes+=int(np.argmax(base[n]+.0625*x[n,:,2])!=np.argmax(base[n]+.0625*x[n,:,3]))
                groups={}
                for i,r in enumerate(p): groups.setdefault((r[0],round(r[2]-r[1],5)),[]).append(i)
                samegroupstd.extend(float(np.std(x[n,ix,2]-x[n,ix,3])) for ix in groups.values() if len(ix)>1)
        out[split]={'queries':len(qids),'eligible_queries':eligible,'raw_candidate_oracle_R1@0.5':100*float(labels[:,:,0].any(axis=1).mean()),'raw_candidate_oracle_R1@0.7':100*float(labels[:,:,1].any(axis=1).mean()),'median_correlation_conjunction_vs_surplus':float(np.median(corr)),'changed_top1_conjunction_vs_surplus':changes,'within_video_duration_reference_std_median':float(np.median(samegroupstd))}
    (ROOT/'runs/attempt1/diagnostics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
