#!/usr/bin/env python3
"""One final official-validation evaluation of all frozen development variants."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from run_attempt import load_arrays
from vcmr.data import load_records
from vcmr.ranking import suppress
from vcmr.evaluation import evaluate_predictions,paired_bootstrap,save_predictions_jsonl
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,required=True);a=ap.parse_args()
    frozen_path=ROOT/f'runs/attempt{a.attempt}/selection/frozen_configs.json'
    frozen=json.loads(frozen_path.read_text());out=ROOT/'runs/final_validation';out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Final validation already complete; do not tune and repeat')
    records=load_records(ROOT/'data/tvr/official_val.jsonl');start=time.time()
    qids,rows,x,labels,names,eligible=load_arrays(ROOT/'runs/demul_val/raw_predictions.jsonl',ROOT/f'runs/attempt{a.attempt}/features_official_val.h5',records)
    strongest=frozen['strongest_control'];candidate_names=[c['name'] for c in frozen['configs'] if c['name'].startswith('attempt')]
    report={'n_queries':len(qids),'eligible_queries':eligible,'frozen_configs':str(frozen_path),'frozen_configs_sha256':hashlib.sha256(frozen_path.read_bytes()).hexdigest(),'strongest_control_fixed_from_selection':strongest,'protocols':{}}
    for limit in [600,100]:
        protocol=f'{limit}_before_nms';results={};predictions={};memo={}
        for c in frozen['configs']:
            key=tuple(c['coefficients']);name=c['name'];coeff=np.asarray(key,np.float32)
            if key in memo:pred,metrics=memo[key]
            else:
                pred={}
                for n,(qid,raw) in enumerate(rows):
                    count=min(len(raw),limit);pred[qid]=suppress(raw[:count],x[n,:count]@coeff,.7,100)
                metrics=evaluate_predictions(pred,records);memo[key]=(pred,metrics)
            predictions[name]=pred;results[name]={'coefficients':c['coefficients'],'metrics':metrics}
            save_predictions_jsonl(out/protocol/(name+'.jsonl'),pred)
            print(json.dumps({'protocol':protocol,'method':name,'full':metrics['full']}),flush=True)
        comparisons={name:{control:paired_bootstrap(predictions[control],predictions[name],records,iterations=2000) for control in dict.fromkeys(['demul','whole_query_control',strongest])} for name in candidate_names}
        report['protocols'][protocol]={'results':results,'candidate_comparisons_R1@0.5':comparisons}
        (out/(protocol+'.json')).write_text(json.dumps(report['protocols'][protocol],indent=2)+'\n')
    report['seconds']=time.time()-start
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('Final evaluation complete',flush=True)
if __name__=='__main__':main()
