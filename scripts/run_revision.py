#!/usr/bin/env python3
"""Select one prespecified residual per variant; evaluate frozen coefficients."""
from pathlib import Path
import argparse,json,time
import numpy as np
from run_attempt import load_arrays,choose,top1
from vcmr.data import load_records
from vcmr.ranking import suppress
from vcmr.evaluation import evaluate_predictions,paired_bootstrap,save_predictions_jsonl
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);p.add_argument('--mode',choices=['select','evaluate'],required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--configs',type=Path);a=p.parse_args()
    if a.attempt not in [2,3]:raise ValueError('Only attempts2/3')
    a.output.mkdir(parents=True,exist_ok=True);start=time.time();records=load_records(a.annotations)
    qids,rows,x,labels,names,eligible=load_arrays(a.predictions,a.features,records)
    candidate=f'attempt{a.attempt}_conditional_reference'
    if a.mode=='select':
        previous=json.loads((ROOT/f'runs/attempt{a.attempt-1}/selection/frozen_configs.json').read_text())
        configs=[]
        for c in previous['configs']:
            cc=c['coefficients']+[0]*(len(names)-len(c['coefficients']))
            configs.append({'name':c['name'],'coefficients':cc})
        fixed=next(c['coefficients'] for c in configs if c['name']==previous['strongest_control'])
        if a.attempt==2: variants=[('whole_local_contrast_control',4),('unconditional_local_control',5),('shared_slope_control',7),(candidate,6)]
        else: variants=[('neighborhood_unconditional_control',8),('neighborhood_shared_slope_control',10),(candidate,9)]
        grids={}
        for name,index in variants:
            options=[]
            for gamma in [0,.03125,.0625,.125,.25,.5,1,2]:
                coeff=fixed.copy();coeff[index]=gamma;options.append({'name':name,'coefficients':coeff})
            best,grid=choose(x,labels,options);configs.append(best);grids[name]=grid
        for c in configs:
            score,_=top1(x,labels,c['coefficients']);c['select_R1@0.5'],c['select_R1@0.7']=map(float,score)
        strongest=max([c for c in configs if c['name']!=candidate],key=lambda c:(c['select_R1@0.5'],c['select_R1@0.7'],-sum(abs(v) for v in c['coefficients'][1:])))
        result={'attempt':a.attempt,'candidate':candidate,'columns':names,'configs':configs,'strongest_control':strongest['name'],'grid':grids,'selection_queries':len(qids),'eligible_queries':eligible,'selection':'dev_select only, R1@.5 then R1@.7 then smaller weight; no finalval tuning','seconds':time.time()-start}
        path=a.output/'frozen_configs.json'
    else:
        frozen=json.loads(a.configs.read_text());results={};predictions={}
        # Existing predecessor controls may be loaded from identical shared predictions.
        for c in frozen['configs']:
            coeff=np.asarray(c['coefficients'],np.float32);pred={}
            for n,(qid,raw) in enumerate(rows):pred[qid]=suppress(raw,x[n,:len(raw)]@coeff,.7,100)
            predictions[c['name']]=pred;metrics=evaluate_predictions(pred,records)
            save_predictions_jsonl(a.output/(c['name']+'.jsonl'),pred)
            results[c['name']]={'coefficients':c['coefficients'],'metrics':metrics}
            print(json.dumps({'method':c['name'],'full':metrics['full']}),flush=True)
        strongest=frozen['strongest_control'];comparison=paired_bootstrap(predictions[strongest],predictions[candidate],records,iterations=2000)
        result={'attempt':a.attempt,'candidate':candidate,'queries':len(qids),'eligible_queries':eligible,'frozen_config_source':str(a.configs),'strongest_control_fixed_from_selection':strongest,'results':results,'candidate_minus_strongest_control_R1@0.5':comparison,'seconds':time.time()-start}
        result['passed_gate']=comparison['delta_b_minus_a_pp']>=.25 and comparison['ci95_pp'][0]>0
        path=a.output/'results.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['grid','results']},indent=2),flush=True)
if __name__=='__main__':main()
