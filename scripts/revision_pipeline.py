#!/usr/bin/env python3
from pathlib import Path
import argparse,fcntl,os,traceback,json,subprocess
from background_pipeline import ROOT,JOB,run,status
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);a=p.parse_args()
if a.attempt not in [2,3]:raise ValueError('Maximum three substantive attempts')
lock=(JOB/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    run(f'tests_attempt{a.attempt}',['-m','pytest','-q','tests'])
    for split in ['dev_select','dev_confirm']:
        dest=ROOT/f'runs/attempt{a.attempt}/features_{split}.h5'
        script='scripts/build_conditional_features.py' if a.attempt==2 else 'scripts/build_neighborhood_features.py'
        run(f'attempt{a.attempt}_features_'+split,[script,'--split',split,'--baseline',ROOT/f'runs/demul_{split}/raw_predictions.jsonl','--input',ROOT/f'runs/attempt{a.attempt-1}/features_{split}.h5','--output',dest],dest)
    frozen=ROOT/f'runs/attempt{a.attempt}/selection/frozen_configs.json'
    for split,mode,out in [('dev_select','select',frozen),('dev_confirm','evaluate',ROOT/f'runs/attempt{a.attempt}/confirmation/results.json')]:
        run(f'attempt{a.attempt}_'+mode,['scripts/run_revision.py','--attempt',a.attempt,'--mode',mode,'--annotations',ROOT/f'data/tvr/splits/demul_{split}.jsonl','--predictions',ROOT/f'runs/demul_{split}/raw_predictions.jsonl','--features',ROOT/f'runs/attempt{a.attempt}/features_{split}.h5','--output',out.parent,'--configs',frozen],out)
    status('complete',f'attempt{a.attempt}_confirmation',result=str(out))
    if a.attempt==3:
        r=json.loads(out.read_text())
        decision={'attempt':3,'decision':'passed_gate' if r['passed_gate'] else 'failed_prespecified_advantage_gate','evidence':r['candidate_minus_strongest_control_R1@0.5'],'next':'Stop substantive development after three attempts; evaluate all frozen variants on final validation.'}
        (ROOT/'runs/attempt3/decision.json').write_text(json.dumps(decision,indent=2)+'\n')
        fcntl.flock(lock,fcntl.LOCK_UN)
        subprocess.run([str(ROOT/'envs/vcmr/bin/python'),'-u','scripts/final_pipeline.py','--attempt','3'],cwd=ROOT,check=True)
except BaseException as e:
    status('failed',f'attempt{a.attempt}',error=repr(e));traceback.print_exc();raise
