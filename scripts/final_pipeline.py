#!/usr/bin/env python3
"""Build label-free final features, evaluate frozen methods, write report."""
import argparse,fcntl,json,traceback
from background_pipeline import ROOT,JOB,run,status
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);a=p.parse_args()
if a.attempt not in [1,2,3]:raise ValueError('At most three attempts')
lock=(JOB/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    # Snapshot proves every method's weights were frozen before this evaluation.
    frozen=ROOT/f'runs/attempt{a.attempt}/selection/frozen_configs.json'
    out=ROOT/'runs/final_validation';out.mkdir(parents=True,exist_ok=True)
    snapshot=out/'frozen_configs_before_validation.json'
    if not snapshot.exists():snapshot.write_text(frozen.read_text())
    for attempt in range(1,a.attempt+1):
        dest=ROOT/f'runs/attempt{attempt}/features_official_val.h5'
        args=['--split','official_val','--baseline',ROOT/'runs/demul_val/raw_predictions.jsonl','--output',dest]
        if attempt==1:script='scripts/build_association_features.py'
        else:
            script='scripts/build_conditional_features.py' if attempt==2 else 'scripts/build_neighborhood_features.py'
            args+=['--input',ROOT/f'runs/attempt{attempt-1}/features_official_val.h5']
        run(f'final_features_attempt{attempt}',[script,*args],dest)
    run('final_evaluation',['scripts/final_evaluate.py','--attempt',a.attempt],ROOT/'runs/final_validation/results.json')
    run('final_report',['scripts/write_final_report.py'],ROOT/'docs/final_results.md')
    status('complete','all_experiments',result=str(ROOT/'docs/final_results.md'),substantive_attempts=a.attempt)
except BaseException as e:
    status('failed','final_pipeline',error=repr(e));traceback.print_exc();raise
