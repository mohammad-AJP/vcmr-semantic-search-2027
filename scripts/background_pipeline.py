#!/usr/bin/env python3
"""Detached, restartable research stages. Failed commands stop dependent work."""
from pathlib import Path
import argparse, datetime, fcntl, json, os, subprocess, sys, traceback
ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'envs/vcmr/bin/python'
JOB=ROOT/'runs/background'
JOB.mkdir(parents=True,exist_ok=True)
def status(state,step,**extra):
    p=JOB/'status.json'; tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,step=step,pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')
    tmp.replace(p)
def run(name,args,artifact=None):
    marker=JOB/(name+'.done.json')
    if marker.exists() and (artifact is None or Path(artifact).exists()):
        print('Already completed:',name,flush=True); return
    if artifact is not None and Path(artifact).exists() and not marker.exists():
        target=Path(artifact)
        # A final renamed HDF5 or parseable JSON can survive a crash just before
        # its completion marker. Verify before accepting that completed artifact.
        if target.suffix=='.h5':
            import h5py
            with h5py.File(target,'r') as h:
                stats=json.loads(h.attrs['stats'])
                if stats['queries']!=len(h):raise RuntimeError('Incomplete recovered HDF5')
        elif target.suffix=='.json':json.loads(target.read_text())
        else:raise RuntimeError('Existing unmarked artifact requires review: '+str(target))
        marker.write_text(json.dumps(dict(command=list(map(str,args)),artifact=str(target),recovered=True))+'\n')
        print('Recovered completed artifact:',name,flush=True);return
    status('running',name,log=str(JOB/(name+'.log')))
    print('Starting:',name,flush=True)
    with (JOB/(name+'.log')).open('a',buffering=1) as log:
        subprocess.run([str(PY),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    if artifact is not None and not Path(artifact).exists(): raise RuntimeError('Missing artifact: '+str(artifact))
    marker.write_text(json.dumps(dict(command=list(map(str,args)),artifact=str(artifact)))+'\n')
def main():
    lock=(JOB/'pipeline.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Another pipeline already holds the lock')
    (JOB/'pipeline.pid').write_text(str(os.getpid())+'\n')
    try:
        run('tests',['-m','pytest','-q','tests'])
        for split in ['dev_select','dev_confirm']:
            dest=ROOT/f'runs/attempt1/features_{split}.h5'
            run('features_'+split,['scripts/build_association_features.py','--split',split,'--baseline',ROOT/f'runs/demul_{split}/raw_predictions.jsonl','--output',dest],dest)
        common=['--annotations',ROOT/'data/tvr/splits/demul_dev_select.jsonl','--predictions',ROOT/'runs/demul_dev_select/raw_predictions.jsonl','--features',ROOT/'runs/attempt1/features_dev_select.h5']
        frozen=ROOT/'runs/attempt1/selection/frozen_configs.json'
        run('attempt1_select',['scripts/run_attempt.py','--mode','select',*common,'--output',frozen.parent],frozen)
        report=ROOT/'runs/attempt1/confirmation/results.json'
        run('attempt1_confirm',['scripts/run_attempt.py','--mode','evaluate','--annotations',ROOT/'data/tvr/splits/demul_dev_confirm.jsonl','--predictions',ROOT/'runs/demul_dev_confirm/raw_predictions.jsonl','--features',ROOT/'runs/attempt1/features_dev_confirm.h5','--output',report.parent,'--configs',frozen],report)
        status('complete','attempt1_confirmation',result=str(report),next_action='Scientifically review confirmation evidence before a substantive revision or final validation.')
        print('Completed through attempt1 confirmation',flush=True)
    except BaseException as e:
        status('failed','see_log',error=repr(e)); traceback.print_exc(); raise
if __name__=='__main__': main()
