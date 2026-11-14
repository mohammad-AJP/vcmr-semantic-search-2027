#!/usr/bin/env bash
set -euo pipefail
cd /home/mohammad/Documents/vcmr-semantic-search-2027
source scripts/environment.sh
mkdir -p runs/background
"$VCMR_PYTHON" - <<'START'
from pathlib import Path
import fcntl,os,subprocess
root=Path.cwd();job=root/'runs/background'
lock=(job/'pipeline.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    print('A background pipeline is already running. Read runs/background/status.json.');raise SystemExit(0)
fcntl.flock(lock,fcntl.LOCK_UN)
if (root/'docs/final_results.md').exists() and (root/'runs/final_validation/results.json').exists():
    print('All experiments are complete. Read docs/final_results.md.');raise SystemExit(0)
command=[os.environ['VCMR_PYTHON'],'-u']
if (root/'configs/attempt3_preregistration.json').exists() and (root/'scripts/build_neighborhood_features.py').exists():
    command+=['scripts/revision_pipeline.py','--attempt','3']
elif (root/'configs/attempt2_preregistration.json').exists():
    command+=['scripts/revision_pipeline.py','--attempt','2']
else:command+=['scripts/background_pipeline.py']
log=(job/'launcher.log').open('ab',buffering=0)
p=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(f'Detached pipeline PID: {p.pid}')
print('Status: '+str(job/'status.json'))
print('Log: '+str(job/'launcher.log'))
START
