#!/usr/bin/env python3
"""Start a command in its own session with a durable log; no terminal required."""
import argparse, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
root=Path(__file__).resolve().parents[1];d=root/'runs/background';d.mkdir(exist_ok=True,parents=True)
if not a.command: p.error('A command is required')
log=(d/(a.name+'.log')).open('ab',buffering=0)
proc=subprocess.Popen(a.command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=os.environ.copy())
(d/(a.name+'.pid')).write_text(str(proc.pid)+'\n');print('Detached PID',proc.pid,'log',log.name)
