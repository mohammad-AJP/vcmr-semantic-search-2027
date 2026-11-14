#!/usr/bin/env bash
# Standalone Linux setup: no grounded-prvr files or existing Conda required.
# Run: TVR_ROOT="$HOME/vcmr-tvr" bash setup_tvr_new_server.sh
set -euo pipefail
export TVR_ROOT="${TVR_ROOT:-$HOME/vcmr-tvr}"
mkdir -p "$TVR_ROOT"/{downloads,data/tvr,envs,tools,logs,tmp,.cache/conda,.cache/pip,manifests}
export CONDA_PKGS_DIRS="$TVR_ROOT/.cache/conda"
export CONDA_ENVS_PATH="$TVR_ROOT/envs"
export PIP_CACHE_DIR="$TVR_ROOT/.cache/pip"
export XDG_CACHE_HOME="$TVR_ROOT/.cache"
export TMPDIR="$TVR_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$TVR_ROOT/.cache/pycache"
exec 9>"$TVR_ROOT/download.lock"
flock -n 9 || { echo 'A TVR setup process is already running.'; exit 1; }
trap 'echo "FAILED: line $LINENO. Correct the error, then rerun the same script." >&2' ERR
[ "$(uname -s)" = Linux ] || { echo 'This script expects Linux.' >&2; exit 1; }
case "$(uname -m)" in x86_64|aarch64) ;; *) echo 'Supported: Linux x86_64 or aarch64.' >&2; exit 1;; esac
for cmd in curl sha256sum flock; do command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 1; }; done
fetch() {
  local url="$1" dest="$2"
  if [ ! -s "$dest" ]; then
    curl --fail --location --retry 5 --output "$dest.part" "$url"
    mv "$dest.part" "$dest"
  fi
}
if [ ! -x "$TVR_ROOT/tools/miniforge/bin/conda" ]; then
  installer="Miniforge3-Linux-$(uname -m).sh"
  release='https://github.com/conda-forge/miniforge/releases/latest/download'
  fetch "$release/$installer" "$TVR_ROOT/downloads/$installer"
  fetch "$release/$installer.sha256" "$TVR_ROOT/downloads/$installer.sha256"
  (cd "$TVR_ROOT/downloads"; sha256sum -c "$installer.sha256")
  bash "$TVR_ROOT/downloads/$installer" -b -p "$TVR_ROOT/tools/miniforge"
fi
conda_bin="$TVR_ROOT/tools/miniforge/bin/conda"
if [ ! -x "$TVR_ROOT/envs/tvr-data/bin/python" ]; then
  "$conda_bin" create -y --override-channels -c conda-forge \
    --prefix "$TVR_ROOT/envs/tvr-data" python=3.11 pip numpy h5py
fi
export PATH="$TVR_ROOT/envs/tvr-data/bin:$PATH"
python -m pip install 'gdown==6.4.2'

# Official temporal annotations, stored under the names used by our experiments.
ann='https://raw.githubusercontent.com/jayleicn/TVRetrieval/master/data'
fetch "$ann/tvr_train_release.jsonl" "$TVR_ROOT/data/tvr/official_train.jsonl"
fetch "$ann/tvr_val_release.jsonl" "$TVR_ROOT/data/tvr/official_val.jsonl"
fetch "$ann/tvr_video2dur_idx.json" "$TVR_ROOT/data/tvr/video2dur_idx.json"

# Download only TVR, not the whole multi-dataset Google Drive folder.
if [ ! -f "$TVR_ROOT/data/tvr/.mssl_extracted" ]; then
  gdown --no-cookies --continue '1RVG3qIy-nf9XdalMw8wpmWNSPcnOAaNK' \
    -O "$TVR_ROOT/downloads/tvr.zip"
  python - <<'PY'
from pathlib import Path
import os,zipfile
r=Path(os.environ['TVR_ROOT']);dest=(r/'data/tvr/mssl').resolve();dest.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(r/'downloads/tvr.zip') as z:
    for member in z.infolist():
        if not (dest/member.filename).resolve().is_relative_to(dest):
            raise RuntimeError('Unsafe zip member: '+member.filename)
    # Extraction reads every member and checks its ZIP CRC.
    z.extractall(dest)
(r/'data/tvr/.mssl_extracted').touch()
PY
fi

# This is the additional feature package used by our published baseline.
if [ ! -f "$TVR_ROOT/data/tvr/.conquer_extracted" ]; then
  gdown --no-cookies --continue '1DFnMNH-oi6-cZbl1coXqa_KjtsIsObxG' \
    -O "$TVR_ROOT/downloads/tvr_feature_release.tar.gz"
  python - <<'PY'
from pathlib import Path
import os,tarfile
r=Path(os.environ['TVR_ROOT']);archive=r/'downloads/tvr_feature_release.tar.gz'
if archive.stat().st_size != 22318784064:
    raise RuntimeError('Archive size differs from the verified release. Inspect the download before proceeding.')
with tarfile.open(archive,'r:gz') as t:
    t.extractall(r/'data/tvr',filter='data')
(r/'data/tvr/.conquer_extracted').touch()
PY
fi

# Independent copies; preserves both original release layout and experiment names.
python - <<'PY'
from pathlib import Path
import json,os,shutil
import h5py,numpy as np
r=Path(os.environ['TVR_ROOT']);d=r/'data/tvr'
for source_name,dest_name in [('new_clip_vit_32_tvr_vid_features.hdf5','clip_video.h5'),('clip_ViT_B_32_tvr_query_feat.hdf5','clip_query.h5')]:
    matches=list((d/'mssl').rglob(source_name))
    if len(matches)!=1: raise RuntimeError(f'Expected one {source_name}; found {matches}')
    dest=d/dest_name
    if not dest.exists():
        temporary=dest.with_suffix('.partial.h5');shutil.copy2(matches[0],temporary);temporary.replace(dest)
records=[];report={'root':str(r),'checks':{}}
for name,expected in [('official_train.jsonl',87175),('official_val.jsonl',10895)]:
    rows=[json.loads(line) for line in (d/name).open()]
    if len(rows)!=expected: raise RuntimeError(f'{name}: expected {expected} records, found {len(rows)}')
    if any(not(0<=row['ts'][0]<row['ts'][1]) for row in rows):raise RuntimeError('Invalid annotation interval')
    report['checks'][name]={'queries':len(rows),'videos':len({x['vid_name'] for x in rows})};records.extend(rows)
json.loads((d/'video2dur_idx.json').read_text())
with h5py.File(d/'clip_video.h5','r') as v,h5py.File(d/'clip_query.h5','r') as q:
    missing={x['vid_name'] for x in records}-set(v)
    if missing:raise RuntimeError(f'Missing CLIP video features for {len(missing)} annotated videos')
    for key in list(v)[:10]:
        x=v[key][()]
        if x.ndim!=2 or x.shape[1]!=512 or not np.isfinite(x).all():raise RuntimeError('Invalid video feature sample')
    for key in list(q)[:10]:
        x=q[key][()]
        if x.shape!=(512,) or not np.isfinite(x).all():raise RuntimeError('Invalid query feature sample')
    report['checks']['clip_video_entries']=len(v);report['checks']['clip_query_entries']=len(q)
for name in ['video_feature','sub_query_feature','first_stage_trained_model','data']:
    p=d/'tvr_feature_release'/name
    if not p.is_dir() or not any(p.iterdir()):raise RuntimeError(f'Missing baseline component: {p}')
report['scope']='Annotation counts, video-ID coverage, sample feature shapes/finiteness, and baseline directory presence; not complete temporal/query-ID reconciliation.'
(r/'manifests/download_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
python -m pip freeze > "$TVR_ROOT/manifests/pip_freeze.txt"
"$conda_bin" list --explicit --prefix "$TVR_ROOT/envs/tvr-data" > "$TVR_ROOT/manifests/conda_explicit.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$TVR_ROOT/DOWNLOAD_COMPLETE"
echo "SUCCESS. Dataset: $TVR_ROOT/data/tvr"
