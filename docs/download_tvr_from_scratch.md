# Download the TVR experiment data on a fresh Linux server

This downloads the annotations and precomputed features used by our experiment.
It is a data setup, not a model training installation or a raw-video download.
No existing grounded-prvr checkout or existing Conda installation is required.
Supports Linux x86_64 and aarch64. Reserve about 120 GB for archives, extracted
packages, independent CLIP copies, and the environment. No GPU is needed.

## 1. Install the small operating-system prerequisites

For Ubuntu/Debian (administrator permissions required only for this step):

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates util-linux coreutils
```

## 2. Save the standalone installer

Save the following as `setup_tvr_new_server.sh` on the new server. It installs
Miniforge inside the chosen root without modifying your shell configuration,
creates a separate `tvr-data` Conda environment, and downloads only TVR.

```bash
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
```

The installer source is also available at `scripts/setup_tvr_new_server.sh` in
this workspace. You may copy that one file to another server; no other project
files are required. For example, from the current machine:

```bash
scp /home/mohammad/Documents/vcmr-semantic-search-2027/scripts/setup_tvr_new_server.sh \
    YOUR_USER@NEW_SERVER:~/setup_tvr_new_server.sh
```

## 3. Run it in a detached background session

On the new server, choose the storage location. Replace `$HOME/vcmr-tvr` with a
large data disk path if appropriate.

```bash
export TVR_ROOT="$HOME/vcmr-tvr"
mkdir -p "$TVR_ROOT/logs"
nohup setsid bash "$HOME/setup_tvr_new_server.sh" \
    > "$TVR_ROOT/logs/setup.log" 2>&1 < /dev/null &
echo $! > "$TVR_ROOT/logs/setup.pid"
```

The process continues if the terminal or SSH session disconnects. It does not
survive machine shutdown. The lock rejects concurrent duplicate installers.
Completed extraction stages are skipped when rerunning the installer; gdown
resumes partial archive downloads.

## 4. Monitor and verify

```bash
export TVR_ROOT="$HOME/vcmr-tvr"
tail -f "$TVR_ROOT/logs/setup.log"
```

Stop following the log with Ctrl+C; that does not stop the detached job.
After completion:

```bash
cat "$TVR_ROOT/DOWNLOAD_COMPLETE"
cat "$TVR_ROOT/manifests/download_verification.json"
```

Expected annotation counts: 87,175 training queries / 17,435 videos, and 10,895
validation queries / 2,179 videos. The supplied CLIP files inspected locally
contain 19,614 video entries and 98,069 query entries. That query-feature count
is one smaller than the sum of official annotation records; our experiment
handles the duplicated training caption via an explicit ID mapping. These
basic download checks do not replace full query-ID/timestamp reconciliation.

## 5. Use the environment

```bash
export TVR_ROOT="$HOME/vcmr-tvr"
source "$TVR_ROOT/tools/miniforge/etc/profile.d/conda.sh"
conda activate "$TVR_ROOT/envs/tvr-data"
```

## Resulting locations

```text
~/vcmr-tvr/
  data/tvr/
    official_train.jsonl
    official_val.jsonl
    video2dur_idx.json
    clip_video.h5
    clip_query.h5
    mssl/                         # original TVR feature-package layout
      FeatureData/
      TextData/
    tvr_feature_release/          # CONQUER/DeMUL feature package
      video_feature/
      sub_query_feature/
      first_stage_trained_model/
      data/
  downloads/                      # retained archives and Miniforge installer
  envs/tvr-data/
  tools/miniforge/
  logs/setup.log
  manifests/download_verification.json
  DOWNLOAD_COMPLETE
```

## Download sources and failures

- [TVRetrieval official annotations](https://github.com/jayleicn/TVRetrieval/tree/master/data).
- [MS-SL official feature release](https://github.com/HuiGuanLab/ms-sl#ms-sl-on-tvr). The TVR-only Google Drive file ID was resolved from the author's public folder on 2026-11-14: `1RVG3qIy-nf9XdalMw8wpmWNSPcnOAaNK`.
- [DeMUL data preparation](https://github.com/iLearn-Lab/TCSVT26-DeMUL#data-preparation) links the CONQUER archive, ID `1DFnMNH-oi6-cZbl1coXqa_KjtsIsObxG`.
- [Miniforge installation](https://github.com/conda-forge/miniforge).
- [gdown download/resume documentation](https://github.com/wkentaro/gdown).

If Google Drive reports a quota error, keep partial files and retry later.
Alternatively, download the same public files through a browser and place them
in the indicated `downloads` directory, then rerun the installer. Do not treat
an HTML quota/login page as a successful archive. Extraction and validation
must succeed before `DOWNLOAD_COMPLETE` is written.

Verified locally: author-folder file IDs, current archive layout, annotation
counts, current feature shapes/video-ID coverage, Bash syntax, and embedded
Python syntax. The complete download/install was not rerun on another server
from this session.
