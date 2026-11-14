# End-to-end VCMR research-agent prompt

You are acting as a research scientist and research engineer. Carry this project from literature review and experimental design through implementation, controlled experiments, failure analysis, and a final reproducible report. Do not stop at giving me suggestions, a plan, or unexecuted code.

## 1. Research objective

I want to develop a new system for **video corpus moment retrieval (VCMR)**. Given a natural-language query and a database of videos, the system must identify both the correct videos and the relevant temporal intervals within them.

For example, given:

> “A man standing in a black suit.”

The system should understand the entities, action/state, attributes, and their relationships, then return a globally ranked list of:

```text
(video_id, start_time_seconds, end_time_seconds, relevance_score)
```

It may also produce a video ranking derived from its moment predictions. A video-level match without correct temporal localization does not count as a successful VCMR prediction. Do not substitute partially relevant video retrieval, ordinary text-to-video retrieval, or grounding within an already supplied correct video for the requested task.

The required research contributions concern **both text semantic parsing and independent search across the entire video corpus defined by the evaluation protocol**. Develop a complete original approach, with the explicit objective of beating the current strongest comparable VCMR results. Aim for a defensible contribution suitable for a strong computer-vision venue, potentially CVPR 2027. Better-than-SOTA performance is the goal to pursue and measure, not a result to assume or promise.

The system must discover its own relevant videos and temporal intervals from the corpus. A semantic wrapper around an existing retriever, or an improvement confined to reranking its candidate list, does not fulfill this project. Design the parser and search algorithm together so that linguistic structure changes what the system searches for, where it searches, and how it establishes a matching moment.

## 2. Innovation is a central requirement

**Do not copy an existing paper’s proposed mechanism and present it as our idea. This is especially important for semantic parsing, video retrieval, temporal grounding, database search, and ranking algorithms.**

Read the literature to understand existing mechanisms and identify actual gaps. Do not assemble familiar modules, rename the combination, and call that sufficient novelty. Specifically, using an LLM, extracting subject–verb–object triples, constructing a scene graph, applying attention, using CLIP, adding an index, reranking candidates, subtracting background similarity, using contrastive loss, or applying a graph/optimization method is not automatically a novel contribution.

Established software, encoders, mathematical tools, and reference baselines may be used as clearly attributed infrastructure and controls. They must not be represented as our inventions. Reproducing a paper as a baseline is different from copying its contribution into our proposed method without a new mechanism.

Do not restrict the research to frozen encoders, small residual scoring terms, linear score mixtures, fixed proposals, or a baseline architecture. You may develop a new representation, learning objective, corpus organization, indexing strategy, search procedure, temporal inference method, or jointly trained system. Choose the design from the scientific hypothesis and hardware evidence. A staged retrieval architecture is allowed when our method controls corpus access and moment discovery; simply adding stages to another system is not sufficient novelty. Existing methods are comparisons and infrastructure, not the boundaries of our search space.

You may draw inspiration from mathematics, physics, statistics, information theory, databases, information retrieval, optimization, algorithms, causal inference, or other fields. For every borrowed concept:

- Identify its origin and the established part.
- Explain the specific VCMR failure it addresses.
- Derive the mechanism that connects that concept to our problem.
- State what, if anything, is new in that connection or formulation.
- Identify assumptions and counterexamples.
- Define an experiment that can distinguish the proposed contribution from a simpler application of the existing concept.

Do not use a physics metaphor, a new acronym, or complicated notation as a substitute for innovation. Do not claim that a targeted literature search proves complete originality. If close prior art is discovered, explain the overlap and revise or reject the novelty claim before investing in a large implementation.

## 3. Work autonomously and communicate

First present a concrete step-by-step plan. Then execute it without stopping for permission after every step. I authorize implementation, dependency installation in the isolated environment, meaningful tests, dataset preparation, model inference, and the experiments described below.

After each meaningful stage, write a short update in the chat stating:

1. What you completed.
2. What evidence or output you obtained.
3. What you learned or what failed.
4. What comes next.

Show an actual parser example when parsing works. Report measured experiment results as they become available. Do not disappear for a long period during a run without a progress update. Ask a question only when genuinely missing information blocks work that cannot reasonably be inferred from the repository, datasets, current artifacts, or these instructions. Continue independent work while waiting where possible.

## 4. Set up a fresh device, isolate everything, and download TVR

This prompt is intended to run independently on a different device. Do not assume any previous repository, research directory, data, environment, checkpoint, script, or experiment result exists. All paths below are portable locations chosen on the new device. **The canonical dataset directory for this experiment is `$RESEARCH_ROOT/data/tvr`.** If files already exist there, audit them before reusing them; otherwise download them with the commands below.

Create a dedicated directory outside any existing project. Use a unique directory for a fresh study; reuse exactly the same directory when resuming that study. Keep all new documents, code, data, archives, environments, dependencies, model weights, indexes, caches, temporary files, logs, and reports inside it. Treat any pre-existing unrelated repository or source data as read-only. Do not overwrite another experiment or interrupt unrelated workloads.

Inspect the actual operating system, CPU architecture, GPU, RAM, available disk, and drivers. The setup below supports Linux x86_64 and aarch64, including a suitable Linux/WSL environment. Adapt installation commands for other supported systems after inspecting their documentation; do not assume the old device's hardware.

Choose the directory once and save its absolute path in your report:

```bash
export RESEARCH_ROOT="$HOME/vcmr-research-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESEARCH_ROOT"/{scripts,docs,logs,manifests}
printf '%s\n' "$RESEARCH_ROOT" > "$RESEARCH_ROOT/manifests/research_root.txt"
uname -sm
df -h "$RESEARCH_ROOT"
free -h
if command -v nvidia-smi >/dev/null; then nvidia-smi; fi
```

Reserve about **120 GB for this data setup**, including retained archives, extracted packages, independent CLIP copies, and a small Conda environment. This is a storage allowance, not the size of TVR raw videos. Training, new perception features, model weights, and indexes require additional space. No GPU is needed to download and validate the data.

The commands require Bash, `curl` with working CA certificates, `sha256sum`, `flock`, `setsid`, and `nohup`. Check for these first. They normally come with Linux; if missing on Ubuntu/Debian, the OS prerequisite command is `sudo apt-get install -y curl ca-certificates util-linux coreutils` after updating package metadata. All Python, research, and model dependencies must be installed in the project-local Conda environment, not system Python. If OS packages cannot be installed, use the available equivalent download/job tools and document the substitution.

Download these components:

| Component | Purpose | Destination relative to `$RESEARCH_ROOT` |
| --- | --- | --- |
| Official TVR training and validation JSONL, duration/index JSON | Authoritative queries, temporal annotations, IDs, and split metadata | `data/tvr/official_train.jsonl`, `official_val.jsonl`, `video2dur_idx.json` |
| TVR feature archive from MS-SL's release | Temporal visual/query feature package, including released CLIP files | `downloads/tvr.zip`, extracted into `data/tvr/mssl/` |
| CLIP video/query files from that archive | Convenient independent copies for method development and controls | `data/tvr/clip_video.h5`, `clip_query.h5` |
| CONQUER-compatible TVR feature archive linked by DeMUL | ResNet/SlowFast visual features, subtitle/query features, and baseline-related artifacts | `downloads/tvr_feature_release.tar.gz`, extracted into `data/tvr/tvr_feature_release/` |

These are annotations and precomputed features. They do not provide raw video pixels, verified object tracks, or every possible future model checkpoint. Downloading baseline-compatible features does not require using CONQUER, HERO, or DeMUL predictions as our search candidates. Select current comparison methods in section 6 and download their code/checkpoints separately under `external/` and `checkpoints/` with recorded source versions. Acquire any additional inputs required by the chosen original method before its main experiments.

Create `$RESEARCH_ROOT/scripts/setup_tvr_new_server.sh` with the following complete contents. No script needs to be copied from another computer:

```bash
#!/usr/bin/env bash
# Standalone TVR data setup for a new Linux device.
# Run: RESEARCH_ROOT="$HOME/vcmr-research" bash setup_tvr_new_server.sh
set -euo pipefail
: "${RESEARCH_ROOT:?Set RESEARCH_ROOT to your chosen experiment directory first.}"
export RESEARCH_ROOT
mkdir -p "$RESEARCH_ROOT"/{downloads,data/tvr,envs,tools,logs,tmp,.cache/conda,.cache/pip,manifests}
export CONDA_PKGS_DIRS="$RESEARCH_ROOT/.cache/conda"
export CONDA_ENVS_PATH="$RESEARCH_ROOT/envs"
export PIP_CACHE_DIR="$RESEARCH_ROOT/.cache/pip"
export XDG_CACHE_HOME="$RESEARCH_ROOT/.cache"
export TMPDIR="$RESEARCH_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$RESEARCH_ROOT/.cache/pycache"
export PYTHONNOUSERSITE=1
export HF_HOME="$RESEARCH_ROOT/.cache/huggingface"
export TORCH_HOME="$RESEARCH_ROOT/.cache/torch"
export CUDA_CACHE_PATH="$RESEARCH_ROOT/.cache/cuda"
export TRITON_CACHE_DIR="$RESEARCH_ROOT/.cache/triton"
export MPLCONFIGDIR="$RESEARCH_ROOT/.cache/matplotlib"
exec 9>"$RESEARCH_ROOT/download.lock"
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
if [ ! -x "$RESEARCH_ROOT/tools/miniforge/bin/conda" ]; then
  installer="Miniforge3-Linux-$(uname -m).sh"
  release='https://github.com/conda-forge/miniforge/releases/latest/download'
  fetch "$release/$installer" "$RESEARCH_ROOT/downloads/$installer"
  fetch "$release/$installer.sha256" "$RESEARCH_ROOT/downloads/$installer.sha256"
  (cd "$RESEARCH_ROOT/downloads"; sha256sum -c "$installer.sha256")
  bash "$RESEARCH_ROOT/downloads/$installer" -b -p "$RESEARCH_ROOT/tools/miniforge"
fi
conda_bin="$RESEARCH_ROOT/tools/miniforge/bin/conda"
if [ ! -x "$RESEARCH_ROOT/envs/tvr-data/bin/python" ]; then
  "$conda_bin" create -y --override-channels -c conda-forge \
    --prefix "$RESEARCH_ROOT/envs/tvr-data" python=3.11 pip numpy h5py
fi
export PATH="$RESEARCH_ROOT/envs/tvr-data/bin:$PATH"
python -m pip install 'gdown==6.4.2'

# Official temporal annotations, stored under the names used by our experiments.
ann='https://raw.githubusercontent.com/jayleicn/TVRetrieval/master/data'
fetch "$ann/tvr_train_release.jsonl" "$RESEARCH_ROOT/data/tvr/official_train.jsonl"
fetch "$ann/tvr_val_release.jsonl" "$RESEARCH_ROOT/data/tvr/official_val.jsonl"
fetch "$ann/tvr_video2dur_idx.json" "$RESEARCH_ROOT/data/tvr/video2dur_idx.json"

# Download only TVR, not the whole multi-dataset Google Drive folder.
if [ ! -f "$RESEARCH_ROOT/data/tvr/.mssl_extracted" ]; then
  gdown --no-cookies --continue '1RVG3qIy-nf9XdalMw8wpmWNSPcnOAaNK' \
    -O "$RESEARCH_ROOT/downloads/tvr.zip"
  python - <<'PY'
from pathlib import Path
import os,zipfile
r=Path(os.environ['RESEARCH_ROOT']);dest=(r/'data/tvr/mssl').resolve();dest.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(r/'downloads/tvr.zip') as z:
    for member in z.infolist():
        if not (dest/member.filename).resolve().is_relative_to(dest):
            raise RuntimeError('Unsafe zip member: '+member.filename)
    # Extraction reads every member and checks its ZIP CRC.
    z.extractall(dest)
(r/'data/tvr/.mssl_extracted').touch()
PY
fi

# Additional baseline-compatible features; these do not prescribe our proposed architecture.
if [ ! -f "$RESEARCH_ROOT/data/tvr/.conquer_extracted" ]; then
  gdown --no-cookies --continue '1DFnMNH-oi6-cZbl1coXqa_KjtsIsObxG' \
    -O "$RESEARCH_ROOT/downloads/tvr_feature_release.tar.gz"
  python - <<'PY'
from pathlib import Path
import os,tarfile
r=Path(os.environ['RESEARCH_ROOT']);archive=r/'downloads/tvr_feature_release.tar.gz'
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
r=Path(os.environ['RESEARCH_ROOT']);d=r/'data/tvr'
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
# Record local archive hashes for provenance, not as independent publisher checksums.
(cd "$RESEARCH_ROOT/downloads"; sha256sum tvr.zip tvr_feature_release.tar.gz) \
  > "$RESEARCH_ROOT/manifests/archive_sha256.txt"
python -m pip freeze > "$RESEARCH_ROOT/manifests/pip_freeze.txt"
"$conda_bin" list --explicit --prefix "$RESEARCH_ROOT/envs/tvr-data" > "$RESEARCH_ROOT/manifests/conda_explicit.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RESEARCH_ROOT/DOWNLOAD_COMPLETE"
echo "SUCCESS. Dataset: $RESEARCH_ROOT/data/tvr"
```

Run the setup detached, retaining the same exported `RESEARCH_ROOT`:

```bash
chmod +x "$RESEARCH_ROOT/scripts/setup_tvr_new_server.sh"
nohup setsid bash "$RESEARCH_ROOT/scripts/setup_tvr_new_server.sh" \
  > "$RESEARCH_ROOT/logs/setup.log" 2>&1 < /dev/null &
echo $! > "$RESEARCH_ROOT/logs/setup_launcher.pid"
tail -n 40 "$RESEARCH_ROOT/logs/setup.log"
```

For monitoring and completion checks:

```bash
tail -f "$RESEARCH_ROOT/logs/setup.log"
# After setup completes:
cat "$RESEARCH_ROOT/DOWNLOAD_COMPLETE"
cat "$RESEARCH_ROOT/manifests/download_verification.json"
```

Stopping `tail` does not stop the setup. After reconnecting, set `RESEARCH_ROOT` to the absolute directory you created; do not generate a new timestamped path. Resume by rerunning the detached launch command with that same root. The lock prevents concurrent installers, download continuation preserves partial progress, and extraction markers skip completed stages. Investigate corrupt or missing outputs even if a marker exists. A launcher PID may differ from the final detached child; inspect processes and logs rather than treating the launcher PID alone as proof of activity. Detachment survives a terminal/SSH disconnect, not a machine reboot.

Activate the local environment before further work:

```bash
source "$RESEARCH_ROOT/tools/miniforge/etc/profile.d/conda.sh"
conda activate "$RESEARCH_ROOT/envs/tvr-data"
```

Persist and source the script's cache environment settings for every subsequent job. Configure any library-specific cache explicitly if it ignores these settings. Install a hardware-compatible PyTorch build and the chosen method's remaining dependencies in a new local research Conda environment under `envs/`, or extend this local environment with a recorded dependency lock. The data installer itself does not install CUDA or a training framework.

Expected layout after successful setup:

```text
$RESEARCH_ROOT/
  data/tvr/
    official_train.jsonl
    official_val.jsonl
    video2dur_idx.json
    clip_video.h5
    clip_query.h5
    mssl/                     # original feature-release layout
    tvr_feature_release/
      video_feature/
      sub_query_feature/
      first_stage_trained_model/
      data/
  downloads/
  envs/tvr-data/
  tools/miniforge/
  scripts/setup_tvr_new_server.sh
  logs/setup.log
  manifests/
  DOWNLOAD_COMPLETE
```

Use these primary sources to check availability and resolve changed links:

- [Official TVR annotations and metadata](https://github.com/jayleicn/TVRetrieval/tree/master/data).
- [MS-SL TVR feature-release instructions](https://github.com/HuiGuanLab/ms-sl#ms-sl-on-tvr) and its [author-hosted feature folder](https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj). The TVR-only archive file ID used above is `1RVG3qIy-nf9XdalMw8wpmWNSPcnOAaNK`.
- [DeMUL data preparation](https://github.com/iLearn-Lab/TCSVT26-DeMUL#data-and-checkpoints), linking the CONQUER-compatible archive with file ID `1DFnMNH-oi6-cZbl1coXqa_KjtsIsObxG`. Do not confuse it with a different archive of the same filename from another release.
- [Miniforge installation](https://github.com/conda-forge/miniforge) and [gdown download/resume options](https://github.com/wkentaro/gdown).
- [Official TVR project](https://tvr.cs.unc.edu/) for dataset access, terms, and linked raw-media resources if the method needs them. Do not invent a public raw-video download URL if access requires a request.

The supplied file IDs and archive-layout expectations were checked for this prompt on 2026-11-14. Recheck official sources if a release changes. If Google Drive imposes a quota, preserve partial files and retry later or obtain the same files through its normal browser download. Never treat an HTML error page as an archive. If the verified archive size changes, inspect the official release before updating the size expectation. Record actual URLs, sizes, hashes, retrieval dates, and package versions. The supplied script has syntax/basic validation support; a full successful installation must be verified on the new device, not assumed.

The download check is only the first readiness check. Perform the full audit in section 5 before training or evaluating.

## 5. Step one: audit the TVR data

Use **TVR** for this experiment. It directly supports corpus moment retrieval with temporal annotations and established evaluation protocols. Work on this dataset throughout the study, using the files in `$RESEARCH_ROOT/data/tvr`. Do not expand to additional datasets to search for a favorable result.

Before declaring the dataset ready, audit:

- Official annotations, split definitions, and the exact searchable corpus for each split.
- Number of queries, videos, and source episodes where available.
- Query/video identifier alignment across every feature and annotation file.
- Temporal feature dimensions, normalization, sampling interval, and timestamp convention.
- Duration mismatches, invalid intervals, missing features, duplicate records, and split aliases.
- Available modalities: temporal visual features, subtitles, query features, audio, raw video, object detections, and tracks.
- Whether each claim of the intended method is observable from those inputs.

For the referenced release, expect 87,175 official training queries over 17,435 videos and 10,895 validation queries over 2,179 videos. The inspected CLIP release contains 19,614 video entries and 98,069 query entries, one fewer query-feature entry than the sum of official annotation records. Investigate the duplicated training-caption mapping and build an explicit query-ID reconciliation; never silently drop an official query or assume row-order alignment. Recompute these facts on the new device and record discrepancies.

Inspect prepared caption files for duplicate validation/test contents. File names alone do not establish independent splits. Use official temporal annotations for VCMR. These feature downloads do not imply raw videos or tracks are available; verify each modality. Do not infer visually verified actor identity, object trajectories, precise spatial relations, or absence from global frame embeddings alone.

If the best hypothesis needs additional perception inputs, pursue the official data-access route and extract them when feasible. Feature archives are a starting resource, not a restriction to a fixed-feature or reranking method. If essential inputs cannot be obtained, report the specific blocker and select a scientifically justified original design that can be tested; do not fabricate evidence or quietly downgrade the scope to a baseline wrapper.

Produce a machine-readable data manifest and a concise readiness report.

## 6. Step two: review the whole literature, choose SOTA, and develop cross-field inspiration

Search primary sources up to the actual date of execution: papers, official project pages, released code, checkpoints, and benchmark protocols. **Consider conference papers, journal papers, and preprints together as one body of research. Do not rank, exclude, or discount a method because of its venue or publication status. Choose the strongest relevant result on the evidence and comparable TVR VCMR performance.** Record provenance and whether numbers come from authors or your own reproduction so the comparison is traceable; this is not a publication-tier hierarchy.

Cover at least:

- VCMR corpus search, video selection, moment discovery, temporal localization, and corpus score calibration.
- Semantic parsing, semantic-role representations, compositional grounding, and temporal-language interpretation.
- Object/action/attribute association and the difference between detecting separate concepts and showing they belong to the same entity/event.
- Database organization, indexing, query planning, approximate/exact search, and algorithms relevant to the proposed mechanism.
- Closest mechanisms from other fields that could inspire a new solution.

Create one comparison matrix containing each relevant method's task, dataset/split, corpus size, modalities, supervision/pretraining, metrics, numerical results, source date, code/checkpoint availability, and mechanism relevant to our proposal. Select the **primary SOTA target** as the strongest supported result under the chosen comparable protocol and preregistered primary metric. Include any other method leading on secondary metrics. Do not select an easier target because its code is convenient, and do not automatically designate DeMUL as SOTA. If the strongest method cannot be reproduced, retain its documented score as the target and explain the reproduction limitation. Do not mix different splits, candidate corpora, or tasks into a misleading leaderboard; single-video grounding and PRVR recall do not count as VCMR results.

**Actively seek inspiration from mathematics, physics, statistics, information theory, algorithms, databases, optimization, geometry, dynamical systems, neuroscience, or other fields. Inspire, do not copy.** Identify an underlying principle, reason about how it could solve a specific semantic parsing or corpus-search failure, and derive an original mechanism for this setting. Directly transplanting a named algorithm, reproducing a known VCMR design, or attaching a physics metaphor is not sufficient. Search both the original field and retrieval literature for equivalent formulations before claiming novelty. Attribute established ingredients and specify the new representation, inference, learning, or search behavior they enable here.

Create a separate **novelty comparison for both the parser and the search algorithm** that answers:

1. What exact unsolved problem does each proposed mechanism address?
2. Which existing methods already solve related versions, including outside VCMR?
3. What is the precise operational and mathematical difference?
4. How do the new parser and search mechanism interact to enable behavior absent from the reference methods?
5. Could a simpler existing technique produce the same behavior?
6. What ablation, minimal pair, or counterexample would expose that equivalence?

Conclude with a reasoned choice of the strongest SOTA target and a defensible original research direction. Do not equate a literature search with proof of universal originality, and do not restrict the direction to incremental changes to the chosen comparison model.

## 7. Step three: define the original system and its hypotheses

Before building the main model, describe the complete information flow, for example:

```text
offline corpus preparation → our searchable corpus representation/index
query → our semantic parsing/inference mechanism
      → our search over all eligible corpus videos
      → discovery and localization of matching moments
      → globally comparable moment scores → optional video ranking
```

This diagram specifies required capabilities, not a mandatory modular architecture. A jointly learned or integrated formulation is allowed. Specify what is computed offline and per query, what is stored or indexed, what is learned, and how ambiguity, missing information, and temporal structure affect search. Justify frozen components when used; do not impose them as a research constraint.

Formulate one main research hypothesis and at most two contingency directions. The main proposal must contain concrete, falsifiable contributions to **both semantic parsing and independent corpus search**, including how they work together. Select the first direction using the literature, analytical reasoning, available or obtainable inputs, and small feasibility checks. Do not spend the full budget implementing three large models simultaneously.

Describe the equations or algorithm, assumptions, complexity, expected advantages, likely failures, and the strongest simpler explanation for a potential improvement. Investigate whether the formulation collapses to an existing mechanism. Prove any theoretical guarantee under explicit assumptions; otherwise label it a heuristic. Do not describe an uncalibrated score as a probability, causal effect, or statistical test.

The objective is a complete new approach that surpasses the selected SOTA. Feasibility checks should resolve risks while preserving that objective, not automatically reduce it to a small modification of an available model.

## 8. Step four: design the semantic parser carefully

Develop and implement an original text semantic parsing mechanism as a core research contribution. Define the new representation, learning objective, inference procedure, or interaction with corpus evidence, and explain the specific limitations it overcomes. Rules, dependency parsers, language models, learned structures, or multiple interpretations may supply infrastructure, but an off-the-shelf parser plus extraction rules is not the requested contribution. Build an ordinary parser as a comparison control alongside the proposed mechanism; it must not become the final method by default.

Do not force the representation to be a fixed predicate list or conventional scene graph. Investigate a formulation that can preserve compositional meaning and guide actual corpus search. If visual/corpus evidence helps resolve language ambiguity, specify the inference procedure and show that it uses no evaluation labels. Demonstrate a measurable behavior of the new parser beyond renaming its output format.

The representation should support, where expressed by the query:

- Entities and stable linguistic references.
- Actions, states, transitions, and their argument roles.
- Attributes with explicit owners.
- Object–object and person–object relations.
- Temporal relations such as before, after, while, until, and then.
- Negation, alternatives, counts, or other constraints when relevant.
- Unresolved ambiguity and unsupported interpretations.
- Exact source spans or provenance for extracted assertions.

For “a man standing in a black suit,” an illustrative representation is:

```text
person(e1)
suit(e2)
black(e2)
wear(e1, e2, I)
stand(e1, I)
```

This is an example of the intended semantics, not a claim that predicate graphs are new. The system must not attach “black” to the person instead of the suit. Linguistic entity references are not evidence of visually verified identity.

For “a man stands up after closing a door,” distinguish the transition “stands up” from the state “stands,” represent the event order, and distinguish the target event from contextual events. Do not force all verbs to happen simultaneously or hallucinate additional actions because they seem plausible.

Use the original whole query as a comparison control; any fallback in the proposed system must be specified and its use measured. Separate parser reliability from retrieval confidence. If probabilities are emitted, explain how they are calibrated; otherwise use explicit unknown/uncertain states.

Validate with meaningful minimal pairs and error cases: attribute ownership, role reversal, multiple people, temporal-order reversal, negation, state versus transition, and ambiguous references. Measure parser behavior separately from downstream retrieval. Do not report parser “accuracy” against automatically generated labels as if those labels were independent human ground truth. Clearly identify the provenance of any diagnostic annotations.

Show actual structured outputs, known parser errors, and the effect of those errors on search.

## 9. Step five: implement independent search over the entire corpus

Develop an original search algorithm that uses our semantic representation to discover, localize, and rank moments across **every eligible video in the benchmark corpus**. The algorithm may use an index, exhaustive search, a learned query planner, adaptive exploration, a hierarchical representation, or another justified design. Full-corpus search means every eligible video is reachable through our own algorithm; it does not require a brute-force scan on every query.

**Our system must not depend on the fixed top-K videos or temporal proposals produced by DeMUL, HERO, CONQUER, or another retrieval model.** It must have its own corpus access and moment-discovery mechanism and be capable of retrieving a correct video or interval outside those models' candidate lists. Standard encoders and indexing libraries may be attributed infrastructure. A custom search algorithm and semantic inference must supply the scientific contribution, not merely selecting a library or increasing K.

Specify and implement:

- The searchable corpus representation, its construction, and how it preserves the information required by the hypothesis.
- How parsed entities, roles, actions, attributes, relations, ambiguity, and temporal structure change retrieval operations.
- How the algorithm explores or prunes videos and intervals, including long-range or ordered events when claimed.
- How candidate intervals are discovered and boundaries are estimated without inheriting another system's fixed proposal list.
- How evidence is linked across time and, when observed, across entities.
- How missing or uncertain evidence affects search and scores.
- Global ranking, length/content-density effects, and duplicate/overlapping results.
- Offline cost, index size, query latency, memory, and scaling with corpus size and video duration.

Do not substitute a reranking-only pilot, a new score mixture, or a semantic wrapper for this deliverable. Reranking may be an internal stage of our independently operating system or a diagnostic comparison, but it cannot define the scope of the research. Removing all baseline-generated candidate files must leave our main inference pipeline able to run on the full corpus.

For any pruning rule, distinguish a proven bound from a heuristic and measure lost recall. Do not claim sublinear complexity without analysis and measurements. For a mathematical or physical formulation, demonstrate the specific operational benefit rather than relying on the terminology.

Measure candidate coverage separately from final ranking and localization. Quantify correct videos and moments found by our search that the strongest reproduced baseline misses; use ground truth only in this post-hoc diagnostic, never to guide retrieval. Analyze errors shared by both systems and new errors introduced by ours.

Use controls to isolate both contributions: whole-query retrieval, an ordinary semantic parser, simple semantic aggregation/search, our parser with conventional corpus search, conventional parsing with our search, and the complete proposed system. Choose controls that fit the actual hypothesis. Compare end-to-end systems under the same official corpus/protocol, report compute and feature differences, and add matched-budget ablations where informative. Do not force the proposed system to share a baseline's candidates in the main comparison, because candidate discovery is part of the innovation.

## 10. Step six: establish the evaluation protocol before tuning

Use official TVR validation as the final evaluation set, accurately named **validation**, not a hidden test set. Construct reproducible development selection and confirmation cohorts from original training data, grouped by source episode where possible. Save the split identities and seeds before experiments.

Document overlap already inherent in pretrained models and the benchmark. A released backbone may have seen our training-derived development examples, and its checkpoint may already have been selected on official validation. Creating new subsets does not erase those facts. Do not compare optimistic development scores directly with published validation scores.

Reproduce at least one strong, relevant released baseline. Verify feature compatibility, preprocessing, checkpoint loading, candidate generation, timestamp conversion, score normalization, and postprocessing. Document every mismatch; do not silently skip checkpoint weights or claim exact reproduction from merely loading a model.

Select the SOTA reference through section 6 rather than inheriting a previous experiment's baseline or scores. Reproduce the strongest comparable method where feasible and at least one strong available reference when it is not. DeMUL is one possible comparison, not a mandatory backbone, candidate generator, or assumed SOTA. Verify all metrics and implementation details directly on this device. Keep baseline execution separate from our independent search pipeline.

Never inject the ground-truth video into a corpus candidate list. Prediction code must not use target video IDs, target intervals, or evaluation labels. Ground-truth-conditioned analyses must be explicitly labeled **oracle diagnostics** and separated from real retrieval results.

Predefine:

- Primary metric, normally corpus R@1 at tIoU 0.5, with a justified choice if different.
- R@1, R@5, R@10, and R@100 at tIoU 0.5 and 0.7 as relevant secondary metrics.
- The strongest matched control selection rule.
- Allowed hyperparameter search and compute budget.
- Practical minimum improvement, uncertainty analysis, and pass/fail gate.
- Treatment of missing queries, unsupported inputs, ties, invalid intervals, and duplicates.
- Independent corpus-search coverage, recovery of baseline-missed moments, and relevant query subgroups.
- Separate end-to-end performance and ablations isolating parser novelty, search novelty, and their interaction.

Choose a pilot gate from the scientific hypothesis, practical effect size, and available statistical power before examining outcomes. A small improvement over an internal control is not completion of the project: the final ambition is to exceed the strongest comparable SOTA target with the complete independent system and demonstrate contributions from both parsing and search. Report all secondary metrics and regressions. Use paired uncertainty analysis where per-query predictions are available; do not claim paired significance against a paper reporting only aggregate numbers. Disclose the resampling/clustering unit, and do not call a bootstrap frequency a formal p-value. Keep the gate fixed after results are observed.

## 11. Step seven: implement, test, and run in the background

Implement a minimal, runnable pipeline before scaling up. Use meaningful tests for temporal IoU, video identity, timestamp conversion, candidate ranking, parser constraints, mathematical properties, and preventing ground-truth leakage. Include deliberately wrong-video and wrong-time cases so apparently high scores cannot hide evaluation mistakes.

Run a smoke experiment, confirm finite outputs and expected data shapes, inspect actual predictions, and verify deterministic behavior where intended. Scale only after these checks pass. Do not spend effort on large training runs when the core mechanism already fails a necessary analytical or empirical check.

Run long jobs in detached background processes, using a new OS session or a suitable persistent job runner. Disconnect standard input and write output/error logs to files. **The code must continue if the chat, SSH, browser, or terminal connection is lost.**

Provide:

- A process lock preventing duplicate jobs.
- Status files and timestamps.
- Per-stage logs and completion markers.
- Checkpoints and resumable stages.
- Atomic finalization of partial artifacts.
- Commands to start, monitor, and resume.
- Verification that the detached process is actually alive.

Explain that ordinary detachment does not survive machine power loss or reboot. Resume from verified completed artifacts after such interruptions. Check for existing live processes before launching a duplicate job. Keep the research decision logic bounded so a disconnected job does not launch unlimited unregistered variants.

Record random seeds, dependency versions, hardware, configurations, checkpoint/source identifiers, inference/training times, memory, and all predictions required for independent evaluation.

## 12. Step eight: follow the three-attempt limit

**At most three substantive research attempts are authorized for one experiment.**

1. Register and test idea 1.
2. If it fails, preserve its code, outputs, and decision evidence. Diagnose the failure. Propose and register one revised mechanism that directly addresses it, then test idea 2.
3. If idea 2 fails, repeat that process once for idea 3.
4. After idea 3, stop substantive method development. Do not introduce a fourth idea, rename an extra idea as a bug fix, or continue searching indefinitely.

Routine engineering corrections and prespecified ablations are not separate research attempts, but a changed scientific mechanism after inspecting outcomes is. Maintain an explicit attempt ledger. This fresh independent study has its own ledger. Resuming the same study inherits that ledger and does not grant three additional attempts. A revised attempt may replace the parser, representation, search algorithm, or entire architecture when the failure analysis supports it; revisions are not restricted to changing weights or reranking formulas.

Useful failure evidence includes:

- No supported improvement over the strongest matched control.
- Ablations show that the proposed parser or search mechanism contributes no supported benefit.
- Gains disappear when feature/compute budgets are equalized.
- A mathematical or empirical diagnostic shows the feature is redundant with a simpler score.
- Oracle/candidate analysis shows the proposed change cannot affect the dominant errors.
- An essential assumption fails on the available data.

Explain what the evidence actually establishes. Failure to demonstrate a benefit is not proof that every possible version of the broad idea is impossible. Conversely, a tiny favorable fluctuation is not proof of success.

If confirmation outcomes guide later revisions, acknowledge that the confirmation set is being used adaptively. Freeze every final mechanism and its weights before the final official-validation comparison. Do not tune on final validation, choose a different metric after seeing it, or hide unsuccessful attempts. If all three fail, deliver the negative results and stop as instructed.

## 13. Step nine: produce the final deliverables

Finish with a reproducible package containing:

- Dataset inventory, provenance, and readiness assessment.
- Current literature matrix and mechanism-level novelty analysis.
- Parser specification, implementation, examples, and error analysis.
- Independent full-corpus search specification, implementation, index/representation construction, equations, assumptions, and complexity.
- Evidence that inference works without any baseline candidate files, plus recovery analysis for moments missed by the baseline.
- Attributed baseline reproduction and documented discrepancies.
- Meaningful test results and smoke-run outputs.
- Frozen configurations, split definitions, seeds, and environment exports.
- Every attempt’s decision record and failure/success evidence.
- Predictions and complete metric tables, not only selected favorable numbers.
- Paired uncertainty analysis and relevant ablations.
- Runtime, memory, indexing/offline costs, and candidate-recall ceilings.
- Source-linked comparisons with the strongest SOTA results across the whole literature, selected without publication-status preference, with protocol differences explained.
- Exact start/resume/reproduce commands and artifact locations.
- A self-contained final report and a concise explanation in the chat.

The final report must answer plainly:

1. What dataset and exact protocol were used?
2. How does the implemented parser interpret the example query?
3. What specifically is proposed as new in both the parser and the independent search algorithm?
4. What established ideas are reused and attributed?
5. Which hypotheses were tested, and why did each pass or fail?
6. What are the actual results against matched baselines and the strongest comparable SOTA reference?
7. What evidence isolates the contribution of the parser, search mechanism, and their interaction from extra features, increased compute, or ordinary recalibration?
8. Can the system discover correct videos and moments outside the baseline candidate lists, and what measured evidence demonstrates this?
9. Has the complete approach beaten the selected SOTA under comparable conditions, and what evidence supports its novelty?
10. What limitations remain?
11. Where are the files and how can the complete experiment be reproduced?

Be rigorous, inventive, and candid. I want a researcher’s reasoned experiment with executable evidence, not a guaranteed-positive story. Start with the plan, then carry the authorized work through to completion.
