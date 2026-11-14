# VCMR semantic search experiments

Isolated research workspace; source datasets are under `/home/mohammad/VisualSearch`.
TVR is the only dataset used for these experiments. The original `grounded-prvr`
repository and downloaded source data are not modified.

## Current execution

The parser, exact CLIP encoder verification, DeMUL checkpoint reconstruction,
all baseline inference runs, and first method implementation are complete.
All three attempts and final official-validation comparisons are complete. See [the final results](docs/final_results.md).
The prespecified gates, all numerical outcomes, and limitations are recorded in the final report; novelty remains unestablished.

The isolated Conda environment is `envs/vcmr`. Run tools using:

```bash
cd /home/mohammad/Documents/vcmr-semantic-search-2027
source scripts/environment.sh
"$VCMR_PYTHON" -m pytest -q tests
```

## Background execution and reconnection

```bash
bash scripts/start_background.sh
cat runs/background/status.json
tail -f runs/background/launcher.log
```

The launcher starts a new OS session, redirects all input/output to files, and
returns immediately. The process survives terminal or chat disconnection.
A file lock prevents duplicate pipelines. Each completed stage has a marker;
relaunching skips completed stages. This does not survive a machine reboot;
relaunch the same command after reboot. Interrupted feature caches are rebuilt
from their stage, while completed caches are retained.

Detailed logs are in `runs/background/`. The initial pipeline stops safely after
attempt 1 confirmation so its evidence can inform the next research decision;
this is an automatic stage boundary, not a request for user permission.

## Reproducibility and interpretation

- [Data and split protocol](manifests/tvr_protocol.json)
- [Checkpoint reconstruction and evaluation differences](docs/demul_reproduction.md)
- [First attempt specification](configs/attempt1_preregistration.json)
- [Literature and novelty assessment](docs/literature.md)
- [Execution plan](docs/execution_plan.md)

The published DeMUL R@1 reference is 21.37 / 11.92 at temporal IoU 0.5 / 0.7.
Our checkpoint-compatible reconstruction obtains 21.13 / 11.78. Exact numerical
reproduction is not claimed. At high K, an upstream candidate-truncation issue
changes results; both postprocessing protocols are saved in
`runs/demul_val/reproduction_comparison.json`.

All new algorithms and controls must use the same candidates, feature inputs,
and evaluation procedure. Improvements against the reconstruction alone do not
establish state of the art. We will attempt at most three substantive variants,
report negative outcomes, and keep final validation separate from development.
