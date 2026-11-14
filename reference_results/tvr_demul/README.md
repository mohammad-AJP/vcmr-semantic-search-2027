# Existing TVR baseline results

Use this directory as fixed comparison evidence. No DeMUL training or inference
is needed on the new device to use these results. This bundle was made by
compressing existing predictions and copying their matching metrics/metadata.

The method is a **DeMUL checkpoint-compatible reconstruction**, evaluated on
10,895 official TVR validation queries over 2,179 videos. It is not an exact
reproduction claim or an assertion that DeMUL remains current SOTA.

| Files | Meaning |
| --- | --- |
| `predictions_600_before_nms.jsonl.gz`, `metrics_600_before_nms.json` | 600 raw proposals before NMS, up to 100 final predictions per query |
| `predictions_100_before_nms.jsonl.gz`, `metrics_100_before_nms.json` | 100 raw proposals before NMS, up to 100 final predictions per query |
| `original_run_manifest.json` | Source checkpoint hash/commit, feature and inference settings |
| `original_wrapper_and_author_metrics.json` | Earlier wrapper results and separate author-reported scores |
| `dataset_protocol.json` | Data/split provenance and timing conventions |
| `video2idx.json` | Original video index mapping; bundled predictions already use video names |
| `manifest.json` | Relative source paths, hashes, query counts, schema, and limits |

Use each prediction file with its matching `metrics_*` file. The earlier
wrapper metrics differ slightly from the final evaluation path and must not be
presented as the exact metrics of these bundled predictions. Keep the 100- and
600-proposal protocols separate, particularly when comparing recall at large K.
See [baseline notes](../../docs/demul_reproduction.md) for historical details.
Their fixed-candidate instructions describe the previous reranking study, not
the scope of the new independent-search project.

Rows contain `desc_id` and `predictions`, with each prediction formatted as
`[video_name, start_seconds, end_seconds, score]`. Scores in these final-evaluation
files are log-domain ranking scores. Preserve their ordering and tie handling.
Read them directly with Python's standard library, from the repository root:

```python
import gzip
import json
from pathlib import Path

folder = Path('reference_results/tvr_demul')
metrics = json.loads((folder / 'metrics_600_before_nms.json').read_text())
print(metrics['full'])
with gzip.open(folder / 'predictions_600_before_nms.jsonl.gz', 'rt') as handle:
    baseline = {str(row['desc_id']): row['predictions']
                for row in map(json.loads, handle)}
print('Queries:', len(baseline))
```

For a paired comparison, download official validation annotations, confirm
their identity against the recorded annotation hash, align query IDs, and
evaluate the new system on the same full corpus. The annotations are inputs
only to evaluation. No model inference is necessary to read or evaluate saved
baseline predictions. The shared evaluator is `src/vcmr/evaluation.py`.

These files contain final top-100 results, not all pre-NMS candidates. They
support final-output hit/miss analysis; do not infer complete raw candidate
coverage from them. The full raw candidates remain local under ignored `runs/`.
Historic absolute paths in copied manifests record provenance; they are not
paths that need to exist on the new device. Do not reinstall the old machine's
Conda explicit environment blindly on a different architecture.
