# DeMUL baseline reproduction notes

Source: https://github.com/iLearn-Lab/TCSVT26-DeMUL (MIT license retained in `external/demul/LICENSE`). The repository is retained unmodified; the independent wrapper is `scripts/reproduce_demul.py`.

## Verified artifacts

- Checkpoint: https://media.githubusercontent.com/media/iLearn-Lab/TCSVT26-DeMUL/main/checkpoints/demul_tvr/model.ckpt
- Size: 67,117,718 bytes.
- SHA256: `723636f3342c5177248a54d7112d4731aaba669133404cf87975875e21615f39`.
- CONQUER feature archive: https://drive.usercontent.google.com/download?id=1DFnMNH-oi6-cZbl1coXqa_KjtsIsObxG&export=download&confirm=t
- Archive size: 22,318,784,064 bytes. HTTP range response and gzip signature verified before download.
- Expected features: ResNet+SlowFast 4352 dimensions; subtitle and query features 768 dimensions; clip spacing 1.5 seconds.

## Why a wrapper is necessary

The public source and checkpoint are not directly compatible. The official inference entry point passes a constructor argument absent from the current model, uses obsolete model/results paths, and strict loading fails because the source renamed layers and changed the shape of channel-wise LayerNorm affine parameters. Source history contains no matching earlier architecture implementation.

The wrapper performs a bijective adapter, then uses `strict=True`; every one of the 195 checkpoint tensor entries is consumed and every expected model tensor is supplied. There are 16,763,874 model parameters. The exact mapping is saved per run in `checkpoint_mapping.json`.

| Checkpoint prefix | Current source prefix |
|---|---|
| `contextual_QDF_visual_feature_learning.` | `qdf_visual.` |
| `contextual_QDF_text_feature_learning.` | `qdf_text.` |
| `fusion_moment_localization_head.` | `moment_head.` |
| `moment_head.BIGRU.` | `moment_head.end_GRU.` |
| `moment_head.begin_score_modeling.` | `moment_head.begin_score.` |
| `moment_head.end_score_modeling.` | `moment_head.end_score.` |
| `query_weight.moe_fc_txt.` | `query_weight.moe_fc.` |

LayerNorm affine weights/biases `[1,384,1]` are reshaped to `[384]`, preserving the channel-wise affine transform. The other tensors retain their values and shapes. The model remains the released source model. We cannot prove equivalence to unavailable historical computation from tensor names alone; report this as a **checkpoint-compatible reconstruction** until numeric reproduction is established. In particular, the current source uses packed GRU sequences, and its historical implementation is unavailable.

## Safe checkpoint loading

An initial request to use upstream-style `weights_only=False` was rejected by automatic approval review because arbitrary pickle execution is possible. The wrapper instead keeps `weights_only=True`, rejects unexpected nonstandard globals, and maps the sole configuration tag `easydict.EasyDict` to the standard `collections.OrderedDict` container. After safe decoding, inert configuration dictionaries are converted to EasyDict for the model API. No unrestricted pickle loading is used.

## Protocol

Defaults match released `checkpoints/demul_tvr/opt.json`:

- 10 HERO-retrieved candidate videos, with no GT video injected.
- Maximum 100 clip features and 30 query tokens.
- Candidate pairs satisfy `1 <= end_index - start_index < 24`.
- Predictions: `start_seconds=1.5*start_index`; `end_seconds=1.5*(end_index+1)`.
- Shared start/end softmax across all candidate videos and temporal positions.
- Each boundary probability is multiplied by the normalized HERO prior; the final joint score consequently contains the squared prior.
- Keep 600 moment candidates before per-video NMS at 0.7; return the globally highest 100 retained moments.
- Evaluate with the released official evaluator at tIoU 0.5 and 0.7.

The upstream example shell script instead specifies 100 candidate videos and minimum index difference 0; these are different settings. The reported reference VR R@100 equals R@10, consistent with the released 10-video protocol.

The upstream dataset prepends the ground-truth video for SVMR and removes it before corpus prediction. Our loader eliminates that extra input entirely. Its input construction reads only the query text/ID, independent candidate ranking, and video/subtitle features. Target video names and timestamps are used only by the evaluator after predictions are complete.

## Published reference results

Source: https://github.com/iLearn-Lab/TCSVT26-DeMUL/blob/main/checkpoints/demul_tvr/metrics.json

| TVR validation metric | R@1 | R@5 | R@10 | R@100 |
|---|---:|---:|---:|---:|
| VCMR tIoU >= 0.5 | 21.37 | 34.11 | 39.31 | 47.43 |
| VCMR tIoU >= 0.7 | 11.92 | 25.27 | 31.34 | 39.79 |
| Video retrieval | 29.01 | 52.82 | 63.07 | 63.07 |

These values are author-reported references, not results from this experiment. The checkpoint itself was selected on official validation; any development done on its original training examples must acknowledge potential overfitting.

## Running and reusable outputs

```bash
envs/vcmr/bin/python scripts/reproduce_demul.py \
  --feature-root data/tvr/tvr_feature_release \
  --output runs/demul_smoke --limit 16 --no-evaluate
```

Use `--annotations` for alternate queries, `--query-ids` for a JSON list or newline-separated subset, and `--rankings` for independent custom candidates. The latter accepts JSONL rows with `desc_id`, `video_indices` or `video_names`, and finite positive `scores`. Custom candidate sets change the published protocol and must be labeled accordingly.

Outputs include original start/end logits `[N,2,K,100]`, candidate video indices, HERO scores, query IDs, raw 600 moment proposals, final 100 moment proposals, protocol/commit/checksum manifests, and optional metrics. `--no-evaluate` suppresses metric computation. Output folders must be empty to protect prior runs.

## Checks completed

- Safely decoded the verified checkpoint: epoch 11.
- Strict adapted load: all 195 tensor entries matched.
- GPU synthetic forward: 2 queries × 3 videos; start/end shape `[2,3,100]`; moment score shape `[2,3,100,100]`; all scores finite.
- No official validation accuracy was computed during these smoke checks.

## Completed official-validation run

`runs/demul_val` evaluates all 10,895 validation queries over the original 2,179-video corpus with the supplied HERO rankings. The reconstructed checkpoint obtains R@1 **21.13 at tIoU 0.5** and **11.78 at tIoU 0.7**, versus the author's **21.37 / 11.92**. Thus we do **not** claim exact numerical reproduction. Its video retrieval recalls exactly match the author-reported 29.01 / 52.82 / 63.07 at K=1/5/10.

A source-level postprocessing discrepancy was discovered: upstream `eval_epoch` invokes `get_submission_top_n`, which mutates the same prediction dictionaries stored in `eval_submission_raw`. As a result, its subsequent NMS receives only the first 100 predictions even when configured for 600. Our wrapper retains all 600 before NMS. This affects high-K recall, so results must disclose this difference. `runs/demul_val/metrics_upstream_truncated_nms.json` separately measures the upstream truncation behavior using the same saved raw predictions; `reproduction_comparison.json` collects both variants and the published reference.

For the new algorithm comparisons, use the same 600 raw proposals, NMS, and evaluation path for the fixed reconstructed baseline and every proposed variant. An improvement against this reconstruction alone does not establish superiority over the published method or overall SOTA.

## Development coverage

The supplied `train_select100_top2000` ranking LMDB covers 5,022 of 5,045 selection queries and 4,991 of 5,010 confirmation queries. Missing query IDs and retained-cohort hashes are in `manifests/demul_train_ranking_coverage.json`. The explicit retained annotation files are `data/tvr/splits/demul_dev_select.jsonl` and `data/tvr/splits/demul_dev_confirm.jsonl`. The original development split files are preserved. Report these retained cohorts, not the original full split sizes, for DeMUL-based comparisons.

A numerical candidate-independence check prepended a duplicate retrieved video (never the ground truth). Under strict float32, the retained start/end logits changed by at most 3.82e-6. Results are recorded in `manifests/demul_candidate_independence.json`; no cross-video evidence transfer from the omitted upstream diagnostic slot was observed in that check.
