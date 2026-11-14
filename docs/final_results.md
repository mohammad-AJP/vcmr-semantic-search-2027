# Final TVR experiment results

Generated 2026-11-14T18:41:38.633686+00:00.

All three permitted method attempts are complete. Method weights were selected on training-derived development data and frozen before this final evaluation. No fourth method was attempted.

## Outcome

**None of the proposed mechanisms passed the prespecified development improvement gate. This experiment does not establish a new state of the art.** A small change in an aggregate metric is insufficient when the proposed feature is unused or fails to outperform simpler controls.

## Final official validation

All **10,895 queries** and **2,179 corpus videos**. Each query uses the same independent HERO top-10 video candidates and the same 600 DeMUL moment proposals. About 35.4% of queries satisfy the conservative semantic eligibility rule. Values are percentages.

| Method | R1@0.5 | R1@0.7 | R5@0.5 | R10@0.5 | R100@0.5 |
|---|---:|---:|---:|---:|---:|
| demul | 21.13 | 11.76 | 33.70 | 39.19 | 55.25 |
| whole_query_control | 21.39 | 11.90 | 33.78 | 39.37 | 55.35 |
| joint_support_control | 21.38 | 11.93 | 33.79 | 39.34 | 55.35 |
| whole_local_contrast_control | 21.38 | 11.93 | 33.82 | 39.36 | 55.36 |
| attempt1_association_reference | 21.36 | 11.91 | 33.77 | 39.34 | 55.36 |
| attempt2_conditional_reference | 21.38 | 11.93 | 33.79 | 39.34 | 55.35 |
| attempt3_conditional_reference | 21.38 | 11.93 | 33.82 | 39.36 | 55.36 |

The names of frozen attempted systems do not imply that their new feature was selected. Its coefficient was:

- Attempt 1: **0.0625**.
- Attempt 2: **0** (proposed feature disabled by development selection).
- Attempt 3: **0** (proposed feature disabled by development selection).

### Published reference comparison

| Published reference | TVR validation R1@0.5 | R1@0.7 | Interpretation |
|---|---:|---:|---|
| [DeMUL, TCSVT 2026](https://github.com/iLearn-Lab/TCSVT26-DeMUL/blob/main/checkpoints/demul_tvr/metrics.json) | 21.37 | 11.92 | Author-reported released checkpoint; strongest directly reproduced reference in this experiment. |
| [GenSpan, June 2026 preprint, Table 1](https://arxiv.org/html/2603.22121v2) | 18.77 | 12.12 | Author-reported VCMR validation; generation/subtitle pipeline and features differ. Not reproduced here. |

DeMUL reconstruction reaches 21.13 / 11.78; its architecture/checkpoint compatibility adapter consumes every tensor but does not exactly reproduce author numbers. Our residual methods add CLIP features; matched whole-query and parser controls separate this added-input benefit from the proposed mechanism. These references are not an exhaustive leaderboard: recent CHDL results were identified but their complete numerical protocol was unavailable. No universal SOTA claim is made.

### Candidate-count sensitivity

The released evaluation code mutates the raw candidate list to 100 before NMS, despite a saved setting of 600. The primary internal comparison gives every variant 600 proposals. The following separate run keeps the original first 100 proposals for every variant before reranking, matching that effective candidate budget. Neither cohort nor model weights change.

| Method | R1@0.5 (100 before NMS) | R1@0.7 | R100@0.5 |
|---|---:|---:|---:|
| demul | 21.13 | 11.76 | 47.57 |
| whole_query_control | 21.39 | 11.90 | 47.57 |
| joint_support_control | 21.38 | 11.93 | 47.57 |
| whole_local_contrast_control | 21.38 | 11.93 | 47.57 |
| attempt1_association_reference | 21.36 | 11.91 | 47.57 |
| attempt2_conditional_reference | 21.38 | 11.93 | 47.57 |
| attempt3_conditional_reference | 21.38 | 11.93 | 47.57 |

### Paired final comparisons

Candidate minus the strongest control chosen on development selection: `whole_local_contrast_control`. Video-cluster percentile bootstrap, 2,000 resamples; percentage-point differences. These intervals are descriptive and are not corrected for all adaptive method comparisons.

| Attempt | Final R1@0.5 difference | 95% interval |
|---|---:|---|
| attempt1_association_reference | -0.018 | [-0.055, +0.018] |
| attempt2_conditional_reference | +0.000 | [-0.037, +0.037] |
| attempt3_conditional_reference | +0.000 | [+0.000, +0.000] |

## Three-attempt ledger

Development confirmation contains 4,991 queries. Success required at least +0.25 percentage points and a strictly positive 95% paired interval against the strongest control selected on the other development cohort. Confirmation was reused to guide revisions, so later confirmations are exploratory; final official validation was reserved until all variants were frozen.

| Attempt | Mechanism | Confirmation R1@0.5 | Gain over selected control | 95% interval | Gate |
|---|---|---:|---:|---|---|
| 1 | Video-wide opportunity reference | 41.475 | +0.000 | [-0.080, +0.080] | Fail |
| 2 | Conditional local score with global residual reference | 41.475 | +0.020 | [+0.000, +0.060] | Fail |
| 3 | Candidate-neighborhood conditional residual reference | 41.455 | +0.000 | [+0.000, +0.000] | Fail |

Attempt 1 could not distinguish equal-duration locations through its reference: that term was constant within video/duration and changed only 14 confirmation top predictions versus raw joint support. Attempt 2 selected zero weight for the proposed conditional feature. Attempt 3 tested whether neighborhood-specific residual distributions repaired the limitation. These are pilot failures under the measured conditions, not impossibility proofs.

## Data, parser, and limitations

- Source data: `/home/mohammad/VisualSearch`. Only TVR was used. Independent copies, acquired baseline features, code, environment, caches, predictions, and documents are under this workspace.
- Official TVR validation: 10,895 queries / 2,179 videos. Development selection: 5,022 retained queries; confirmation: 4,991. Missing train HERO rankings are explicitly recorded, without ground-truth candidate injection.
- The parser extracts source-anchored entities, attributes, actions, roles, and temporal relations. Example: “a man standing in a black suit” has man and suit entities, black attached to suit, standing attached to man, and a wear relation. It does not invent visual tracks.
- Frozen CLIP text embeddings match supplied full-query embeddings with median cosine 0.999999089 on 128 training examples. New phrase embeddings are computed with the same encoder.
- These are feature-based reranking experiments over a fixed search candidate pool, not a new full-corpus index or a trained end-to-end replacement. No raw videos or actor tracks were available. Global frame features cannot prove that multiple predicates concern the same person.
- The DeMUL checkpoint was pretrained on the original training set, including our development examples. The official benchmark also shares source episodes between train/validation. Development performance is optimistic and cannot be compared numerically with validation performance.
- Local validation/test feature annotations are aliases; no claim is made about held-out test results. Temporal features use nominal 1.5-second spacing, with flagged length/duration discrepancies retained and documented.
- Circular-shift surrogate scores are heuristic references, not valid causal tests or calibrated probabilities. Novelty is unproven; semantic parsing, OIC, regression, and permutation statistics are established methods and credited in the revision notes.
- Baseline checkpoint/source differences and pre-NMS candidate handling limit external comparability. Full metrics, visual-only subgroup results, all frozen weights, and bootstrap outputs are saved; no failed attempt is omitted.

## Reproduce and resume

```bash
cd /home/mohammad/Documents/vcmr-semantic-search-2027
source scripts/environment.sh
"$VCMR_PYTHON" -m pytest -q tests
cat runs/background/status.json
```

Long jobs use detached OS sessions with standard input disconnected and file-backed logs. Completed stages have markers and failed stages stop dependent work. Terminal/chat disconnection does not kill these jobs; machine reboot does. Environment exports are in `manifests/conda_explicit.txt` and `manifests/pip_freeze.txt`.

Artifacts: [final JSON](../runs/final_validation/results.json), [baseline reproduction](demul_reproduction.md), [revision reasoning and prior art](research_revision_notes.md), [reusable agent prompt](reusable_agent_prompt.md), [parser examples](../examples/implemented_parser_examples.json), [original source check](../manifests/original_source_execution_check.json).
