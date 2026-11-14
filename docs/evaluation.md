# Evaluation and execution plan

Status: planned only. No model results, synthetic demonstrations or SOTA improvements are claimed.

**Data gate**

The local inventory found the following prepared records:

| Dataset | Training queries/videos | Local validation queries/videos | Current limitation |
|---|---:|---:|---|
| Charades-STA | 12,408 / 5,338 | 3,720 / 1,334 | val and test caption files are identical; raw videos absent |
| ActivityNet-Captions | 33,721 / 9,043 | 15,753 / 4,430 | val/test alias; 4 fuzzy training matches and 17 fuzzy validation matches; 4 degenerate training moments |
| TVR | 87,174 / 17,435 | 10,895 / 2,179 | val/test alias; current visual feature package differs from several VCMR releases |

Counts come from existing prep reports; 100% matched does not imply every timestamp is valid. Check original annotations, durations, IDs and timestamp conversion independently. A dataset is not a final protocol until the complete retrieval corpus and split mapping are frozen.

Current pooled I3D/ResNet/CLIP features and query text permit a temporal feature pilot. They do not expose reliable person tracks, garment colors or actor-object assignments. No raw video files were found under the inspected data root. No FIG dataset directories were found.

Use Charades for inexpensive temporal plumbing and ActivityNet for longer event sequences. Prefer TVR as an established original VCMR comparison target if its exact official features, modalities and split can be reproduced. Multi-actor binding may need TVR or a deliberately sampled raw-video diagnostic set; Charades alone may offer too few relevant cases. Fine-grained FIG annotations are a later separate evaluation, not a relabeling of current original queries.

Before implementation:
1. Map local split names to official source splits. Never tune on local val and report its identical test alias as held-out performance.
2. Create a video-disjoint development split from training when required; reserve the official evaluation split. Deduplicate at source/episode level when the protocol permits it.
3. Record candidate corpus IDs, query IDs, video durations, missing records, feature stride/dimensions, modalities and supervision in immutable manifests.
4. Reconcile corpus size and feature streams with the chosen baseline. For example, EventFormer uses a different ActivityNet corpus size from the local MS-SL subset and different TVR visual features. Published results cannot be copied into a supposedly matched comparison. [EventFormer](https://arxiv.org/html/2402.13566)
5. Acquire only the raw-video/object evidence needed for the binding experiment, inside the independent data/ directory. Do not pretend reconstructed labels from global embeddings are equivalent to object observations.
6. Copy necessary existing artifacts into this root for execution; preserve originals. Set cache/temp/environment paths inside the root.

**Stage 1: determine whether the proposed failure matters**

Choose an initial 200-300-query development pilot before inspecting model improvements. Stratify by attribute ownership, actor-object roles, temporal composition, and simple controls; retain a random component to estimate prevalence without enrichment bias.

Have humans inspect original queries and pooled top candidates from multiple methods, with model identities hidden. Mark:
- every required predicate supported or contradicted or unobservable;
- whether the predicates bind to compatible actors and event times;
- parser correctness;
- plausible alternate positive moments and boundary ambiguity.

Use two annotators with adjudication for the diagnostic subset where feasible. Report disagreement and unobservable cases. A model judge can prioritize review but is not the sole ground truth for validating that model family's outputs.

Run an oracle diagnostic with manually checked programs and reliable annotated evidence. Oracle labels are used only to estimate headroom and diagnose bottlenecks; they are not inputs to the real test-time method. If compatible assignments cannot outperform independent scores on the target failure cases, reject or revise the binding hypothesis.

A small pilot estimates feasibility, not paper-level superiority. Use observed variance to choose a larger final sample.

**Stage 2: establish a faithful VCMR baseline**

Use a released corpus retriever/localizer with a frozen protocol and verify its reproduced scores before modifying search. Current PRVR Grounded Recall is not the VCMR evaluator.

Predictions must be ranked (video_id, start, end, score). A hit requires both the correct video and temporal IoU above the selected threshold. Use the baseline's candidate limits, temporal NMS, rounding and duration handling. Never append the known positive to real inference shortlists; existing GT-appended diagnostic dumps must not leak into evaluation.

Keep a separate oracle-video localization evaluation to distinguish video-retrieval failure from boundary failure. A reranker cannot exceed the relevant candidate-coverage ceiling.

**Stage 3: controlled method ladder**

| Condition | Purpose |
|---|---|
| Full-query baseline | Establish performance without semantic decomposition |
| Independent predicate fusion | Measure whether parsing alone helps or hurts |
| Compatible joint assignment | Isolate actor/time consistency |
| Joint assignment plus simple duration/actor-count normalization | Test a cheap alternative to the reference idea |
| Joint assignment plus restricted association reference | Test the additional ranking hypothesis |
| Uniform, random, rare-predicate, entropy and largest-gap scheduling | Equal-budget search-policy controls |
| Proposed ranking-sensitive scheduling | Test adaptive computation beyond simpler scheduling |
| Standard single-parse and fixed multi-parse versions | Control parser complexity |
| Proposed selective ambiguity resolution | Test whether adaptive parsing adds value |

Use identical visual evidence and candidate sets when isolating ranking changes. In an end-to-end search-policy comparison, allow expansion under the same cost budget and report its contribution separately. Include an ordinary full-query VLM verifier using the same frame/crop inputs to test whether the graph contributes beyond stronger perception.

Run two separate tracks:
- Matched features: all methods receive the same features; this isolates the proposed search/ranking algorithm.
- Enhanced perception: raw frames, object tracks or stronger encoders are allowed; disclose all added inputs, training data and compute.

Keep published protocol numbers separate from locally reproduced matched comparisons. Include relevant structured prior art, not only old embedding baselines. Refresh the literature and publicly released models before a SOTA claim.

**Stage 4: metrics and statistical analysis**

Primary metric: corpus R@1 at tIoU 0.5, with R@1 at 0.7 and R@5/10/100 at both thresholds also reported. Add tIoU 0.3 where the selected FIG protocol uses it. If a benchmark uses a different primary convention, follow it and declare it before final testing.

Additional measurements:
- Candidate video recall and candidate interval coverage at stated depths.
- Oracle-video temporal localization.
- Binding error on the independently audited subset.
- Per-query-type performance, including simple queries, repeated events and sparse/occluded evidence.
- Parser correctness and predicted-parser versus oracle-parser gap.
- Median/p95 latency, decoded frames, detector/LLM calls, GPU memory, index storage, and offline indexing time.
- Accuracy versus compute budget and versus distractor-corpus size.

Do not silently count all unannotated candidates as semantically false in the diagnostic analysis. Report official single-positive metrics unchanged and any pooled multi-positive audit separately. Edited queries are not negatives until validated: the edited action or attribute may actually occur.

For trained variants, use at least three seeds and report dispersion. Use paired bootstrap confidence intervals with video-level clustering where multiple queries share a video; consider source/episode dependence. Include all parser/index/verification costs and separate cold-cache and warm-cache runs.

Freeze the main metric, model selection protocol, hyperparameters, budgets and corpus before final testing. No tuning or repeated selection against the held-out test set.

**Decision gates**

Continue to full experiments only if:
- The baseline can be reproduced under the chosen protocol.
- Candidate coverage leaves room for improvement.
- The oracle analysis shows binding-related headroom.
- Gains survive comparison with ordinary joint matching and equal-budget verification.
- The mechanism improves the main VCMR metric or gives a clear accuracy/cost benefit, with ordinary queries and repeat events reported.
- The learned parser/perception version retains a meaningful part of the oracle gain.

Stop or pivot if better visual features explain all the gain, oracle binding does not help, the reference only rewards rarity, or parser/tracker noise eliminates the benefit. Do not set an unsupported numeric promise of SOTA improvement.

**Future correctness checks**

Once implementation starts, small tests should check:
- attribute ownership, shared identity, target/context boundaries, and before/after direction;
- unknown evidence versus negation;
- exhaustive tiny-example agreement with the structured solver;
- lower/upper bound validity on fixed evidence, agreement with exhaustive fixed-score rankings, and correct bound invalidation after new observations;
- a counterexample where overlapping high-score proposals make pre-NMS top-K pruning unsafe after NMS;
- consistent track-ID relabeling invariance;
- no GT injection, split leakage or duplicate alias-as-test reporting;
- temporal IoU, duration clipping and global tuple ranking.

These are planned algorithm/evaluation tests. No experiment code or smoke tests were run for this theory-only deliverable.
