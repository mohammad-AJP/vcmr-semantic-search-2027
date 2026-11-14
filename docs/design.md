# Parser and search hypotheses

Status: proposed design, not an implemented or experimentally validated method.

The task is to return globally ranked tuples (video_id, start_seconds, end_seconds, score). A separate video list can aggregate each video's best moment score. VCMR must retain temporal intervals and may return multiple distinct moments from one video.

**Represent the query as a small executable program**

For "a man standing in black suit", normalize the grammar while preserving the meaning:

```text
exists person x, garment y, interval I:
    man(x)
    suit(y)
    black(y)
    wearing(x, y, I)
    standing(x, I)
return I
```

The color belongs to the suit; the suit belongs to the standing person. Standing is a state, not automatically the transition "stands up". Do not invent a location, occasion, identity, or extra action. A standing person in white plus another seated person in black is an instructive mismatch.

The parser should retain:
- Entity variables and coreference, with lexical source spans.
- Actions/states and their argument roles: who does what to whom.
- Attributes attached to their owners.
- Spatial and temporal relations, including before/after/overlap.
- Target event versus contextual events. For "stands up after closing the door", return the standing-up interval; the earlier closing event is contextual evidence.
- Negation scope, alternatives and cardinality when explicitly stated.
- A small set of legitimate alternative parses where ambiguity exists.
- The evidence modality needed by each predicate and an unknown/contradicted/supported observation state.

Do not impose same-frame conjunction on sequential actions. A temporal program binds distinct intervals and enforces the stated relation. Do not treat a missed detection as proof of absence. Temporal coverage requirements should depend on the predicate: a state and a brief transition need different observation rules.

**Parser options**

P0: dependency/semantic-role parsing plus transparent conversion rules. This is a cheap baseline with limited scope handling.

P1: a frozen instruction model emitting a schema-validated program. Require source-span provenance; reject unbound variables and unsupported mandatory facts; preserve the original query as a holistic retrieval channel. Log parser/model/prompt versions. Do not trust a self-reported confidence as a calibrated probability.

P2: a small distribution over plausible P1 programs. Learn or calibrate weights on training/development annotations, then marginalize candidate scores over the programs. Only retain alternatives supported by the language. Retrieval must not rewrite "black" into "white" to fit an attractive candidate.

Research hypothesis for P2: ambiguities should receive additional parsing/visual computation only if their alternatives change the candidate ordering or temporal target. Compare with fixed top-M parses at equal total computation. Latent parsing and uncertainty are established; this decision criterion and its measured retrieval effect would need to carry any contribution.

Evaluate predicate/argument correctness, attachment, coreference, target/context selection and temporal/negation scope separately. Syntax validity is not semantic accuracy. Keep a manually checked oracle parser condition.

**Recommended first search direction: resolve consequential assignment gaps**

Use the same visual representations, interval proposals, candidates and holistic baseline for all variants. Perception supplies evidence records with video/time, entity track, predicate scores, visibility and source frame. Full actor binding requires raw frames or track features; clip embeddings support only a limited temporal proxy.

Let G be a parse, v a video and I an output interval. Let Z_G(v,I) contain allowed joint assignments of query entities and event intervals to evidence. For factor scores s_j in [0,1] and nonnegative weights summing to one, define:

```text
C_G(v,I) = max over z in Z_G(v,I) of sum_j w_j s_j(z)
R_G(v,I) = sum_j w_j max over z_j in Z_j(v,I) of s_j(z_j)
```

Each Z_j is a relaxation containing every projection of a legal joint assignment. C requires compatible bindings; R lets each factor choose its evidence independently. Therefore C <= R. Define C=0 for a proven empty feasible set with these nonnegative scores.

This inequality is elementary relaxation theory, not a new theorem. It applies only to the defined model and evidence space, not semantic truth.

If a solver has found a feasible assignment with score L but has not proved the optimum, L <= C <= R. A large R-L can mean either inconsistent composition or an unfinished search. It must not be reported as a confirmed binding error without sufficient inference or annotation.

Mix the compatible score with a frozen whole-query score using a validation-selected, nonnegative weight. Keeping this mixture monotone preserves bounds if the holistic term is fixed. For multiple parses, a fixed nonnegative weighted sum also preserves bounds; arbitrary learned fusion need not.

Proposed search:
1. Obtain a broad candidate set through holistic and predicate retrieval. Use a union to avoid losing candidates solely because one noisy detector missed a predicate. Measure shortlist recall.
2. Generate intervals without ground-truth timestamps. Compute cheap relaxed scores and feasible assignment scores.
3. Maintain lower/upper score bounds for unresolved candidates. Discard a candidate only when a valid upper bound cannot enter the requested top-K under the defined ranking rule.
4. Among candidates that can still affect the ranking, choose the next assignment expansion according to expected reduction in top-K ranking error per unit cost.
5. Return the best compatible moments at the budget limit. Apply the benchmark's duplicate/NMS rules consistently across methods.

Here, ranking error for fixed-evidence scheduling means disagreement with exhaustive inference under the same fixed score, not disagreement with semantic ground truth. Bounds certify only inference under that score. Budget-limited outputs remain approximate unless the bounds certify their ordering. Scheduling alone cannot outperform exhaustive evaluation of the identical objective; its goal is better approximation at a fixed cost. Any policy trained directly on ground-truth VCMR errors is a separate heuristic, not a consequence of these bounds.

The simple top-K stopping condition applies to pre-NMS rankings. It is unsafe for final post-NMS top-K: high-scoring overlapping intervals may suppress one another, making a lower-scoring unexamined interval necessary. Either declare a pre-NMS proposal cap and limit the claim to that set, or derive suppression-aware conservative bounds. Test this counterexample explicitly.

The contribution to investigate is the binding-sensitive allocation policy across corpus candidates, not graph search itself. Start with fixed evidence and exact small-instance enumeration as a reference. Compare learned allocation with rare-predicate-first, best-upper-bound-first, largest-gap-first, uniform allocation and generic entropy selection.

A more ambitious extension acquires new actor crops, denser frames or boundary evidence. New observations change the score model: old bounds may become invalid. Recompute them or explicitly treat this extension as a heuristic policy. Learned confidence intervals are not automatically admissible search bounds.

An ANN shortlist limits coverage. Exact search within it is not globally exact VCMR. A global guarantee would require covering/bounding excluded videos and the entire declared interval space. Continuous timestamps outside the candidate grid remain a localization limitation.

Why it could help: whole-query similarity can admit partially matching videos; independent factor search can combine incompatible evidence; uniform verification wastes computation on easy candidates. The hypothesis predicts gains on compositional queries and better accuracy/cost tradeoffs, not automatic gains on every dataset.

Why it could fail: tracking can be worse than holistic features; most benchmark queries may not require binding; the shortlist may omit the answer; graph inference may cost more than a strong reranker; a gap may identify an obvious negative that deserves no extra work.

**Second direction: correct for accidental combinations of evidence**

Even compatible matching may assign a high score by chance when a long video contains many people or repeated events. Test a reference score which preserves unary evidence distributions while selectively changing the actor/time associations required by the query.

For evidence E, construct restricted alternative associations E^pi, and compute:

```text
B(q,v,I) = C(q,E,I) - mean_pi C(q,E^pi,I)
```

Use absolute compatibility and B as separate inputs to a small calibrated ranker alongside the unchanged holistic score. The earlier bounds do not automatically apply to this ranker; derive new bounds or use it as an un-certified reranker. Compare against simple length/entity-count normalization. B alone is not a relevance probability: a common but correct event can have low B.

For video-level maximum scores, every surrogate must repeat the same maximization over assignments and candidate intervals as the real score. Otherwise the reference ignores selection bias. For interval-level scores, condition the reference construction on comparable interval duration and opportunity count.

The parser can identify which associations a reference may disrupt, but the operation requires explicit design:
- A global permutation of track IDs is only relabeling and changes nothing for a correct matcher.
- Permuting every temporal channel together also preserves conjunction.
- Selectively reassigning predicate trajectories across matched tracks or shifting one event stream can disrupt binding while retaining some unary statistics.
- Independent frame shuffling destroys persistence and is an inappropriate sole control.
- Single-actor cases, repeated actions, occlusion and interactions need dedicated rules; a valid reference may not exist.

Call this a reference-normalized feature until assumptions and calibration are demonstrated. Preserving marginals does not establish exchangeability, causality or a false-positive guarantee. Prior randomized graph matching and permutation calibration are explicitly acknowledged in literature.md.

Potential contribution: a precise, validated reference for video actor/time bindings that improves VCMR ranking beyond existing joint matching at equal visual quality. If it adds nothing over C or cheap opportunity normalization, abandon it.

**Third direction: resolve only consequential parse ambiguity**

Hold the search model fixed. Predict whether alternative text-supported parses lead to different top moments. Spend another parsing/verification step only when the ranking is sensitive, then marginalize remaining uncertainty. Never discard a required query attribute because it reduces recall.

Example: "the man watches the woman with binoculars" may attach the instrument to different participants. Both interpretations can be represented; the model should not silently choose the easier match. In benchmark evaluation the system must emit a defined ranking without interactive clarification.

Potential benefit: fewer cascading parser errors at the same language-model budget. Risks: extra parses can inflate false matches and latency, benchmark queries may be too simple, and latent-variable parsing has extensive prior art. Treat this as an ablation or backup until the oracle-parser gap shows room for improvement.

**Scope and supervision**

The initial controlled study can use available training timestamps to train/calibrate localization and ranking. Label it supervised VCMR; it is not a weakly supervised claim. Start frozen wherever possible to isolate search. A later weakly supervised track must exclude training timestamps and be compared separately.

Do not simultaneously invent a new detector, parser, retriever, boundary network and RL policy. Establish whether the proposed ranking/search mechanism helps when the other components are fixed, then add complexity only where the ablations justify it.
