# Research revision notes

Snapshot: 2026-11-13. This document records mathematical diagnostics and a targeted prior-art check. It does not establish exhaustive novelty. Run artifacts and decision files remain the authoritative record of completed experiments.

## Evidence from attempt 1

The association-reference method failed its prespecified advantage gate on the retained development-confirmation cohort of 4,991 queries. Both the candidate and the strongest raw-conjunction control obtained R@1 at temporal IoU 0.5 of **41.474654%**, a difference of **0.000 percentage points**, with paired 95% interval **[-0.080128, +0.080080] percentage points**. See `runs/attempt1/decision.json`.

The reference was numerically constant within each `(video, candidate duration)` group: median within-group standard deviation was about **1.4e-8**. Only **14 / 4,991** top-ranked predictions changed relative to the conjunction control. The raw-candidate oracle at IoU 0.5 was **76.838309%**. This leaves ranking headroom, but does not show that the available CLIP/parser evidence can recover it. The oracle is a diagnostic using labels, never an inference method.

The first reference maximized each surrogate over every window of a candidate's duration. Its reference therefore depended on video and duration, not on the candidate's position. This is a mathematical limitation of that construction, not proof that association information is inherently useless.

## Why ordinary local background subtraction is a control

Subtracting mean similarity in surrounding context from mean similarity inside a candidate is closely related to established Outer-Inner-Contrast scoring. AutoLoc introduced an Outer-Inner-Contrastive loss in temporal action localization, and later work explicitly used outer-inner-contrast as proposal confidence:

- [AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos, ECCV 2018](https://www.openaccess.thecvf.com/content_ECCV_2018/papers/Zheng_Shou_AutoLoc_Weakly-supervised_Temporal_ECCV_2018_paper.pdf).
- [Learning Action Completeness From Points for Weakly-Supervised Temporal Action Localization, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Learning_Action_Completeness_From_Points_for_Weakly-Supervised_Temporal_Action_Localization_ICCV_2021_paper.pdf).

Applying this operation to parsed CLIP channels can be a useful controlled comparison. It is not, by itself, an original search algorithm. Requiring every attribute to exceed its context also creates a concrete failure case: a black suit may persist throughout a scene while only the standing action changes. A minimum across all inside-minus-outside predicate scores can suppress that correct event because the suit's contrast is approximately zero.

Background disambiguation and temporal context debiasing are also existing research themes. For example, [Background-Aware Moment Detection for Video Moment Retrieval, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Jung_Background-Aware_Moment_Detection_for_Video_Moment_Retrieval_WACV_2025_paper.pdf) uses other queries from the same video, filtered by temporal/semantic relationships, to disambiguate target evidence. These are relevant comparators rather than mechanisms to present as our invention.

## Registered attempt 2: conditional residual shifts

The parent experiment registers and evaluates this global conditional construction as attempt 2. Let `z_j(t)` be the standardized trace for parsed predicate `j`, and `h(t)` the holistic query trace. Fit, using only observed features within the video and no moment labels,

    z_j(t) = a_j + b_j h(t) + r_j(t).

Preserve `a_j + b_j h(t)` at its original time, independently circular-shift each residual trace, and compute the surrogate conjunction over the same proposed interval. Unlike attempt 1, do not maximize the reference over other locations.

For soft conjunction `J` and candidate interval `I`, the proposed feature is

    E(I) = J_I(z) - mean_surrogates J_I(a + b*h + shifted_residuals).

Its defensible interpretation is a heuristic estimate of incremental simultaneous association beyond observed whole-query support. It is not a causal effect, calibrated significance level, or probability of query correctness.

### Exact algebraic degeneracy and required control

Define

    softmin_tau(x_1,...,x_P) = -tau * log(mean_j exp(-x_j/tau)).

Translation equivariance gives

    softmin_tau(x + c*1) = softmin_tau(x) + c.

If all fitted slopes equal a shared `b`, the surrogate conjunction is

    b*h(t) + softmin_tau(a_j + r_j(t + shift_j)).

Under independent uniformly sampled shifts over the complete circular video, the expected residual term is a constant `C_v` at every time. Therefore the expected same-window surplus becomes exactly

    E(I) = J_I(z) - b*mean_I(h) - C_v.

For ranking equal-video candidates, this adds no information beyond raw conjunction and the holistic trace; across videos, it can additionally supply a video-dependent calibration term. A finite Monte Carlo average can display apparent position variation even when the expected reference is constant. Such variation must not be confused with a discovered semantic mechanism.

With unequal slopes, the expected global reference is a pointwise function

    F_v(h(t)) = E_R softmin_tau(a_j + b_j*h(t) + R_j),

followed by averaging over the candidate. It can supply nonlinear holistic-score recalibration, but this alone does not demonstrate extra information about actor identity or semantic binding.

Consequently include a shared-slope linear correction control and measure how well candidate correction scores can be predicted from existing conjunction, holistic support, video identity, and duration. Disclose the correlation and any remaining residual variation. A gain that a matched recalibration control explains should not be attributed to a new association mechanism.

### Statistical and perceptual limitations

- Regression makes residuals linearly uncorrelated with `h` in the fitted sample; it does not establish conditional independence, exchangeability, or a correct conditional distribution.
- The holistic trace may encode true event alignment. Holding it fixed can appropriately test incremental evidence, but also leave little usable residual signal. Keep the original holistic/baseline evidence in the final ranker.
- Circular shifts can join unrelated shots and assume temporal stability that edited TV clips do not possess.
- Residuals may retain common scene, camera, or actor-visibility changes. Removing one holistic channel does not remove all common causes.
- Distinct text phrases can still have nearly redundant CLIP traces. Minimum/softminimum fusion can be dominated by the weakest or noisiest phrase.
- A feature-level temporal association score cannot establish that the same person satisfies all requested actions and attributes. No actor tracks are available in this experiment.
- Preserve the gate against forcing simultaneous conjunction onto sequential or negated queries. Short/flat residual traces should fall back to the unchanged baseline.

Conditional and local permutation are established statistical ideas, and their validity depends on assumptions not demonstrated here. They are inspiration from outside video retrieval, not an originality claim or a source of automatic error guarantees:

- [The Conditional Permutation Test for Independence While Controlling for Confounders, JRSS B 2020](https://academic.oup.com/jrsssb/article/82/1/175/7056014).
- [Local permutation tests for conditional independence, arXiv 2112.11666](https://arxiv.org/abs/2112.11666).

## Possible attempt 3, only after documented attempt-2 failure

A candidate-specific local residual reference is the proposed next revision, not an already completed experiment. For candidate `I=[s,e]` of length `L`, define a local neighborhood `B(I)=[s-L,e+L]`, clipped to observed support. Preserve the same fitted holistic component, independently shift residuals within `B(I)`, and score only the original `I`.

    E_local(I) = J_I(z)
                 - E_shift J_I(a + b*h + residuals shifted within B(I)).

The candidate-specific residual distribution responds to the constant-reference diagnosis and reduces dependence on a global stationarity approximation. In the equal-slope limit the correction becomes

    J_I(z) - b*mean_I(h) - C_B(I),

where the recombined-residual reference can vary between local neighborhoods. Unlike ordinary scalar inner/outer mean subtraction, `C_B(I)` mixes independently recombined predicate residuals through the conjunction. Whether this difference matters must be measured rather than asserted.

Required comparisons include the registered raw-conjunction and whole-query controls, local unconditional permutation, global conditional permutation, ordinary whole-query context contrast, and an appropriately matched recalibration control. Preserve identical candidates, features, query cohorts, postprocessing, and tuning access. Handle clipped neighborhoods and inadequate residual variation explicitly. The exact control set and gate must be frozen before confirmation outcomes are inspected.

No fourth substantive method is authorized. Engineering bug fixes are separate from new hypotheses, but must not be used to disguise extra adaptive attempts. The latest `runs/attempt*/decision.json` files take precedence over this snapshot's planned next steps.
