# Execution plan, 2026-11-13
One dataset: TVR. Original data /home/mohammad/VisualSearch/tvr; all experiment data copies and outputs here.
Conda environment: envs/vcmr, cloned from known-working gprvr then independently managed.
Original validation is the final held-out validation evaluation, not an independent test. Develop only on video-disjoint subsets carved from the original training set.
Steps: data and feature audit; frozen protocol and manifests; compatible text encoder and parser verification; baseline plus diagnostics; at most three substantive candidate mechanisms with failure evidence and revision log; final fixed-protocol evaluation; comparison/report.
Research success requires improvements over matched local baselines, with uncertainty and reproducible artifacts. Published SOTA with different features/modalities is contextual until reproduced. No promised novelty or improvement.
A failed method means its prespecified primary development metric does not improve over its comparison baseline with supporting paired uncertainty, or a required mechanism fails oracle/sanity checks. Engineering fixes do not consume method attempts. No fourth substantive candidate.
Do not use held-out validation to pick hypotheses, thresholds or checkpoints. Do not append GT videos to real retrieval. Record all model compute and feature additions.
