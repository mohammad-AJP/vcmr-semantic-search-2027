# Reusable research-agent prompt

Copy the following instructions to an independent research agent. Paths describe this system; verify them before use. For an independent rerun, choose your own experiment directory so concurrent agents do not overwrite one another. For an explicitly assigned resume, inspect the current artifacts and running jobs first.

---

I want you to conduct a reproducible research experiment for video corpus moment retrieval (VCMR). A user supplies a query such as "a man standing in a black suit". A semantic parser should represent entities, actions or states, attributes, argument roles, relevant temporal relationships, and uncertainty. A search/ranking algorithm should return globally ranked `(video_id, start_seconds, end_seconds, score)` predictions. A separate video ranking may aggregate each video's strongest moment.

Aim to develop a genuinely new, useful mechanism that improves retrieval. Do not copy a mechanism from VCMR/video-retrieval papers and present it as original. Read relevant literature, credit established ingredients and baselines, investigate close prior art, and explain the narrow proposed contribution. Inspiration from statistics, mathematics, physics, and other fields is welcome. Neither complete novelty nor beating state of the art can be guaranteed in advance: report the actual evidence honestly.

First give me a concrete plan, then execute it step by step. After each meaningful stage, post a short report in the chat describing what worked, what was learned, and what comes next; include a parser example. Continue autonomously rather than stopping for approval after every stage. I authorize dependency installation, implementation, meaningful tests, inference, and experiments within the isolated workspace.

Keep my existing repository completely clean. The original repository is:

    /home/mohammad/Documents/grounded-prvr

The source datasets were found at:

    /home/mohammad/VisualSearch/charades
    /home/mohammad/VisualSearch/activitynet
    /home/mohammad/VisualSearch/tvr
    /home/mohammad/VisualSearch/annotations

The existing research workspace is:

    /home/mohammad/Documents/vcmr-semantic-search-2027

All new documents, source code, copied/downloaded data, model weights, environments, package/model caches, temporary files, logs, tests, and results belong inside the experiment directory. Treat original dataset files and the original repository as read-only. Create a dedicated Conda environment inside your experiment root and install packages only there; do not change the system environment or the original project's environment. For an independent run, use a distinct root such as `/home/mohammad/Documents/vcmr-semantic-search-2027-independent` and retarget all environment/cache paths accordingly.

Use only **TVR** for this experiment. It was selected because it provides a conventional corpus moment retrieval benchmark with timestamped queries, available features, and a reproducible contemporary baseline. Do not expand to Charades or ActivityNet during this run. Source downloads contain three datasets, but features and annotation completeness must be distinguished from raw-video availability. No raw videos were found in the initial dataset audit. Therefore the current experiment uses fixed temporal video features; it cannot honestly claim visual actor tracking or object-level identity verification.

## Required protocol and safeguards against misleading results

1. Audit actual files, feature dimensions, query/video IDs, durations, timestamps, and official split definitions. Local PRVR `val` and `test` caption files are aliases; do not call them independent evaluation sets. Preserve provenance and hashes.
2. Use official TVR validation as the final evaluation set, not as an independent test set. Develop the new mechanisms on frozen, episode-disjoint subsets of original training annotations. Keep selection and confirmation access distinct. Any adaptive reuse of confirmation outcomes across revisions must be disclosed.
3. The pretrained baseline has already seen original training examples, and its author selected its checkpoint on official validation. Development subsets therefore protect new-ranker selection only; they do not make the pretrained baseline itself independent of those examples. Disclose this limitation.
4. Do not inject the ground-truth video into retrieval candidates. Do not use target timestamps or video labels while producing predictions. Evaluate labels only after predictions exist. Clearly label oracle diagnostics.
5. Reproduce a strong released baseline and compare all proposed methods with matched candidates, visual/text features, supervision, corpus, postprocessing, and computational budget. Keep extra CLIP evidence available to appropriate controls. Report corpus Recall@K at temporal IoU 0.5 and 0.7, confidence intervals where appropriate, candidate coverage, computation, and known limitations.
6. Use source-grounded parsing with exact text offsets and explicit entity argument references. Preserve the whole query. Do not force sequential events to occur simultaneously, treat missed evidence as proof of absence, or confuse linguistic identity references with verified visual binding.
7. Establish that a new feature provides information beyond simple alternatives: whole-query similarity, raw conjunction, ordinary context subtraction, and matched score recalibration. Include the specific controls needed to test the claimed mechanism.
8. Treat unsuccessful results as legitimate research outcomes. Do not fabricate scores, cherry-pick examples or metrics, tune on final validation, or claim universal SOTA from a changed protocol.

## At most three substantive attempts

Try a first registered mechanism. Freeze its design, allowed tuning, strongest controls, primary metric, and failure criterion before confirmation evaluation. If it fails with clear numerical or analytical evidence, stop improving that version, explain the failure, formulate one substantive revision, and implement it. Repeat only once more if necessary: **at most three substantive candidate mechanisms total**. Then stop proposing or implementing additional variants and report all results. Engineering fixes do not consume attempts, but cannot be used to disguise additional hypothesis searches. Respect the current attempt ledger when resuming; do not reset the counter.

A method fails when its prespecified primary development metric does not show a supported advantage over its matched strongest control, or when required sanity/oracle tests expose a fundamental mechanism failure. A negative experiment is evidence against that implementation under the tested conditions, not proof that the broad scientific question is impossible. If all three fail, provide the final negative findings and their implications rather than claiming SOTA.

## Existing environment and artifacts to inspect when resuming

From the current experiment root:

    source scripts/environment.sh
    "$VCMR_PYTHON" -m pytest -q tests

The isolated interpreter is `envs/vcmr/bin/python`. Environment settings route Conda, pip, Hugging Face, Torch, CLIP, Python bytecode, and temporary storage under the experiment root. Verify these settings instead of activating an unrelated environment.

Read `README.md`, `docs/execution_plan.md`, `docs/literature.md`, `docs/design.md`, `docs/evaluation.md`, `docs/demul_reproduction.md`, `docs/research_revision_notes.md`, `manifests/tvr_protocol.json`, all `configs/attempt*_preregistration.json`, and the latest `runs/attempt*/decision.json` before resuming. Files may have progressed beyond the snapshot below.

Parser API:

    from vcmr.parser import parse_query
    result = parse_query("a man standing in a black suit")

It is a deterministic spaCy dependency-parser baseline, not a novel parsing model. It outputs `man(e0)`, `suit(e1)`, `black(e1)`, `stand(e0,I0)`, and a lexical-rule `wear(e0,e1,I0)`, with source offsets and limitations. The parser's semantic tests passed, but this is not a corpus-level accuracy claim.

The exact OpenAI CLIP ViT-B/32 encoder was checked against 128 original training-query embeddings: median cosine **0.999999089**, minimum **0.999996600**. Phrase features are normalized float32 arrays, keyed by string `desc_id`, with at most six rows and 512 columns. Row zero is the unchanged whole query. Relevant files are:

    src/vcmr/parser.py
    src/vcmr/clip_encoding.py
    scripts/encode_queries.py
    data/tvr/phrases_{dev_select,dev_confirm,official_val}.h5
    data/tvr/parses_{dev_select,dev_confirm,official_val}.jsonl
    runs/clip_encoder_verification.json

Encoding covers 5,045 selection queries, 5,010 confirmation queries, and 10,895 official-validation queries. Encoding official-validation text did not compute retrieval metrics or tune a method.

The chosen baseline is the released DeMUL TVR checkpoint, reconstructed with a strict, documented compatibility adapter because the released checkpoint and current source disagree. Keep the upstream source and license intact. Use the independent wrapper `scripts/reproduce_demul.py`; do not imply that our reconstruction is numerically identical to the original author implementation. The downloaded CONQUER feature release supplies the baseline's ResNet+SlowFast visual features and subtitle/query features. Relevant source, checkpoint, mappings, checksums, safe loading details, and protocol are documented in `docs/demul_reproduction.md`.

The published DeMUL TVR validation R@1 reference is **21.37 / 11.92** at temporal IoU **0.5 / 0.7**. The reconstructed baseline obtained **21.13 / 11.78** over 10,895 queries and the original 2,179-video validation corpus. Exact reproduction has not been achieved. Baseline validation was evaluated to assess reproduction; do not use those outcomes or new method validation outcomes to tune revisions. Upstream candidate truncation before NMS changes high-K results; both protocols are saved. All new mechanisms must use the same 600 raw proposals, NMS, and evaluator as their matched reconstructed baseline. Beating that reconstruction alone does not establish overall SOTA or superiority to the author's reported result.

Supplied training candidate rankings cover only **5,022** selection queries and **4,991** confirmation queries. Use the explicitly retained cohorts and coverage manifest for baseline-based comparisons:

    data/tvr/splits/demul_dev_select.jsonl
    data/tvr/splits/demul_dev_confirm.jsonl
    manifests/demul_train_ranking_coverage.json

Snapshot of the attempt ledger:

- **Attempt 1 completed and failed.** Its shifted-reference score and strongest raw-conjunction control both obtained confirmation R@1@0.5 **41.474654%** over 4,991 queries; paired difference **0.000 pp**, 95% interval approximately **[-0.080, +0.080] pp**. Its reference was effectively constant within video/duration. See `runs/attempt1/decision.json` and `diagnostics.json`. Do not rerun it as a new first attempt.
- **Attempt 2 is being implemented/registered:** preserve each predicate trace's fitted dependence on the holistic query trace, independently shift residuals, and score the same candidate interval. Include a shared-slope correction control because the conditional surplus can collapse algebraically to reweighting existing evidence. Inspect current configs/runs for its latest status.
- **Attempt 3 is only a possible revision if attempt 2 fails:** candidate-neighborhood-local residual references, with its own preregistered controls and analytical tests. Check the latest ledger before acting; it may already have run.

## Background execution and reconnection

Long experiments should run in an OS session detached from the chat/terminal, with logs, explicit completion markers, and a lock preventing duplicate pipelines. Inspect existing processes and `runs/background/status.json` before launching work. Do not assume a saved PID is still live. The existing first-attempt pipeline can be started or resumed with:

    bash scripts/start_background.sh
    cat runs/background/status.json
    tail -f runs/background/launcher.log

That pipeline is intentionally bounded at the first-attempt decision boundary; launching it does not automatically perform later research revisions. Inspect `scripts/background_pipeline.py`, the registered later-stage commands, and current logs. Extend or launch only the remaining authorized stages. Successful stages have `.done.json` records; distinguish complete files from temporary/partial outputs before reuse. Detached processes survive a chat or terminal disconnection, not a machine reboot. After reboot, resume from verified artifacts instead of silently starting everything again.

Post short progress reports while work runs and continue through completion. At the end, provide a self-contained report with the chosen dataset and exact protocol, parser example, every attempted idea and its failure/success evidence, matched baseline comparisons, published SOTA context, limitations, reproduction commands, and links to artifacts. State plainly whether a novel mechanism or a SOTA improvement has actually been demonstrated.
