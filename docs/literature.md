# Literature and novelty assessment

Review date: 2026-11-13. This is a targeted primary-source review, not proof of exhaustive novelty. Distinguish corpus moment retrieval (VCMR), grounding within a supplied video (VMR/VTG), whole-video retrieval, and spatiotemporal entity grounding (STVG). Scores from these tasks are not interchangeable.

| Source | Existing contribution relevant to this proposal | Consequence |
|---|---|---|
| [Visual Semantic Search, CVPR 2014](https://openaccess.thecvf.com/content_cvpr_2014/papers/Lin_Visual_Semantic_Search_2014_CVPR_paper.pdf) | Language semantic graphs, object tracks, appearance/motion/relations and constrained matching. | Objects plus actions plus graph search is longstanding prior art. |
| [Probabilistic Semantic Retrieval for Surveillance Videos with Activity Graphs, TMM 2018](https://arxiv.org/pdf/1712.06204) | Probabilistic activity grounding under detector and tracker errors; discriminative subgraph search. | Neither uncertain graph matching nor rare-predicate-first pruning is new. |
| [Fast Video Moment Retrieval, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Gao_Fast_Video_Moment_Retrieval_ICCV_2021_paper.html) | Semantic-role tree information distilled into efficient moment/query embeddings. | Structured language for efficient retrieval is already established. |
| [VISA, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Compositional_Temporal_Grounding_With_Structured_Variational_Cross-Graph_Correspondence_Learning_CVPR_2022_paper.pdf) | Object/action/event graph correspondence and compositional evaluation splits. | A compositional graph is not itself new; these splits are single-video grounding protocols. |
| [SQuiDNet, ECCV 2022](https://arxiv.org/abs/2210.08714) | Selective query-guided debiasing of object-associated retrieval shortcuts; TVR, ActivityNet, DiDeMo. | Object/action mismatch and debiasing are already studied directly in VCMR. |
| [TFVTG, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10687.pdf) | LLM subevent parsing, static/transition distinctions and temporal filtering of proposals. | Parser plus temporal constraints is a particularly close existing recipe. |
| [SHINE, ECCV 2024](https://arxiv.org/abs/2407.05118) | Compositional hard negatives and hierarchical saliency ranking. | Swapping verbs, objects or prepositions is not a standalone new contribution. |
| [VERIFIED, NeurIPS 2024](https://arxiv.org/html/2410.08593v1) | Filtered fine-grained static/dynamic queries for Charades-FIG, DiDeMo-FIG, ActivityNet-FIG. | Fine-grained matching has established corpus benchmarks. Original annotations are not FIG annotations. |
| [Object-Centric Framework for VMR, AAAI 2026](https://arxiv.org/html/2512.18448v1) | Scene-graph query parsing, object/relation tracklets and temporal object-state modeling; Charades-STA, QVHighlights, TACoS. | Direct competitor to the proposed perception/parser pipeline, but evaluated as single-video VMR. |
| [MAVIS, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1094/) | Structured semantic library, attribute indexing, query decomposition and logical candidate pruning. | Semantic indexed video search is already explicit prior art. Its whole-video retrieval task differs from VCMR. |
| [GenSpan, June 2026 preprint revision](https://arxiv.org/html/2603.22121v2) | VCMR query/subevent decomposition, subtitle cues, generated motion priors and temporal localization. | Ordered multi-action parsing is also used in recent VCMR. |
| [VideoSearch-R1, July 2026 paper](https://arxiv.org/html/2607.00446v1) | Iterative corpus retrieval, soft query refinement, verification and grounding on FIG datasets. | An iterative retrieve/verify/refine loop is not a new claim. |
| [CoMET, June 2026 preprint](https://arxiv.org/html/2606.15320v1) | Conditional multi-event grounding, hierarchical graph search and verification. | Conditions, empty answers and multi-event planning are occupied territory. |
| [CompSTVG, 2026](https://arxiv.org/html/2608.30584v1) | Compositional entity grounding, graph-based difficulty and constraint-controlled query generation. | Distinguishing entities with combinations of attributes/relations is a close neighboring task. |
| [Permute-then-Adapt, November 2026](https://arxiv.org/html/2611.10714) | Conditional permutation references for image/text adaptation. | Permutation-based similarity calibration is not generally new, even outside video retrieval. |

For the optional randomization idea, inspect [SAGA](https://doi.org/10.1093/bioinformatics/btl571) before any novelty claim: randomized graph-query significance is an older idea. The potential distinction would be a precise video association reference preserving temporal persistence, with demonstrated benefits for VCMR ranking. Merely applying a permutation test to graph scores is insufficient.

**Specific gaps worth testing**

1. Does independently matching predicates inflate the scores of corpus candidates whose evidence cannot be assigned to compatible actors and times?
2. Can the unresolved difference between relaxed and compatible scores identify which additional computation changes the top-ranked moments?
3. Does a carefully conditioned association reference improve ranking beyond ordinary compatible matching, duration/actor-count normalization, and stronger visual features?
4. Does keeping multiple legitimate parses help where committing to a single parse causes candidate loss?

The search found relevant pieces of all four ideas. It did not establish that the exact proposed mechanism has been published, nor that it is original. A successful pilot should be followed by a narrower citation search around the final mechanism and current releases.

**Benchmark comparison targets**

Use a reproducible traditional VCMR baseline such as [EventFormer](https://arxiv.org/html/2402.13566) or [DeMUL](https://github.com/iLearn-Lab/TCSVT26-DeMUL) under its specified protocol, plus the strongest reproducible contemporary method for the chosen track. GenSpan is a recent decomposition baseline. VideoSearch-R1 is relevant to the separate raw-video/FIG track. Include simple structured-search baselines and an ordinary VLM reranker with the same inputs and budget.

Do not call any one method universal SOTA. Different feature streams, pretraining, supervision, corpora, splits and IoU thresholds change the comparison. The [November 2026 CHDL publisher record](https://signalprocessingsociety.org/publications-resources/ieee-transactions-image-processing/2026/11/contextual-hashing-decoupled-localization-efficient-video-corpus-moment-retrieval) is also relevant for efficient corpus search; the complete numerical results were not accessible during this review.
