#!/usr/bin/env python3
"""Render saved measurements without choosing new parameters or hiding failures."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    final=json.loads((ROOT/'runs/final_validation/results.json').read_text())
    frozen=json.loads((ROOT/'runs/final_validation/frozen_configs_before_validation.json').read_text())
    results=final['protocols']['600_before_nms']['results'];short=final['protocols']['100_before_nms']['results']
    candidates=[x for x in results if x.startswith('attempt')];strong=final['strongest_control_fixed_from_selection']
    lines=['# Final TVR experiment results','',f"Generated {datetime.datetime.now(datetime.timezone.utc).isoformat()}.",'',
           'All three permitted method attempts are complete. Method weights were selected on training-derived development data and frozen before this final evaluation. No fourth method was attempted.','',
           '## Outcome','']
    decisions=[]
    for i in range(1,4):
        r=json.loads((ROOT/f'runs/attempt{i}/confirmation/results.json').read_text())
        control=r['strongest_control_fixed_from_selection']
        comparison=r['candidate_minus_controls_R1@0.5'][control] if i==1 else r['candidate_minus_strongest_control_R1@0.5']
        passed=comparison['delta_b_minus_a_pp']>=.25 and comparison['ci95_pp'][0]>0
        decisions.append((i,r,control,comparison,passed))
    if not any(d[-1] for d in decisions):
        lines+=['**None of the proposed mechanisms passed the prespecified development improvement gate. This experiment does not establish a new state of the art.** A small change in an aggregate metric is insufficient when the proposed feature is unused or fails to outperform simpler controls.','']
    else:
        lines+=['At least one development comparison passed its pilot gate. Final validation and protocol differences below determine the limits of the claim; novelty and universal state-of-the-art status are not established.','']
    lines+=['## Final official validation','',f"All **{final['n_queries']:,} queries** and **2,179 corpus videos**. Each query uses the same independent HERO top-10 video candidates and the same 600 DeMUL moment proposals. About {100*final['eligible_queries']/final['n_queries']:.1f}% of queries satisfy the conservative semantic eligibility rule. Values are percentages.",'',
            '| Method | R1@0.5 | R1@0.7 | R5@0.5 | R10@0.5 | R100@0.5 |','|---|---:|---:|---:|---:|---:|']
    display=list(dict.fromkeys(['demul','whole_query_control','joint_support_control',strong,*candidates]))
    for name in display:
        m=results[name]['metrics']['full'];lines.append('| '+name+' | '+' | '.join(f'{m[k]:.2f}' for k in ['R1@0.5','R1@0.7','R5@0.5','R10@0.5','R100@0.5'])+' |')
    lines+=['','The names of frozen attempted systems do not imply that their new feature was selected. Its coefficient was:','']
    for i,name in enumerate(candidates,1):
        idx={1:3,2:6,3:9}[i];weight=results[name]['coefficients'][idx]
        lines.append(f'- Attempt {i}: **{weight:g}**'+(' (proposed feature disabled by development selection).' if weight==0 else '.'))
    lines+=['','### Published reference comparison','','| Published reference | TVR validation R1@0.5 | R1@0.7 | Interpretation |','|---|---:|---:|---|',
            '| [DeMUL, TCSVT 2026](https://github.com/iLearn-Lab/TCSVT26-DeMUL/blob/main/checkpoints/demul_tvr/metrics.json) | 21.37 | 11.92 | Author-reported released checkpoint; strongest directly reproduced reference in this experiment. |',
            '| [GenSpan, June 2026 preprint, Table 1](https://arxiv.org/html/2603.22121v2) | 18.77 | 12.12 | Author-reported VCMR validation; generation/subtitle pipeline and features differ. Not reproduced here. |','',
            'DeMUL reconstruction reaches 21.13 / 11.78; its architecture/checkpoint compatibility adapter consumes every tensor but does not exactly reproduce author numbers. Our residual methods add CLIP features; matched whole-query and parser controls separate this added-input benefit from the proposed mechanism. These references are not an exhaustive leaderboard: recent CHDL results were identified but their complete numerical protocol was unavailable. No universal SOTA claim is made.','',
            '### Candidate-count sensitivity','',
            'The released evaluation code mutates the raw candidate list to 100 before NMS, despite a saved setting of 600. The primary internal comparison gives every variant 600 proposals. The following separate run keeps the original first 100 proposals for every variant before reranking, matching that effective candidate budget. Neither cohort nor model weights change.','',
            '| Method | R1@0.5 (100 before NMS) | R1@0.7 | R100@0.5 |','|---|---:|---:|---:|']
    for name in display:
        m=short[name]['metrics']['full'];lines.append(f"| {name} | {m['R1@0.5']:.2f} | {m['R1@0.7']:.2f} | {m['R100@0.5']:.2f} |")
    lines+=['','### Paired final comparisons','',f'Candidate minus the strongest control chosen on development selection: `{strong}`. Video-cluster percentile bootstrap, 2,000 resamples; percentage-point differences. These intervals are descriptive and are not corrected for all adaptive method comparisons.','',
            '| Attempt | Final R1@0.5 difference | 95% interval |','|---|---:|---|']
    for name in candidates:
        b=final['protocols']['600_before_nms']['candidate_comparisons_R1@0.5'][name][strong]
        lines.append(f"| {name} | {b['delta_b_minus_a_pp']:+.3f} | [{b['ci95_pp'][0]:+.3f}, {b['ci95_pp'][1]:+.3f}] |")
    lines+=['','## Three-attempt ledger','','Development confirmation contains 4,991 queries. Success required at least +0.25 percentage points and a strictly positive 95% paired interval against the strongest control selected on the other development cohort. Confirmation was reused to guide revisions, so later confirmations are exploratory; final official validation was reserved until all variants were frozen.','',
            '| Attempt | Mechanism | Confirmation R1@0.5 | Gain over selected control | 95% interval | Gate |','|---|---|---:|---:|---|---|']
    descriptions={1:'Video-wide opportunity reference',2:'Conditional local score with global residual reference',3:'Candidate-neighborhood conditional residual reference'}
    for i,r,control,b,passed in decisions:
        name=next(n for n in r['results'] if n.startswith(f'attempt{i}_'));m=r['results'][name]['metrics']['full']
        lines.append(f"| {i} | {descriptions[i]} | {m['R1@0.5']:.3f} | {b['delta_b_minus_a_pp']:+.3f} | [{b['ci95_pp'][0]:+.3f}, {b['ci95_pp'][1]:+.3f}] | {'Pass' if passed else 'Fail'} |")
    lines+=['','Attempt 1 could not distinguish equal-duration locations through its reference: that term was constant within video/duration and changed only 14 confirmation top predictions versus raw joint support. Attempt 2 selected zero weight for the proposed conditional feature. Attempt 3 tested whether neighborhood-specific residual distributions repaired the limitation. These are pilot failures under the measured conditions, not impossibility proofs.','',
            '## Data, parser, and limitations','',
            '- Source data: `/home/mohammad/VisualSearch`. Only TVR was used. Independent copies, acquired baseline features, code, environment, caches, predictions, and documents are under this workspace.',
            '- Official TVR validation: 10,895 queries / 2,179 videos. Development selection: 5,022 retained queries; confirmation: 4,991. Missing train HERO rankings are explicitly recorded, without ground-truth candidate injection.',
            '- The parser extracts source-anchored entities, attributes, actions, roles, and temporal relations. Example: “a man standing in a black suit” has man and suit entities, black attached to suit, standing attached to man, and a wear relation. It does not invent visual tracks.',
            '- Frozen CLIP text embeddings match supplied full-query embeddings with median cosine 0.999999089 on 128 training examples. New phrase embeddings are computed with the same encoder.',
            '- These are feature-based reranking experiments over a fixed search candidate pool, not a new full-corpus index or a trained end-to-end replacement. No raw videos or actor tracks were available. Global frame features cannot prove that multiple predicates concern the same person.',
            '- The DeMUL checkpoint was pretrained on the original training set, including our development examples. The official benchmark also shares source episodes between train/validation. Development performance is optimistic and cannot be compared numerically with validation performance.',
            '- Local validation/test feature annotations are aliases; no claim is made about held-out test results. Temporal features use nominal 1.5-second spacing, with flagged length/duration discrepancies retained and documented.',
            '- Circular-shift surrogate scores are heuristic references, not valid causal tests or calibrated probabilities. Novelty is unproven; semantic parsing, OIC, regression, and permutation statistics are established methods and credited in the revision notes.',
            '- Baseline checkpoint/source differences and pre-NMS candidate handling limit external comparability. Full metrics, visual-only subgroup results, all frozen weights, and bootstrap outputs are saved; no failed attempt is omitted.','',
            '## Reproduce and resume','',
            '```bash','cd /home/mohammad/Documents/vcmr-semantic-search-2027','source scripts/environment.sh','"$VCMR_PYTHON" -m pytest -q tests','cat runs/background/status.json','```','',
            'Long jobs use detached OS sessions with standard input disconnected and file-backed logs. Completed stages have markers and failed stages stop dependent work. Terminal/chat disconnection does not kill these jobs; machine reboot does. Environment exports are in `manifests/conda_explicit.txt` and `manifests/pip_freeze.txt`.','',
            'Artifacts: [final JSON](../runs/final_validation/results.json), [baseline reproduction](demul_reproduction.md), [revision reasoning and prior art](research_revision_notes.md), [reusable agent prompt](reusable_agent_prompt.md), [parser examples](../examples/implemented_parser_examples.json), [original source check](../manifests/original_source_execution_check.json).','']
    (ROOT/'docs/final_results.md').write_text('\n'.join(lines))
    ledger={'completed_substantive_attempts':3,'no_fourth_attempt':True,'decisions':[{'attempt':i,'passed_gate':p,'comparison':b} for i,r,c,b,p in decisions],'final_report':str(ROOT/'docs/final_results.md')}
    (ROOT/'runs/experiment_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    readme=ROOT/'README.md';s=readme.read_text();s=s.replace('The first method is being tested on training-derived development cohorts.','All three attempts and final official-validation comparisons are complete. See [the final results](docs/final_results.md).');s=s.replace('Research success and novelty remain unestablished.','The prespecified gates, all numerical outcomes, and limitations are recorded in the final report; novelty remains unestablished.')
    readme.write_text(s)
    print('Wrote',ROOT/'docs/final_results.md',flush=True)
if __name__=='__main__':main()
