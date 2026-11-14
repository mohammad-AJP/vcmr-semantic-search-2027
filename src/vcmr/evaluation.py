"""Corpus moment retrieval evaluation with explicit video identity and temporal IoU.

Scores rank moments globally across videos. Missing predictions count as misses.
Ground-truth records are used only by evaluation, never inserted into predictions.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def temporal_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection/union of half-open temporal intervals; invalid spans score 0."""
    a0, a1 = float(a[0]), float(a[1])
    b0, b1 = float(b[0]), float(b[1])
    if not all(math.isfinite(x) for x in (a0, a1, b0, b1)):
        return 0.0
    if a1 <= a0 or b1 <= b0:
        return 0.0
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - intersection
    return intersection / union if union > 0 else 0.0


def _ground_truth(records: Iterable[dict]) -> dict[str, dict]:
    grouped = {}
    for record in records:
        qid = str(record['desc_id'])
        caption = ' '.join(str(record.get('desc', record.get('caption', ''))).split())
        if qid not in grouped:
            grouped[qid] = {'desc': caption, 'type': record.get('type'), 'positives': []}
        elif grouped[qid]['desc'] != caption:
            raise ValueError(f'Conflicting captions for duplicate desc_id {qid}')
        if 'positives' in record:
            positives = [(str(p.get('vid_name', p.get('video_id'))), p.get('ts', p.get('span'))) for p in record['positives']]
        else:
            video = str(record.get('vid_name', record.get('video_id')))
            spans = record.get('moments')
            if spans is None:
                spans = [record['ts']]
            positives = [(video, span) for span in spans]
        for video, span in positives:
            if span is None or len(span) != 2 or temporal_iou(span, span) != 1.0:
                raise ValueError(f'Invalid ground-truth span for desc_id {qid}: {span}')
            positive = (video, float(span[0]), float(span[1]))
            if positive not in grouped[qid]['positives']:
                grouped[qid]['positives'].append(positive)
    return grouped


def _canonical_predictions(predictions: Mapping) -> dict[str, list[list]]:
    canonical = {}
    for key, candidates in predictions.items():
        qid = str(key)
        if qid in canonical:
            raise ValueError(f'Duplicate prediction query identity after string conversion: {qid}')
        checked = []
        for item in candidates:
            if len(item) != 4:
                raise ValueError(f'Expected [video, start, end, score] for query {qid}')
            video, start, end, score = str(item[0]), float(item[1]), float(item[2]), float(item[3])
            if not all(math.isfinite(x) for x in (start, end, score)) or start < 0 or end <= start:
                raise ValueError(f'Invalid prediction for query {qid}: {item}')
            checked.append([video, start, end, score])
        # Stable ties retain supplied candidate order. Duplicate moments consume slots.
        canonical[qid] = sorted(checked, key=lambda x: -x[3])
    return canonical


def per_query_hits(predictions: Mapping, records: Iterable[dict], ranks=(1, 5, 10, 100), tiou_thresholds=(0.5, 0.7)) -> dict[str, dict[str, bool]]:
    truth = _ground_truth(records)
    ranked = _canonical_predictions(predictions)
    result = {}
    for qid, target in truth.items():
        candidates = ranked.get(qid, [])[:max(ranks)]
        ious = []
        for video, start, end, _ in candidates:
            ious.append(max((temporal_iou((start, end), (a, b)) for v, a, b in target['positives'] if v == video), default=0.0))
        result[qid] = {f'R{k}@{threshold:g}': bool(any(iou >= threshold for iou in ious[:k]))
                       for threshold in tiou_thresholds for k in ranks}
    return result


def evaluate_predictions(predictions: Mapping, records: Iterable[dict], ranks=(1, 5, 10, 100), tiou_thresholds=(0.5, 0.7)) -> dict:
    records = list(records)
    if not ranks or min(ranks) < 1:
        raise ValueError('Ranks must be positive integers')
    truth = _ground_truth(records)
    canonical = _canonical_predictions(predictions)
    hits = per_query_hits(canonical, records, ranks, tiou_thresholds)
    keys = [f'R{k}@{threshold:g}' for threshold in tiou_thresholds for k in ranks]
    groups = {'full': list(truth), 'visual_only': [qid for qid, x in truth.items() if x['type'] == 'v']}
    result = {}
    for name, qids in groups.items():
        result[name] = {'n_queries': len(qids), **{key: 100.0 * np.mean([hits[qid][key] for qid in qids]) if qids else None for key in keys}}
    result['diagnostics'] = {
        'n_input_gt_records': len(records), 'n_unique_gt_queries': len(truth),
        'n_missing_prediction_queries': sum(qid not in canonical or not canonical[qid] for qid in truth),
        'n_extra_prediction_queries': len(set(canonical) - set(truth)),
        'ranking': 'descending score, stable ties; duplicate moments consume rank slots',
        'units': 'percent', 'tiou_thresholds': list(tiou_thresholds), 'ranks': list(ranks),
    }
    return result


def paired_bootstrap(pred_a: Mapping, pred_b: Mapping, records: Iterable[dict], rank=1, tiou=0.5, iterations=2000, seed=2027) -> dict:
    """Paired cluster bootstrap of B minus A, resampling ground-truth videos.

For multiply-positive queries, the lexically first positive video is the fixed cluster.
Each bootstrap statistic is query-weighted after resampling clusters.
"""
    records = list(records)
    truth = _ground_truth(records)
    key = f'R{rank}@{tiou:g}'
    hit_a = per_query_hits(pred_a, records, (rank,), (tiou,))
    hit_b = per_query_hits(pred_b, records, (rank,), (tiou,))
    clusters = defaultdict(list)
    for qid, target in truth.items():
        video = min(p[0] for p in target['positives'])
        clusters[video].append(float(hit_b[qid][key]) - float(hit_a[qid][key]))
    if not clusters or iterations < 1:
        raise ValueError('Bootstrap requires nonempty data and positive iterations')
    values = list(clusters.values())
    sums = np.array([sum(v) for v in values], dtype=np.float64)
    counts = np.array([len(v) for v in values], dtype=np.int64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations)
    for i in range(iterations):
        indices = rng.integers(0, len(values), size=len(values))
        bootstrap[i] = 100.0 * sums[indices].sum() / counts[indices].sum()
    return {'metric': key, 'delta_b_minus_a_pp': float(100 * sums.sum() / counts.sum()),
            'ci95_pp': np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            'probability_delta_nonpositive': float(np.mean(bootstrap <= 0)),
            'n_queries': int(counts.sum()), 'n_video_clusters': len(values),
            'iterations': iterations, 'seed': seed,
            'procedure': 'paired percentile bootstrap, video-cluster sampling, query-weighted recall'}


def save_predictions_jsonl(path: str | Path, predictions: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_predictions(predictions)
    with path.open('w') as handle:
        for qid in sorted(canonical):
            handle.write(json.dumps({'desc_id': qid, 'predictions': canonical[qid]}, allow_nan=False) + '\n')


def load_predictions_jsonl(path: str | Path) -> dict[str, list[list]]:
    predictions = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row['desc_id'])
            if qid in predictions:
                raise ValueError(f'Duplicate prediction row for desc_id {qid}')
            predictions[qid] = row['predictions']
    return _canonical_predictions(predictions)
