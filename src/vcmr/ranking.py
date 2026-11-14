"""Shared global ranking and temporal suppression for all experimental variants."""
from __future__ import annotations
import numpy as np


def suppress(predictions, scores, threshold=0.7, topk=100):
    """Greedy same-video temporal NMS; same > threshold convention as DeMUL."""
    if not len(predictions):
        return []
    rows = np.asarray([[p[1], p[2]] for p in predictions], dtype=float)
    videos = np.asarray([p[0] for p in predictions])
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept = []
    while len(order) and len(kept) < topk:
        i = int(order[0])
        kept.append([str(videos[i]), float(rows[i,0]), float(rows[i,1]), float(scores[i])])
        order = order[1:]
        if not len(order):
            break
        intersection = np.maximum(0, np.minimum(rows[i,1], rows[order,1]) - np.maximum(rows[i,0], rows[order,0]))
        # For overlapping intervals, outer span equals interval union.
        union = np.maximum(rows[i,1], rows[order,1]) - np.minimum(rows[i,0], rows[order,0])
        overlaps = intersection / np.maximum(union, 1e-12)
        order = order[~((videos[order] == videos[i]) & (overlaps > threshold))]
    return kept


def read_prediction_rows(path):
    import json
    from pathlib import Path
    mapping_path = Path(path).parent / "video2idx.json"
    inverse = None
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            predictions = row["predictions"]
            if predictions and isinstance(predictions[0][0], (int, float)):
                if inverse is None:
                    if not mapping_path.exists():
                        raise FileNotFoundError(f"Indexed predictions require {mapping_path}")
                    inverse = {int(v): k for k, v in json.loads(mapping_path.read_text()).items()}
                predictions = [[inverse[int(p[0])], *p[1:]] for p in predictions]
            yield str(row["desc_id"]), predictions
