#!/usr/bin/env python3
"""Append attempt-3 local-neighborhood features; no labels or tuning are read."""
from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import json
import os
import time

import h5py
import numpy as np

from vcmr.conditional import local_null_signals
from vcmr.neighborhood import candidate_shifts, neighborhood_null_features
from vcmr.ranking import read_prediction_rows


COLUMNS = [
    "base_log", "whole_query", "conjunction", "surplus",
    "whole_local_contrast", "unconditional_local_excess",
    "conditional_local_excess", "shared_slope_control",
    "neighborhood_unconditional_control", "neighborhood_conditional_excess",
    "neighborhood_shared_slope_control",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.h5")
    started = time.time()
    stats = {
        "queries": 0, "eligible_queries": 0, "eligible_video_pairs": 0,
        "eligible_moments": 0, "columns": COLUMNS, "surrogates": 16,
        "tau": 0.5, "std_floor": 0.01, "clip": 3, "seed": 2027,
        "seed_key": "2027:query_id:video_id:start_frame:end_frame",
        "neighborhood": "[start-duration,end+duration] clipped to observed support",
        "conditioning_fit": "unchanged video-level fit from conditional.local_null_signals",
        "timestamp_coverage_min": 0.8, "seconds_per_frame": 1.5,
        "reads_labels": False,
    }
    with h5py.File(root / "data/tvr/clip_video.h5", "r") as vf, \
            h5py.File(root / f"data/tvr/phrases_{args.split}.h5", "r") as qf, \
            h5py.File(args.input, "r") as old, \
            h5py.File(temporary, "w") as out:
        old_columns = json.loads(old.attrs["columns"])
        if old_columns != COLUMNS[:8]:
            raise ValueError(f"Input must contain the registered eight attempt-2 columns: {old_columns}")
        out.attrs["columns"] = json.dumps(COLUMNS)
        out.attrs["feature_version"] = "neighborhood_v3"
        out.attrs["baseline_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
        out.attrs["input_features"] = str(args.input)
        out.attrs["complete"] = False

        @lru_cache(maxsize=5000)
        def video(name):
            x = vf[name][()].astype(np.float32)
            return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)

        for qid, predictions in read_prediction_rows(args.baseline):
            ds = old[qid]
            indices = json.loads(ds.attrs["phrase_indices"])
            gate = bool(ds.attrs["eligible"])
            if ds.shape != (len(predictions), 8):
                raise ValueError(f"Candidate/input-feature shape mismatch for query {qid}: {ds.shape}")
            features = np.zeros((len(predictions), len(COLUMNS)), dtype=np.float32)
            features[:, :8] = ds[()]
            if gate:
                embeddings = qf[qid][()]
                groups = {}
                for index, prediction in enumerate(predictions):
                    groups.setdefault(prediction[0], []).append(index)
                for name, locations in groups.items():
                    if name not in vf:
                        continue
                    x = video(name)
                    scores = embeddings @ x.T
                    starts = np.floor(np.asarray([predictions[i][1] for i in locations]) / 1.5 + 1e-6).astype(int)
                    ends = np.ceil(np.asarray([predictions[i][2] for i in locations]) / 1.5 - 1e-6).astype(int)
                    coverage = (np.minimum(ends, len(x)) - starts) / np.maximum(ends - starts, 1)
                    valid = (starts >= 0) & (starts < len(x)) & (ends > starts) & (coverage >= 0.8)
                    if not valid.any() or not np.all(scores[indices].std(axis=1) > 0.001):
                        continue
                    use = np.asarray(locations)[valid]
                    aa = starts[valid]
                    bb = np.minimum(ends[valid], len(x))
                    # Reuse the existing normalization/regression implementation
                    # exactly; the zero-shift call is only a fit extractor.
                    signals = local_null_signals(scores[indices], scores[0],
                                                 np.zeros((1, len(indices)), dtype=int), tau=0.5)
                    z = signals["fitted"] + signals["residual"]
                    shifts = candidate_shifts(aa, bb, len(x), len(indices),
                                              f"2027:{qid}:{name}", n_surrogates=16)
                    local = neighborhood_null_features(z, signals["fitted"], aa, bb, shifts, tau=0.5)
                    features[use, 8] = np.clip(local["unconditional"], -3, 3)
                    features[use, 9] = np.clip(local["conditional"], -3, 3)
                    features[use, 10] = np.clip(local["shared_slope"], -3, 3)
                    stats["eligible_video_pairs"] += 1
                    stats["eligible_moments"] += len(use)
            if not np.isfinite(features).all():
                raise ValueError(f"Nonfinite features for query {qid}")
            dataset = out.create_dataset(qid, data=features, compression="lzf")
            dataset.attrs["eligible"] = gate
            dataset.attrs["phrase_indices"] = json.dumps(indices)
            stats["queries"] += 1
            stats["eligible_queries"] += int(gate)
            if stats["queries"] % 500 == 0:
                print(json.dumps({"queries": stats["queries"], "seconds": time.time() - started}), flush=True)
        if stats["queries"] != len(old):
            raise ValueError("Input query feature coverage differs from baseline prediction coverage")
        stats["seconds"] = time.time() - started
        out.attrs["stats"] = json.dumps(stats)
        out.attrs["complete"] = True
        out.flush()
    os.replace(temporary, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
