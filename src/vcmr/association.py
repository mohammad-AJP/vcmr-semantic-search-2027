"""Original experimental temporal-association reference; not a semantic verifier.

The score compares a same-time soft conjunction with independently phase-shifted
traces, repeating the maximum over every window of the candidate's length.
It is a heuristic under a circular-stationarity assumption, not a p-value.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp


def soft_conjunction(z, tau=0.5):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1 or tau <= 0:
        raise ValueError("Expected nonempty [predicates,time] and tau>0")
    return -tau * (logsumexp(-z / tau, axis=0) - np.log(z.shape[0]))


def interval_means(signal, starts, ends):
    signal = np.asarray(signal, dtype=np.float64)
    starts, ends = np.asarray(starts, dtype=int), np.asarray(ends, dtype=int)
    if np.any(starts < 0) or np.any(ends > signal.shape[-1]) or np.any(ends <= starts):
        raise ValueError("Intervals must be nonempty and within observed support")
    prefix = np.concatenate([np.zeros(signal.shape[:-1] + (1,)), np.cumsum(signal, axis=-1)], axis=-1)
    return (prefix[..., ends] - prefix[..., starts]) / (ends - starts)


def reference_maxima(z, lengths, shifts, tau=0.5):
    z = np.asarray(z, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=int)
    lengths = np.asarray(lengths, dtype=int)
    if z.ndim != 2 or shifts.ndim != 2 or shifts.shape[1] != z.shape[0] or not len(shifts):
        raise ValueError("shifts must have shape [surrogates,predicates]")
    if np.any(lengths < 1) or np.any(lengths > z.shape[1]):
        raise ValueError("Invalid reference window lengths")
    null = np.stack([soft_conjunction(np.stack([np.roll(trace, int(shift)) for trace, shift in zip(z, row)]), tau)
                     for row in shifts])
    prefix = np.pad(np.cumsum(null, axis=1), ((0, 0), (1, 0)))
    # Linear windows do not wrap; circularity is used only to construct surrogates.
    return np.asarray([np.mean(np.max((prefix[:, length:] - prefix[:, :-length]) / length, axis=1))
                       for length in lengths], dtype=np.float64)


def association_features(similarities, starts, ends, shifts, tau=0.5):
    similarities = np.asarray(similarities, dtype=np.float64)
    if similarities.ndim != 2 or similarities.shape[0] < 2:
        raise ValueError("Association requires at least two distinct predicates")
    scale = np.maximum(similarities.std(axis=1, keepdims=True), 0.01)
    z = (similarities - similarities.mean(axis=1, keepdims=True)) / scale
    conjunction = interval_means(soft_conjunction(z, tau), starts, ends)
    lengths = np.asarray(ends) - np.asarray(starts)
    unique, inverse = np.unique(lengths, return_inverse=True)
    ref = reference_maxima(z, unique, shifts, tau)[inverse]
    return {"conjunction": conjunction, "surplus": conjunction - ref,
            "reference": ref, "predicate_scale": scale[:, 0]}


def select_phrase_indices(parse, embeddings, cosine_threshold=0.95):
    """Remove holistic text, lexical containment, and near-duplicate CLIP phrases."""
    phrases = parse["retrieval_phrases"]
    priority = {"action": 0, "attributed_entity": 1, "relation": 2, "entity": 3}
    candidates = sorted(range(1, len(phrases)), key=lambda i: (priority.get(phrases[i]["kind"], 4), i))
    selected = []
    for i in candidates:
        text = phrases[i]["text"].strip().lower()
        redundant = False
        for j in selected:
            prior = phrases[j]["text"].strip().lower()
            if text in prior or prior in text or float(embeddings[i] @ embeddings[j]) >= cosine_threshold:
                redundant = True
                break
        if not redundant:
            selected.append(i)
    return selected


def eligible(parse, phrase_indices):
    return (len(phrase_indices) >= 2 and parse.get("temporal_mode") == "simultaneous"
            and parse.get("polarity") == "positive")
