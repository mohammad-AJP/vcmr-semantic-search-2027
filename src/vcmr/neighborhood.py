"""Candidate-neighborhood residual surrogates for the third registered attempt.

This is a score feature, not a calibrated conditional-randomization test. The
conditioning fit is supplied explicitly. Locality guarantees below apply to
fixed standardized traces/fits, not to refitting them after changing a video.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.special import logsumexp

from .association import interval_means, soft_conjunction


def neighborhood_bounds(starts, ends, time_length):
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if starts.ndim != 1 or ends.shape != starts.shape:
        raise ValueError("starts and ends must be equally sized one-dimensional arrays")
    if time_length < 1 or np.any(starts < 0) or np.any(ends > time_length) or np.any(ends <= starts):
        raise ValueError("Candidates must be nonempty and within observed video support")
    lengths = ends - starts
    return np.maximum(0, starts - lengths), np.minimum(time_length, ends + lengths)


def candidate_shifts(starts, ends, time_length, predicates, seed_prefix,
                     n_surrogates=16):
    """Deterministic candidate-local shifts, invariant to candidate enumeration.

    Use seed_prefix='2027:<query_id>:<video_id>'. Start/end frame coordinates
    complete the seed. Shifts independently sample every predicate and surrogate.
    """
    if predicates < 2 or n_surrogates < 1:
        raise ValueError("Need at least two predicates and one surrogate")
    left, right = neighborhood_bounds(starts, ends, time_length)
    result = np.empty((len(left), n_surrogates, predicates), dtype=np.int64)
    for index, (start, end, lo, hi) in enumerate(zip(starts, ends, left, right)):
        key = f"{seed_prefix}:{int(start)}:{int(end)}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        result[index] = np.random.default_rng(seed).integers(
            0, int(hi - lo), size=(n_surrogates, predicates))
    return result


def neighborhood_null_features(z, fitted, starts, ends, shifts, tau=0.5,
                               candidate_block_size=64):
    """Evaluate fixed candidates under local unconditional/conditional nulls.

    Args:
        z, fitted: [predicates,time] standardized observations and fitted traces.
        starts, ends: half-open candidate frame coordinates.
        shifts: [candidates,surrogates,predicates], or [surrogates,predicates]
            broadcast across candidates. Positive shifts follow np.roll.

    Shared-slope control:
        J(I) - mean_I(mean_predicates(fitted))
             - mean_surrogates(mean_B(softmin(shifted residual))).

    Every surrogate samples only B(I)=[start-L,end+L], clipped to observed
    support. The fitted component stays at its original local times. Summation
    masks exclude padded frames and keep clipped edge neighborhoods honest.
    """
    z = np.asarray(z, dtype=np.float64)
    fitted = np.asarray(fitted, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1 or fitted.shape != z.shape:
        raise ValueError("z and fitted must have the same [predicates>=2,time] shape")
    if not np.isfinite(z).all() or not np.isfinite(fitted).all():
        raise ValueError("Nonfinite standardized trace or fitted component")
    if tau <= 0 or candidate_block_size < 1:
        raise ValueError("tau and candidate_block_size must be positive")
    left, right = neighborhood_bounds(starts, ends, z.shape[1])
    shifts = np.asarray(shifts)
    if not np.issubdtype(shifts.dtype, np.integer):
        raise ValueError("Shifts must be integers")
    if shifts.ndim == 2:
        shifts = np.broadcast_to(shifts, (len(starts),) + shifts.shape)
    if shifts.ndim != 3 or shifts.shape[0] != len(starts) or shifts.shape[2] != z.shape[0] or shifts.shape[1] < 1:
        raise ValueError("Shifts must match [candidates,surrogates,predicates]")
    residual = z - fitted
    observed = interval_means(soft_conjunction(z, tau), starts, ends)
    fitted_means = interval_means(fitted.mean(axis=0), starts, ends)
    unconditional = np.empty(len(starts), dtype=np.float64)
    conditional = np.empty(len(starts), dtype=np.float64)
    shared_slope = np.empty(len(starts), dtype=np.float64)
    n_predicates = z.shape[0]
    log_predicates = np.log(n_predicates)
    predicate_ids = np.arange(n_predicates)[None, None, :, None]

    def conjunction(values):
        return -tau * (logsumexp(-values / tau, axis=2) - log_predicates)

    for begin in range(0, len(starts), candidate_block_size):
        stop = min(begin + candidate_block_size, len(starts))
        lo, hi = left[begin:stop], right[begin:stop]
        lengths = hi - lo
        positions = np.arange(int(lengths.max()))
        # A padded position is gathered from the last real frame; masks ensure
        # its value never enters either the candidate or neighborhood average.
        original_ids = lo[:, None] + np.minimum(positions[None, :], lengths[:, None] - 1)
        original_fitted = fitted[:, original_ids].transpose(1, 0, 2)
        rolled_positions = (positions[None, None, None, :]
                            - shifts[begin:stop, :, :, None]) % lengths[:, None, None, None]
        donor_ids = lo[:, None, None, None] + rolled_positions
        shifted_z = z[predicate_ids, donor_ids]
        unconditional_signal = conjunction(shifted_z)
        del shifted_z
        shifted_residual = residual[predicate_ids, donor_ids]
        conditional_signal = conjunction(original_fitted[:, None, :, :] + shifted_residual)
        residual_signal = conjunction(shifted_residual)
        del shifted_residual
        candidate_mask = ((positions[None, :] >= (starts[begin:stop] - lo)[:, None])
                          & (positions[None, :] < (ends[begin:stop] - lo)[:, None]))
        neighborhood_mask = positions[None, :] < lengths[:, None]
        candidate_lengths = ends[begin:stop] - starts[begin:stop]
        unconditional_reference = (
            (unconditional_signal * candidate_mask[:, None, :]).sum(axis=2)
            / candidate_lengths[:, None]).mean(axis=1)
        conditional_reference = (
            (conditional_signal * candidate_mask[:, None, :]).sum(axis=2)
            / candidate_lengths[:, None]).mean(axis=1)
        residual_reference = (
            (residual_signal * neighborhood_mask[:, None, :]).sum(axis=2)
            / lengths[:, None]).mean(axis=1)
        unconditional[begin:stop] = observed[begin:stop] - unconditional_reference
        conditional[begin:stop] = observed[begin:stop] - conditional_reference
        shared_slope[begin:stop] = observed[begin:stop] - fitted_means[begin:stop] - residual_reference
    return {
        "unconditional": unconditional,
        "conditional": conditional,
        "shared_slope": shared_slope,
        "observed": observed,
        "neighborhood_starts": left,
        "neighborhood_ends": right,
    }
