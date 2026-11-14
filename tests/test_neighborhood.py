"""Analytical checks for local reference arithmetic and its actual guarantees."""
from itertools import product
import numpy as np
import pytest

from vcmr.association import soft_conjunction
from vcmr.conditional import local_null_signals
from vcmr.neighborhood import candidate_shifts, neighborhood_bounds, neighborhood_null_features


def test_vectorized_matches_exhaustive_brute_force_with_clipped_neighborhoods():
    z = np.array([[0.0, 1.5, -0.4, 0.8, 2.0], [0.7, -0.2, 1.1, -0.9, 0.3]])
    fitted = np.array([[0.1, 0.4, -0.1, 0.2, 0.5], [0.2, 0.2, -0.3, 0.4, 0.1]])
    starts, ends = np.array([0, 1, 3]), np.array([2, 3, 5])
    # Enumerate all 20x20 donor shifts; uniform modulo 4 and modulo 5 both
    # cover every shift equally in the clipped neighborhoods of lengths 4/5.
    shifts = np.array(list(product(range(20), repeat=2)))
    actual = neighborhood_null_features(z, fitted, starts, ends, shifts, candidate_block_size=2)
    left, right = neighborhood_bounds(starts, ends, z.shape[1])
    for i, (a, b, lo, hi) in enumerate(zip(starts, ends, left, right)):
        residual = z[:, lo:hi] - fitted[:, lo:hi]
        observed = soft_conjunction(z[:, a:b]).mean()
        unconditional, conditional, shared = [], [], []
        for row in shifts:
            shifted_z = np.stack([np.roll(trace, int(k)) for trace, k in zip(z[:, lo:hi], row)])
            shifted_r = np.stack([np.roll(trace, int(k)) for trace, k in zip(residual, row)])
            unconditional.append(soft_conjunction(shifted_z)[a-lo:b-lo].mean())
            conditional.append(soft_conjunction(fitted[:, lo:hi] + shifted_r)[a-lo:b-lo].mean())
            shared.append(soft_conjunction(shifted_r).mean())
        np.testing.assert_allclose(actual["unconditional"][i], observed - np.mean(unconditional), atol=1e-12)
        np.testing.assert_allclose(actual["conditional"][i], observed - np.mean(conditional), atol=1e-12)
        np.testing.assert_allclose(actual["shared_slope"][i], observed - fitted[:, a:b].mean() - np.mean(shared), atol=1e-12)


def test_zero_shifts_have_zero_conditional_and_unconditional_excess():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(3, 30))
    fitted = rng.normal(size=(3, 30))
    result = neighborhood_null_features(z, fitted, [0, 8, 26], [3, 13, 30], np.zeros((2, 3), dtype=int))
    np.testing.assert_allclose(result["conditional"], 0, atol=1e-12)
    np.testing.assert_allclose(result["unconditional"], 0, atol=1e-12)


def test_persistent_common_mode_has_zero_conditional_excess():
    h = np.array([0., 1., 3., 2., 0., -1., -2., 1., 3., 0.])
    s = np.stack([2*h+4, 3*h-2])
    signals = local_null_signals(s, h, np.zeros((1, 2), dtype=int))
    shifts = candidate_shifts([0, 3, 7], [2, 6, 10], len(h), 2, "2027:test:video")
    result = neighborhood_null_features(signals["fitted"] + signals["residual"], signals["fitted"], [0, 3, 7], [2, 6, 10], shifts)
    np.testing.assert_allclose(result["conditional"], 0, atol=1e-10)
    assert np.max(np.abs(result["unconditional"])) > 0.1


def test_context_inside_neighborhood_changes_reference_but_outside_does_not():
    z = np.zeros((2, 10))
    fitted = np.zeros_like(z)
    shifts = np.array(list(product(range(6), repeat=2)))
    original = neighborhood_null_features(z, fitted, [2], [4], shifts)
    changed = z.copy()
    changed[0, 0], changed[1, 1] = 4, 4  # inside B=[0,6), outside I=[2,4)
    local = neighborhood_null_features(changed, fitted, [2], [4], shifts)
    np.testing.assert_allclose(local["observed"], original["observed"])
    assert abs(local["conditional"][0] - original["conditional"][0]) > 0.01
    changed[:, 8:] = np.array([[100], [-100]])  # strictly outside B
    outside = neighborhood_null_features(changed, fitted, [2], [4], shifts)
    for key in ["unconditional", "conditional", "shared_slope", "observed"]:
        np.testing.assert_allclose(outside[key], local[key], atol=1e-12)


def test_candidate_seeds_are_order_independent_and_duplicate_stable():
    starts = np.array([2, 10, 2])
    ends = np.array([4, 15, 4])
    original = candidate_shifts(starts, ends, 20, 3, "2027:qid:video")
    order = np.array([1, 2, 0])
    reordered = candidate_shifts(starts[order], ends[order], 20, 3, "2027:qid:video")
    np.testing.assert_array_equal(reordered, original[order])
    np.testing.assert_array_equal(original[0], original[2])


def test_equal_slope_control_matches_complete_shift_expectation():
    # A complete set of shifts makes the expected residual reference constant
    # over positions inside B, exposing the exact shared-slope equivalence.
    h = np.array([-1., 0., 2., 1., -2., 0.5])
    residual = np.array([[0., 2., -1., 0.2, -0.3, 1.], [0.4, -0.2, 1.3, -1., 0., 0.7]])
    fitted = np.stack([0.4*h, 0.4*h])
    shifts = np.array(list(product(range(6), repeat=2)))
    result = neighborhood_null_features(fitted + residual, fitted, [2], [4], shifts)
    np.testing.assert_allclose(result["conditional"], result["shared_slope"], atol=1e-12)


def test_rejects_invalid_support_and_nonintegral_shifts():
    z = np.zeros((2, 5))
    with pytest.raises(ValueError):
        neighborhood_null_features(z, z, [-1], [2], np.zeros((1, 2), dtype=int))
    with pytest.raises(ValueError):
        neighborhood_null_features(z, z, [1], [6], np.zeros((1, 2), dtype=int))
    with pytest.raises(ValueError):
        neighborhood_null_features(z, z, [1], [3], np.zeros((1, 2), dtype=float))
