"""Deterministic mathematical checks for the association scorer.

These verify the implemented statistic, not novelty or actor-identity grounding.
In particular, duplicated/common-cause traces intentionally receive the same score
as true co-occurrence: global temporal evidence cannot identify the cause.
"""
import itertools

import numpy as np
import pytest

from vcmr.association import association_features, reference_maxima, soft_conjunction


TAU = 0.5
ALL_SHIFTS = np.asarray(list(itertools.product(range(4), repeat=2)), dtype=np.int64)
ALIGNED = np.asarray([[1, 1, -1, -1], [1, 1, -1, -1]], dtype=np.float64)
OPPOSED = np.asarray([[1, 1, -1, -1], [-1, -1, 1, 1]], dtype=np.float64)


def brute_conjunction(z, tau=TAU):
    """Independent scalar reference; tiny inputs avoid exponential overflow."""
    z = np.asarray(z, dtype=np.float64)
    return np.asarray([-tau * np.log(np.mean(np.exp(-z[:, t] / tau)))
                       for t in range(z.shape[1])])


def brute_reference_maxima(z, lengths, shifts, tau=TAU):
    values = []
    for length in lengths:
        length = int(length)
        maxima = []
        for offset in shifts:
            shifted = np.stack([np.roll(z[p], int(offset[p])) for p in range(z.shape[0])])
            conjunction = brute_conjunction(shifted, tau)
            # Candidate windows are linear and do not wrap around video endpoints.
            windows = [conjunction[start:start + length].mean()
                       for start in range(z.shape[1] - length + 1)]
            maxima.append(max(windows))
        values.append(np.mean(maxima))
    return np.asarray(values)


def test_soft_conjunction_preserves_equal_predicates_and_closed_form():
    # Dividing the exponential sum by P matters: equal predicates return that value.
    np.testing.assert_allclose(soft_conjunction(ALIGNED, tau=TAU), ALIGNED[0], atol=1e-7)
    expected_opposed = -TAU * np.log(np.cosh(1 / TAU))
    np.testing.assert_allclose(soft_conjunction(OPPOSED, tau=TAU), expected_opposed, atol=1e-7)
    z = np.asarray([[0.2, -0.7, 1.2], [-0.3, 0.8, 0.4]])
    closed = z.mean(axis=0) - TAU * np.log(np.cosh((z[0] - z[1]) / (2 * TAU)))
    np.testing.assert_allclose(soft_conjunction(z, tau=TAU), closed, atol=1e-7)


def test_reference_maxima_matches_independent_nonwrapping_bruteforce():
    lengths = np.asarray([1, 2, 3, 4], dtype=np.int64)
    expected = brute_reference_maxima(ALIGNED, lengths, ALL_SHIFTS)
    actual = reference_maxima(ALIGNED, lengths, ALL_SHIFTS, tau=TAU)
    assert np.asarray(actual).shape == (4,)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    # Exhaustive independent shifts erase observed relative alignment.
    np.testing.assert_allclose(reference_maxima(OPPOSED, lengths, ALL_SHIFTS, tau=TAU), expected, atol=1e-7)
    assert expected[1] == pytest.approx(0.1062493131605339, abs=1e-7)


def test_nonwrapping_maximum_differs_from_circular_window_maximum():
    # The strongest conjunction is split between the first and last frames.
    z = np.asarray([[1, -1, -1, 1], [1, -1, -1, 1]], dtype=float)
    shifts = np.zeros((1, 2), dtype=np.int64)
    actual = reference_maxima(z, np.asarray([2]), shifts, tau=TAU)
    # Valid windows average to 0, -1, 0. A wraparound window would incorrectly score 1.
    np.testing.assert_allclose(actual, [0.0], atol=1e-7)


def test_aligned_and_opposed_interval_surplus_separate_exactly():
    starts = np.asarray([0, 2], dtype=np.int64)
    ends = np.asarray([2, 4], dtype=np.int64)
    aligned = association_features(ALIGNED, starts, ends, ALL_SHIFTS, tau=TAU)
    opposed = association_features(OPPOSED, starts, ends, ALL_SHIFTS, tau=TAU)
    reference = brute_reference_maxima(ALIGNED, [2], ALL_SHIFTS)[0]
    expected_disjoint = -TAU * np.log(np.cosh(1 / TAU))
    np.testing.assert_allclose(aligned['conjunction'], [1.0, -1.0], atol=1e-7)
    np.testing.assert_allclose(aligned['surplus'], np.asarray([1.0, -1.0]) - reference, atol=1e-7)
    np.testing.assert_allclose(opposed['conjunction'], [expected_disjoint, expected_disjoint], atol=1e-7)
    np.testing.assert_allclose(opposed['surplus'], expected_disjoint - reference, atol=1e-7)
    assert aligned['surplus'][0] > 0 > opposed['surplus'][0]


def test_predicate_permutation_with_coordinated_shift_columns_is_invariant():
    z = np.asarray([[0.2, -0.3, 1.1, -0.9], [-0.8, 0.7, 0.1, 0.4], [0.3, -1.2, 0.8, 0.6]])
    shifts = np.asarray([[0, 1, 2], [3, 0, 1], [2, 3, 0], [1, 2, 3]], dtype=np.int64)
    permutation = np.asarray([2, 0, 1])
    lengths = np.asarray([1, 2, 4])
    np.testing.assert_allclose(soft_conjunction(z, tau=TAU), soft_conjunction(z[permutation], tau=TAU), atol=1e-7)
    np.testing.assert_allclose(reference_maxima(z, lengths, shifts, tau=TAU),
                               reference_maxima(z[permutation], lengths, shifts[:, permutation], tau=TAU), atol=1e-7)
    starts, ends = np.asarray([0, 1]), np.asarray([2, 4])
    before = association_features(z, starts, ends, shifts, tau=TAU)
    after = association_features(z[permutation], starts, ends, shifts[:, permutation], tau=TAU)
    for key in ('conjunction', 'surplus'):
        np.testing.assert_allclose(before[key], after[key], atol=1e-7)


def test_population_standardization_and_positive_affine_invariance_above_floor():
    similarities = np.asarray([[0.1, 0.7, -0.2, 0.4], [-0.3, 0.2, 0.6, 0.1]])
    starts, ends = np.asarray([0, 1, 0]), np.asarray([2, 4, 4])
    std = similarities.std(axis=1, keepdims=True, ddof=0)
    assert np.all(std > 0.01)
    z = (similarities - similarities.mean(axis=1, keepdims=True)) / std
    c = brute_conjunction(z)
    expected = np.asarray([c[start:end].mean() for start, end in zip(starts, ends)])
    before = association_features(similarities, starts, ends, ALL_SHIFTS, tau=TAU)
    np.testing.assert_allclose(before['conjunction'], expected, atol=1e-6)
    transformed = similarities * np.asarray([[2.0], [0.4]]) + np.asarray([[3.0], [-0.8]])
    assert np.all(transformed.std(axis=1) > 0.01)
    after = association_features(transformed, starts, ends, ALL_SHIFTS, tau=TAU)
    # This invariance is a limitation: absolute visual evidence strength is removed.
    # It is not expected when either standard deviation is clipped by the 0.01 floor.
    for key in ('conjunction', 'surplus'):
        np.testing.assert_allclose(before[key], after[key], atol=1e-6)


def test_duplicate_or_common_cause_traces_are_not_actor_truth_evidence():
    starts, ends = np.asarray([0]), np.asarray([2])
    true_cooccurrence = ALIGNED.copy()
    # Identical observations can arise from repeating the same phrase, person visibility,
    # or a shared camera change. No function of these traces can distinguish the causes.
    shared_nuisance = np.stack([ALIGNED[0], ALIGNED[0]])
    a = association_features(true_cooccurrence, starts, ends, ALL_SHIFTS, tau=TAU)
    b = association_features(shared_nuisance, starts, ends, ALL_SHIFTS, tau=TAU)
    np.testing.assert_allclose(a['surplus'], b['surplus'], atol=1e-7)
    assert b['surplus'][0] > 0  # Explicit counterexample to interpreting surplus as binding truth.
