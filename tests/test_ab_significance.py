"""Tests for ab_significance.

Reference values were computed once with SciPy 1.x (scipy.stats) and statsmodels
(proportion_confint method wilson, confint_proportions_2indep method newcomb) and are
hard-coded here. Neither package is a dependency of this library.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random

import pytest

from ab_significance import (
    MAX_ITERATIONS,
    MAX_SAMPLE_SIZE,
    AbSignificanceError,
    InvalidInput,
    SignificanceResult,
    min_sample_size,
    permutation_test,
    two_proportion_test,
    welch_t_test,
    wilson_interval,
)
from ab_significance._dist import (
    norm_cdf,
    norm_ppf,
    regularized_beta,
    t_crit,
    t_two_sided_p,
)

_Rng = random.Random

WELCH_A = [0.71, 0.64, 0.80, 0.75, 0.69, 0.77, 0.73, 0.70]
WELCH_B = [0.62, 0.66, 0.59, 0.70, 0.61, 0.64, 0.68]
PERM_A = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6, 0.35]
PERM_B = [x + 0.02 for x in PERM_A]


# --- acceptance items -------------------------------------------------------------------


def test_two_proportion_reference_values():
    result = two_proportion_test(45, 100, 60, 100)
    assert result.statistic == pytest.approx(-2.1239769762, abs=1e-6)
    assert result.p_value == pytest.approx(0.0336720689, abs=1e-6)
    assert result.effect == pytest.approx(-0.15, abs=1e-6)
    assert result.ci_low == pytest.approx(-0.2804485210, abs=1e-6)
    assert result.ci_high == pytest.approx(-0.0117239353, abs=1e-6)
    assert result.df is None
    assert result.method == "two_proportion_z"


def test_wilson_reference_values():
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775327999, abs=1e-6)
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7224672001, abs=1e-6)
    assert high == 1.0
    low, high = wilson_interval(7, 20)
    assert low == pytest.approx(0.1811918241, abs=1e-6)
    assert high == pytest.approx(0.5671457233, abs=1e-6)


def test_welch_reference_values():
    result = welch_t_test(WELCH_A, WELCH_B)
    assert result.statistic == pytest.approx(3.4923733505, abs=1e-6)
    assert result.df == pytest.approx(12.8858651776, abs=1e-6)
    assert result.p_value == pytest.approx(0.0040205765, abs=1e-6)
    assert result.ci_low == pytest.approx(0.0308077561, abs=1e-6)
    assert result.ci_high == pytest.approx(0.1309779582, abs=1e-6)
    assert result.effect == pytest.approx(sum(WELCH_A) / 8 - sum(WELCH_B) / 7, abs=1e-12)
    assert result.method == "welch_t"


def test_permutation_is_reproducible_with_nan_interval():
    first = permutation_test(PERM_A, PERM_B, iterations=500, seed=3)
    second = permutation_test(PERM_A, PERM_B, iterations=500, seed=3)
    assert first.statistic == second.statistic
    assert first.p_value == second.p_value
    assert first.effect == second.effect
    assert math.isnan(first.ci_low)
    assert math.isnan(first.ci_high)
    assert first.df is None
    assert first.method == "permutation"


def test_permutation_p_value_is_count_based():
    result = permutation_test(PERM_A, PERM_B, iterations=500)
    scaled = result.p_value * 501
    count_plus_one = round(scaled)
    assert abs(scaled - count_plus_one) < 1e-9
    assert result.p_value == count_plus_one / 501
    assert 1 <= count_plus_one <= 501


def test_permutation_paired_versus_unpaired():
    a = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6, 0.35]
    b = [x + 0.02 for x in a]
    paired = permutation_test(a, b, paired=True, seed=0, iterations=10_000)
    unpaired = permutation_test(a, b, paired=False, seed=0, iterations=10_000)
    assert paired.p_value < 0.01
    assert unpaired.p_value > 0.5
    assert paired.method == "paired_permutation"
    assert paired.effect == pytest.approx(-0.02, abs=1e-12)


def test_min_sample_size_reference():
    assert min_sample_size(0.45, 0.60) == 173
    assert min_sample_size(0.45, 0.60, alpha=0.05, power=0.8) == 173
    with pytest.raises(InvalidInput) as info:
        min_sample_size(0.5, 0.5)
    assert info.value.code == "invalid_input"


_ACCEPTANCE_BAD_CALLS = [
    lambda: wilson_interval(1, 0),
    lambda: wilson_interval(11, 10),
    lambda: wilson_interval(True, 10),
    lambda: wilson_interval(7.0, 20),
    lambda: welch_t_test([1.0], [1.0, 2.0]),
    lambda: welch_t_test([float("nan"), 1.0], [1.0, 2.0]),
    lambda: welch_t_test([float("inf"), 1.0], [1.0, 2.0]),
    lambda: wilson_interval(5, 10, confidence=1.0),
    lambda: permutation_test([1.0, 2.0], [3.0, 4.0], iterations=0),
    lambda: permutation_test([1.0, 2.0], [3.0, 4.0], iterations=1_000_001),
    lambda: welch_t_test([0.0] * 1_000_001, [1.0, 2.0]),
    lambda: permutation_test([0.0] * 1_000_001, [1.0]),
]


@pytest.mark.parametrize("call", _ACCEPTANCE_BAD_CALLS)
def test_acceptance_invalid_inputs(call):
    with pytest.raises(InvalidInput) as info:
        call()
    assert info.value.code == "invalid_input"
    assert len(str(info.value)) <= 200


# --- further invalid inputs --------------------------------------------------------------

_MORE_BAD_CALLS = [
    lambda: wilson_interval("7", 20),
    lambda: wilson_interval("x" * 1000, 10),
    lambda: wilson_interval([1], 10),
    lambda: wilson_interval(0, 10**16),
    lambda: wilson_interval(10**40, 10**41),
    lambda: wilson_interval(-1, 10),
    lambda: wilson_interval(5, True),
    lambda: wilson_interval(5, 10, confidence=True),
    lambda: wilson_interval(5, 10, confidence=0.0),
    lambda: wilson_interval(5, 10, confidence="0.95"),
    lambda: wilson_interval(5, 10, confidence=float("nan")),
    lambda: two_proportion_test(1, 0, 1, 10),
    lambda: two_proportion_test(1, 10, 11, 10),
    lambda: two_proportion_test(1, 10, 1, 10, confidence=1.0),
    lambda: welch_t_test("abc", [1.0, 2.0]),
    lambda: welch_t_test(iter([1.0, 2.0]), [1.0, 2.0]),
    lambda: welch_t_test([True, 1.0], [1.0, 2.0]),
    lambda: welch_t_test(["x", 1.0], [1.0, 2.0]),
    lambda: welch_t_test([1e151, 1.0], [1.0, 2.0]),
    lambda: welch_t_test([10**400, 1.0], [1.0, 2.0]),
    lambda: welch_t_test([1.0, 2.0], [1.0, 2.0], confidence=1.0),
    lambda: permutation_test([], [1.0]),
    lambda: permutation_test([1.0, 2.0], [1.0], paired=True),
    lambda: permutation_test([1.0], [1.0], paired="yes"),
    lambda: permutation_test([1.0], [1.0], iterations=True),
    lambda: permutation_test([1.0], [1.0], iterations=10.0),
    lambda: permutation_test([1.0], [1.0], seed=True),
    lambda: permutation_test([1.0], [1.0], seed=2**64 + 1),
    lambda: permutation_test([float("nan")], [1.0]),
    lambda: min_sample_size(0.0, 0.5),
    lambda: min_sample_size(1.0, 0.5),
    lambda: min_sample_size(1, 0.5),
    lambda: min_sample_size(True, 0.5),
    lambda: min_sample_size(0.45, 0.6, alpha=0.0),
    lambda: min_sample_size(0.45, 0.6, alpha=1.0),
    lambda: min_sample_size(0.45, 0.6, power=1.0),
    lambda: min_sample_size(0.45, 0.6, power=float("nan")),
    lambda: min_sample_size(0.45, 0.6, alpha=0.5, power=0.01),
    lambda: min_sample_size(1e-320, 2e-320),
]


@pytest.mark.parametrize("call", _MORE_BAD_CALLS)
def test_further_invalid_inputs(call):
    with pytest.raises(InvalidInput) as info:
        call()
    assert info.value.code == "invalid_input"
    assert isinstance(info.value, ValueError)
    assert len(str(info.value)) <= 200


def test_error_message_never_contains_whole_sample():
    sample = [0.5] * 50 + [float("nan")]
    with pytest.raises(InvalidInput) as info:
        welch_t_test(sample, [1.0, 2.0])
    assert "0.5, 0.5" not in str(info.value)
    assert "a[50]" in str(info.value)


def test_error_classes():
    assert issubclass(InvalidInput, AbSignificanceError)
    assert issubclass(AbSignificanceError, ValueError)
    assert InvalidInput("x").code == "invalid_input"
    assert AbSignificanceError("x").code == "ab_significance_error"
    assert len(str(AbSignificanceError("x" * 500))) == 200


def test_limits_are_exported():
    assert MAX_SAMPLE_SIZE == 1_000_000
    assert MAX_ITERATIONS == 1_000_000


def test_library_does_not_configure_logging():
    core_logger = logging.getLogger("ab_significance._core")
    assert any(isinstance(h, logging.NullHandler) for h in core_logger.handlers)
    assert logging.getLogger("ab_significance").handlers


# --- Wilson ------------------------------------------------------------------------------


def test_wilson_confidence_99_is_wider():
    low95, high95 = wilson_interval(7, 20, 0.95)
    low99, high99 = wilson_interval(7, 20, confidence=0.99)
    assert low99 < low95
    assert high99 > high95
    _, upper = wilson_interval(0, 10, confidence=0.99)
    assert upper == pytest.approx(0.39885, abs=1e-4)


@pytest.mark.parametrize(("k", "n"), [(1, 10), (3, 7), (9, 10), (1, 1), (0, 1), (499, 1000)])
def test_wilson_symmetry_and_bounds(k, n):
    low, high = wilson_interval(k, n)
    mirror_low, mirror_high = wilson_interval(n - k, n)
    assert 0.0 <= low <= k / n <= high <= 1.0
    assert low == pytest.approx(1.0 - mirror_high, abs=1e-12)
    assert high == pytest.approx(1.0 - mirror_low, abs=1e-12)


def test_wilson_large_counts():
    low, high = wilson_interval(5 * 10**14, 10**15)
    assert low < 0.5 < high
    assert high - low < 1e-6


# --- two proportions ---------------------------------------------------------------------


def test_two_proportion_identical_arms():
    result = two_proportion_test(10, 20, 10, 20)
    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(1.0)
    assert result.effect == 0.0
    assert result.ci_low < 0.0 < result.ci_high


@pytest.mark.parametrize(("k", "n"), [(0, 10), (10, 10)])
def test_two_proportion_degenerate_pooled_variance(k, n):
    result = two_proportion_test(k, n, k, n)
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.effect == 0.0


def test_two_proportion_extreme_arms():
    result = two_proportion_test(0, 10, 10, 10)
    assert result.p_value < 0.001
    assert result.ci_low == pytest.approx(-1.0, abs=1e-12)
    assert result.ci_high < 0.0


def test_two_proportion_swap_symmetry():
    forward = two_proportion_test(45, 100, 60, 100)
    backward = two_proportion_test(60, 100, 45, 100)
    assert backward.statistic == pytest.approx(-forward.statistic, abs=1e-12)
    assert backward.p_value == pytest.approx(forward.p_value, abs=1e-12)
    assert backward.effect == pytest.approx(-forward.effect, abs=1e-12)
    assert backward.ci_low == pytest.approx(-forward.ci_high, abs=1e-12)
    assert backward.ci_high == pytest.approx(-forward.ci_low, abs=1e-12)


def test_two_proportion_confidence_99_is_wider():
    narrow = two_proportion_test(45, 100, 60, 100, confidence=0.95)
    wide = two_proportion_test(45, 100, 60, 100, confidence=0.99)
    assert wide.ci_low < narrow.ci_low
    assert wide.ci_high > narrow.ci_high


def test_result_is_immutable():
    result = two_proportion_test(1, 10, 2, 10)
    assert isinstance(result, SignificanceResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.p_value = 0.0  # type: ignore[misc]


# --- Welch -------------------------------------------------------------------------------


def test_welch_small_example_by_hand():
    result = welch_t_test([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert result.statistic == pytest.approx(-math.sqrt(1.5), abs=1e-9)
    assert result.df == pytest.approx(4.0, abs=1e-9)
    assert result.effect == pytest.approx(-1.0, abs=1e-12)
    assert result.ci_low == pytest.approx(-3.266958, abs=1e-4)
    assert result.ci_high == pytest.approx(1.266958, abs=1e-4)
    assert 0.27 < result.p_value < 0.31


def test_welch_swap_symmetry():
    forward = welch_t_test(WELCH_A, WELCH_B)
    backward = welch_t_test(WELCH_B, WELCH_A)
    assert backward.statistic == pytest.approx(-forward.statistic, abs=1e-12)
    assert backward.p_value == pytest.approx(forward.p_value, abs=1e-12)
    assert backward.df == pytest.approx(forward.df, abs=1e-9)


def test_welch_zero_variance_equal_means():
    result = welch_t_test([1.0, 1.0], [1.0, 1.0, 1.0])
    assert result.statistic == math.inf
    assert result.p_value == 1.0
    assert result.df is None
    assert result.ci_low == result.ci_high == result.effect == 0.0


def test_welch_zero_variance_different_means():
    result = welch_t_test([2.0, 2.0], [1.0, 1.0])
    assert result.statistic == math.inf
    assert result.p_value == 0.0
    assert result.effect == 1.0
    assert (result.ci_low, result.ci_high) == (1.0, 1.0)


def test_welch_one_sample_zero_variance_is_defined():
    result = welch_t_test([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])
    assert math.isfinite(result.statistic)
    assert result.df == pytest.approx(2.0, abs=1e-9)
    assert 0.0 <= result.p_value <= 1.0


def test_welch_large_samples_and_confidence_99():
    a = [float(i % 7) for i in range(2000)]
    b = [float(i % 7) + 0.1 for i in range(2000)]
    result = welch_t_test(a, b)
    assert 3000 < result.df <= 3998.0 + 1e-6
    assert 0.05 < result.p_value < 0.2
    wide = welch_t_test(a, b, confidence=0.99)
    assert wide.ci_low < result.ci_low
    assert wide.ci_high > result.ci_high


def test_welch_accepts_tuples_and_ints():
    result = welch_t_test((1, 2, 3, 4), (3, 4, 5, 6))
    assert result.effect == pytest.approx(-2.0, abs=1e-12)


def test_welch_huge_effect_gives_zero_p_value():
    result = welch_t_test([1e149, 1e149 + 1e133], [-1e149, -1e149 + 1e133])
    assert result.p_value == 0.0


# --- permutation -------------------------------------------------------------------------


def _reference_p_value(a, b, *, paired, iterations, seed):
    rng = _Rng(seed)
    observed = math.fsum(a) / len(a) - math.fsum(b) / len(b)
    threshold = abs(observed) - 1e-12
    count = 0
    if paired:
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        for _ in range(iterations):
            flipped = [-d if rng.random() < 0.5 else d for d in diffs]
            if abs(math.fsum(flipped) / len(flipped)) >= threshold:
                count += 1
    else:
        pooled = list(a) + list(b)
        for _ in range(iterations):
            for i in range(len(pooled) - 1, 0, -1):
                j = int(rng.random() * (i + 1))
                pooled[i], pooled[j] = pooled[j], pooled[i]
            stat = math.fsum(pooled[: len(a)]) / len(a) - math.fsum(pooled[len(a) :]) / len(b)
            if abs(stat) >= threshold:
                count += 1
    return (count + 1) / (iterations + 1)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("seed", [0, 7])
def test_permutation_matches_documented_algorithm(paired, seed):
    result = permutation_test(PERM_A, PERM_B, paired=paired, iterations=300, seed=seed)
    expected = _reference_p_value(PERM_A, PERM_B, paired=paired, iterations=300, seed=seed)
    assert result.p_value == expected


def test_permutation_default_seed_is_zero():
    default = permutation_test(PERM_A, PERM_B, iterations=200)
    explicit = permutation_test(PERM_A, PERM_B, iterations=200, seed=0)
    assert default.p_value == explicit.p_value
    assert default.statistic == explicit.statistic


def test_permutation_other_seed_is_valid():
    result = permutation_test(PERM_A, PERM_B, iterations=200, seed=12345)
    assert 0.0 < result.p_value <= 1.0


@pytest.mark.parametrize("paired", [False, True])
def test_permutation_identical_samples_have_p_one(paired):
    result = permutation_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], paired=paired, iterations=200)
    assert result.p_value == 1.0
    assert result.statistic == 0.0


def test_permutation_unpaired_known_probability():
    result = permutation_test([1.0, 2.0], [10.0, 20.0], iterations=2000)
    assert result.statistic == pytest.approx(-13.5, abs=1e-12)
    assert 0.28 < result.p_value < 0.39


def test_permutation_paired_known_probability():
    result = permutation_test([1.0, 2.0, 3.0, 4.0], [0.0] * 4, paired=True, iterations=4000)
    assert result.statistic == pytest.approx(2.5, abs=1e-12)
    assert 0.09 < result.p_value < 0.16


@pytest.mark.parametrize("paired", [False, True])
def test_permutation_single_values_have_p_one(paired):
    result = permutation_test([1.0], [2.0], paired=paired, iterations=50)
    assert result.p_value == 1.0


# --- min_sample_size ---------------------------------------------------------------------


def test_min_sample_size_other_values():
    assert min_sample_size(0.1, 0.2) == 199
    assert min_sample_size(0.6, 0.45) == 173
    assert isinstance(min_sample_size(0.45, 0.6), int)
    assert min_sample_size(0.45, 0.6, power=0.9) > 173
    assert min_sample_size(0.45, 0.6, alpha=0.01) > 173
    assert min_sample_size(0.45, 0.55) > min_sample_size(0.45, 0.6)


# --- distribution helpers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.5, 0.0),
        (0.975, 1.959963985),
        (0.8, 0.841621234),
        (0.995, 2.575829304),
        (0.01, -2.326347874),
        (0.99, 2.326347874),
        (0.001, -3.090232306),
        (0.999, 3.090232306),
    ],
)
def test_norm_ppf_known_values(p, expected):
    assert norm_ppf(p) == pytest.approx(expected, abs=1e-8)


def test_norm_ppf_far_tails():
    assert norm_ppf(1e-10) == pytest.approx(-6.3613409, abs=1e-5)
    assert norm_ppf(1e-320) < -37.0


@pytest.mark.parametrize("p", [1e-8, 0.001, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98, 0.999])
def test_norm_ppf_round_trip(p):
    assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_norm_ppf_rejects_bad_probability(p):
    with pytest.raises(InvalidInput):
        norm_ppf(p)


def test_regularized_beta_known_values():
    assert regularized_beta(1.0, 1.0, 0.3) == pytest.approx(0.3, abs=1e-12)
    assert regularized_beta(2.0, 2.0, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert regularized_beta(0.5, 0.5, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert regularized_beta(2.0, 3.0, 0.9) == pytest.approx(0.9963, abs=1e-9)
    assert regularized_beta(2.0, 3.0, 0.1) == pytest.approx(0.0523, abs=1e-9)
    assert regularized_beta(2.0, 3.0, 0.0) == 0.0
    assert regularized_beta(2.0, 3.0, -1.0) == 0.0
    assert regularized_beta(2.0, 3.0, 1.0) == 1.0
    assert regularized_beta(2.0, 3.0, 1.5) == 1.0


def test_t_two_sided_p_known_values():
    assert t_two_sided_p(1.0, 1.0) == pytest.approx(0.5, abs=1e-9)
    assert t_two_sided_p(2.0, 1.0) == pytest.approx(0.2951672, abs=1e-5)
    assert t_two_sided_p(1.0, 2.0) == pytest.approx(1.0 - 1.0 / math.sqrt(3.0), abs=1e-9)
    assert t_two_sided_p(-1.0, 2.0) == pytest.approx(t_two_sided_p(1.0, 2.0), abs=1e-15)
    assert t_two_sided_p(0.0, 5.0) == 1.0
    assert t_two_sided_p(1e200, 5.0) == 0.0
    assert t_two_sided_p(math.inf, 5.0) == 0.0
    assert t_two_sided_p(1.96, 1e6) == pytest.approx(0.05, abs=1e-4)


@pytest.mark.parametrize(
    ("confidence", "df", "expected"),
    [
        (0.95, 1.0, 12.7062047),
        (0.95, 2.0, 4.3026527),
        (0.95, 10.0, 2.2281389),
        (0.95, 30.0, 2.0422725),
        (0.99, 10.0, 3.1692727),
        (0.95, 1e6, 1.959964),
    ],
)
def test_t_crit_known_values(confidence, df, expected):
    assert t_crit(confidence, df) == pytest.approx(expected, abs=1e-4)


def test_t_crit_low_confidence():
    assert 0.0 < t_crit(0.01, 10.0) < 0.1
