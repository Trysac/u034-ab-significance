"""Significance tests and confidence intervals for A/B comparisons.

All functions are pure: no I/O, no state, no clock. Randomness comes only from
``random.Random(seed).random()``, so resampling results are reproducible.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from ._dist import norm_ppf, t_crit, t_two_sided_p
from .errors import InvalidInput

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Largest number of values accepted in one sample.
MAX_SAMPLE_SIZE = 1_000_000
# Largest number of permutations accepted.
MAX_ITERATIONS = 1_000_000

_MAX_COUNT = 10**15
_MAX_ABS_VALUE = 1e150
_MAX_SEED = 2**64
_DESCRIBE_INT_LIMIT = 10**30
_PERMUTATION_TOLERANCE = 1e-12
_SQRT2 = math.sqrt(2.0)

# Seeded, non-cryptographic generator by design: reproducibility is the requirement.
_Rng = random.Random


@dataclass(frozen=True)
class SignificanceResult:
    """Outcome of a significance test.

    Attributes:
        statistic: the test statistic (z, t, or the observed mean difference).
        p_value: two-sided p-value.
        ci_low: lower bound of the confidence interval of ``effect``; NaN for permutation tests.
        ci_high: upper bound of the confidence interval of ``effect``; NaN for permutation tests.
        effect: estimated difference (first arm minus second arm).
        df: degrees of freedom, or None when the method has none.
        method: short identifier of the method used.
    """

    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    effect: float
    df: float | None
    method: str


def _describe(value: object) -> str:
    """Short, bounded description of an argument for error messages."""
    if isinstance(value, bool | float):
        return repr(value)
    if isinstance(value, int):
        return repr(value) if abs(value) < _DESCRIBE_INT_LIMIT else "<large int>"
    if isinstance(value, str):
        return repr(value[:40])
    return f"<{type(value).__name__}>"


def _check_int(name: str, value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInput(f"{name} must be an int, got {_describe(value)}")
    if not low <= value <= high:
        raise InvalidInput(f"{name} must be in [{low}, {high}], got {_describe(value)}")
    return value


def _check_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidInput(f"{name} must be a real number, got {_describe(value)}")
    try:
        number = float(value)
    except OverflowError:
        raise InvalidInput(f"{name} is too large, got {_describe(value)}") from None
    if not math.isfinite(number):
        raise InvalidInput(f"{name} must be finite, got {_describe(value)}")
    return number


def _check_open_unit(name: str, value: object) -> float:
    number = _check_real(name, value)
    if not 0.0 < number < 1.0:
        raise InvalidInput(f"{name} must be strictly between 0 and 1, got {_describe(value)}")
    return number


def _check_counts(k_name: str, k: object, n_name: str, n: object) -> tuple[int, int]:
    trials = _check_int(n_name, n, 1, _MAX_COUNT)
    successes = _check_int(k_name, k, 0, _MAX_COUNT)
    if successes > trials:
        raise InvalidInput(f"{k_name} must not exceed {n_name}, got {successes} and {trials}")
    return successes, trials


def _check_sample(name: str, value: object, minimum: int) -> list[float]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise InvalidInput(f"{name} must be a sequence of numbers, got {_describe(value)}")
    size = len(value)
    if size < minimum or size > MAX_SAMPLE_SIZE:
        raise InvalidInput(f"{name} must hold {minimum} to {MAX_SAMPLE_SIZE} values, got {size}")
    values: list[float] = []
    for index, item in enumerate(value):
        number = _check_real(f"{name}[{index}]", item)
        if abs(number) > _MAX_ABS_VALUE:
            raise InvalidInput(f"{name}[{index}] must not exceed 1e150 in magnitude")
        values.append(number)
    return values


def _z_two_sided(confidence: float) -> float:
    return -norm_ppf((1.0 - confidence) / 2.0)


def _wilson(k: int, n: int, z: float) -> tuple[float, float]:
    proportion = k / n
    z_sq = z * z
    denominator = 1.0 + z_sq / n
    centre = (proportion + z_sq / (2.0 * n)) / denominator
    spread = proportion * (1.0 - proportion) / n + z_sq / (4.0 * n * n)
    half = z * math.sqrt(spread) / denominator
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``.

    Args:
        k: number of successes, ``0 <= k <= n``, an ``int`` (not ``bool``).
        n: number of trials, ``n >= 1``.
        confidence: confidence level strictly between 0 and 1.

    Returns:
        ``(low, high)``, clamped to ``[0, 1]``; ``low`` is exactly 0.0 when ``k == 0`` and
        ``high`` is exactly 1.0 when ``k == n``.

    Raises:
        InvalidInput: on any invalid argument.
    """
    successes, trials = _check_counts("k", k, "n", n)
    level = _check_open_unit("confidence", confidence)
    return _wilson(successes, trials, _z_two_sided(level))


def two_proportion_test(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    confidence: float = 0.95,
) -> SignificanceResult:
    """Pooled two-proportion z test with a Newcombe hybrid score interval.

    ``effect`` is ``k1/n1 - k2/n2``. The interval is Newcombe's method 10 (1998), built from
    the Wilson intervals of both arms. When both arms are all successes or all failures the
    pooled standard error is zero and the result is ``statistic=0.0``, ``p_value=1.0``.

    Raises:
        InvalidInput: on any invalid argument.
    """
    successes1, trials1 = _check_counts("k1", k1, "n1", n1)
    successes2, trials2 = _check_counts("k2", k2, "n2", n2)
    level = _check_open_unit("confidence", confidence)
    p1 = successes1 / trials1
    p2 = successes2 / trials2
    pooled = (successes1 + successes2) / (trials1 + trials2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / trials1 + 1.0 / trials2))
    z_stat = (p1 - p2) / se if se > 0.0 else 0.0
    p_value = math.erfc(abs(z_stat) / _SQRT2)
    z = _z_two_sided(level)
    low1, high1 = _wilson(successes1, trials1, z)
    low2, high2 = _wilson(successes2, trials2, z)
    effect = p1 - p2
    ci_low = effect - math.sqrt((p1 - low1) ** 2 + (high2 - p2) ** 2)
    ci_high = effect + math.sqrt((high1 - p1) ** 2 + (p2 - low2) ** 2)
    logger.debug("two_proportion_test: n1=%d n2=%d", trials1, trials2)
    return SignificanceResult(
        statistic=z_stat,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        effect=effect,
        df=None,
        method="two_proportion_z",
    )


def welch_t_test(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
) -> SignificanceResult:
    """Welch's unequal-variance t test with Welch-Satterthwaite degrees of freedom.

    ``effect`` is ``mean(a) - mean(b)`` and the interval is ``effect +/- t_crit * se``.
    Each sample needs at least 2 values. When both samples have zero variance the t statistic
    is undefined: the result has ``statistic=inf``, ``df=None``, a degenerate interval
    ``(effect, effect)`` and ``p_value`` 1.0 when the means are equal, otherwise 0.0.

    Raises:
        InvalidInput: on any invalid argument.
    """
    xs = _check_sample("a", a, 2)
    ys = _check_sample("b", b, 2)
    level = _check_open_unit("confidence", confidence)
    n1 = len(xs)
    n2 = len(ys)
    mean1 = math.fsum(xs) / n1
    mean2 = math.fsum(ys) / n2
    term1 = math.fsum((x - mean1) * (x - mean1) for x in xs) / (n1 - 1) / n1
    term2 = math.fsum((y - mean2) * (y - mean2) for y in ys) / (n2 - 1) / n2
    effect = mean1 - mean2
    se_sq = term1 + term2
    logger.debug("welch_t_test: n_a=%d n_b=%d", n1, n2)
    if se_sq == 0.0:
        return SignificanceResult(
            statistic=math.inf,
            p_value=1.0 if effect == 0.0 else 0.0,
            ci_low=effect,
            ci_high=effect,
            effect=effect,
            df=None,
            method="welch_t",
        )
    weight1 = term1 / se_sq
    weight2 = term2 / se_sq
    df = 1.0 / (weight1 * weight1 / (n1 - 1) + weight2 * weight2 / (n2 - 1))
    se = math.sqrt(se_sq)
    statistic = effect / se
    half = t_crit(level, df) * se
    return SignificanceResult(
        statistic=statistic,
        p_value=t_two_sided_p(statistic, df),
        ci_low=effect - half,
        ci_high=effect + half,
        effect=effect,
        df=df,
        method="welch_t",
    )


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    paired: bool = False,
    iterations: int = 10_000,
    seed: int = 0,
) -> SignificanceResult:
    """Two-sided seeded permutation test on the difference of means.

    ``statistic`` and ``effect`` are ``mean(a) - mean(b)``. The confidence interval fields
    are NaN because no interval is computed.

    Unpaired: the pooled list ``a + b`` is Fisher-Yates shuffled in place for ``i`` from
    ``len - 1`` down to 1 with ``j = int(rng.random() * (i + 1))``; the first ``len(a)``
    items form the new first group. Paired: each difference ``a[i] - b[i]`` is sign-flipped
    when ``rng.random() < 0.5``. The generator is ``random.Random(seed)`` and only its
    ``random()`` method is used. ``count`` is the number of permutations with
    ``abs(stat) >= abs(observed) - 1e-12`` and ``p_value = (count + 1) / (iterations + 1)``.

    Raises:
        InvalidInput: on any invalid argument, including unequal lengths when ``paired``.
    """
    if not isinstance(paired, bool):
        raise InvalidInput(f"paired must be a bool, got {_describe(paired)}")
    first = _check_sample("a", a, 1)
    second = _check_sample("b", b, 1)
    if paired and len(first) != len(second):
        raise InvalidInput(f"paired samples need equal length, got {len(first)} and {len(second)}")
    rounds = _check_int("iterations", iterations, 1, MAX_ITERATIONS)
    seed_value = _check_int("seed", seed, -_MAX_SEED, _MAX_SEED)
    size_a = len(first)
    size_b = len(second)
    observed = math.fsum(first) / size_a - math.fsum(second) / size_b
    threshold = abs(observed) - _PERMUTATION_TOLERANCE
    rng = _Rng(seed_value)
    count = 0
    logger.debug(
        "permutation_test: n_a=%d n_b=%d rounds=%d paired=%s", size_a, size_b, rounds, paired
    )
    if paired:
        diffs = [x - y for x, y in zip(first, second, strict=True)]
        for _ in range(rounds):
            flipped = [-d if rng.random() < 0.5 else d for d in diffs]
            if abs(math.fsum(flipped) / size_a) >= threshold:
                count += 1
    else:
        pooled = first + second
        last = len(pooled) - 1
        for _ in range(rounds):
            for i in range(last, 0, -1):
                j = int(rng.random() * (i + 1))
                pooled[i], pooled[j] = pooled[j], pooled[i]
            stat = math.fsum(pooled[:size_a]) / size_a - math.fsum(pooled[size_a:]) / size_b
            if abs(stat) >= threshold:
                count += 1
    return SignificanceResult(
        statistic=observed,
        p_value=(count + 1) / (rounds + 1),
        ci_low=math.nan,
        ci_high=math.nan,
        effect=observed,
        df=None,
        method="paired_permutation" if paired else "permutation",
    )


def min_sample_size(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size of Fleiss, without continuity correction.

    ``ceil((z_{1-alpha/2} * sqrt(2 * pbar * (1 - pbar)) + z_{power} * sqrt(p1 * (1 - p1)
    + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2)`` with ``pbar = (p1 + p2) / 2``.

    Raises:
        InvalidInput: if a probability is outside (0, 1), ``p1 == p2``, the power is too low
            for the alpha to give a meaningful size, or the size is not representable.
    """
    prob1 = _check_open_unit("p1", p1)
    prob2 = _check_open_unit("p2", p2)
    level = _check_open_unit("alpha", alpha)
    target = _check_open_unit("power", power)
    if prob1 == prob2:
        raise InvalidInput("p1 and p2 must differ")
    z_alpha = -norm_ppf(level / 2.0)
    z_power = norm_ppf(target)
    pbar = (prob1 + prob2) / 2.0
    numerator = z_alpha * math.sqrt(2.0 * pbar * (1.0 - pbar)) + z_power * math.sqrt(
        prob1 * (1.0 - prob1) + prob2 * (1.0 - prob2)
    )
    if numerator <= 0.0:
        raise InvalidInput("power is too low for the given alpha")
    ratio = numerator / abs(prob1 - prob2)
    required = ratio * ratio
    if not math.isfinite(required):
        raise InvalidInput("required sample size is not representable")
    return math.ceil(required)
