"""Probability distributions needed by the significance tests (standard library only).

The normal CDF uses ``math.erfc``. The inverse normal CDF is Acklam's rational approximation
refined with one Halley step. The Student t distribution goes through the regularized
incomplete beta function (Lentz continued fraction, Numerical Recipes 6.4) and
``math.lgamma``. Inverse t is found by bisection on the tail probability.
"""

from __future__ import annotations

import math

from .errors import InvalidInput

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_REFINE_LIMIT = 37.0

_FPMIN = 1e-300
_EPS = 1e-14
_MAX_ITER = 5000
_MAX_BRACKET = 200
_MAX_BISECT = 200
_T_TOLERANCE = 1e-12


def _horner(coefficients: tuple[float, ...], x: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * x + coefficient
    return result


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * math.erfc(-x / _SQRT2)


def norm_sf(x: float) -> float:
    """Standard normal survival function ``1 - cdf(x)`` without cancellation."""
    return 0.5 * math.erfc(x / _SQRT2)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF for ``0 < p < 1``.

    Raises:
        InvalidInput: if ``p`` is not strictly between 0 and 1.
    """
    if not 0.0 < p < 1.0:
        raise InvalidInput(f"probability must be in (0, 1), got {p!r}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = _horner(_C, q) / (_horner(_D, q) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = _horner(_A, r) * q / (_horner(_B, r) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -_horner(_C, q) / (_horner(_D, q) * q + 1.0)
    if abs(x) > _REFINE_LIMIT:
        return x
    error = (1.0 - p) - norm_sf(x) if x > 0.0 else norm_cdf(x) - p
    u = error * _SQRT_2PI * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def _clamp(value: float) -> float:
    return value if abs(value) >= _FPMIN else _FPMIN


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 / _clamp(1.0 - qab * x / qap)
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / _clamp(1.0 + aa * d)
        c = _clamp(1.0 + aa / c)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / _clamp(1.0 + aa * d)
        c = _clamp(1.0 + aa / c)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def regularized_beta(a: float, b: float, x: float, y: float | None = None) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``.

    ``y`` may carry ``1 - x`` computed without cancellation; it defaults to ``1 - x``.
    Values of ``x`` at or below 0 give 0.0 and at or above 1 give 1.0.
    """
    complement = 1.0 - x if y is None else y
    if x <= 0.0:
        return 0.0
    if complement <= 0.0:
        return 1.0
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_norm + a * math.log(x) + b * math.log(complement))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, complement) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of Student's t statistic ``t`` with ``df`` degrees of freedom."""
    t_sq = t * t
    if math.isinf(t_sq):
        return 0.0
    denominator = df + t_sq
    value = regularized_beta(df / 2.0, 0.5, df / denominator, t_sq / denominator)
    return min(1.0, max(0.0, value))


def t_crit(confidence: float, df: float) -> float:
    """Two-sided critical value of Student's t, found by bisection to 1e-12."""
    tail = (1.0 - confidence) / 2.0
    low = 0.0
    high = 1.0
    for _ in range(_MAX_BRACKET):
        if 0.5 * t_two_sided_p(high, df) <= tail:
            break
        low = high
        high *= 2.0
    for _ in range(_MAX_BISECT):
        if high - low <= _T_TOLERANCE:
            break
        mid = 0.5 * (low + high)
        if 0.5 * t_two_sided_p(mid, df) > tail:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
