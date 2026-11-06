# Usage

`ab_significance` provides significance tests for comparing two prompt or model variants, using only the Python standard library.

## Functions

- `wilson_interval(k, n, confidence=0.95)` returns the Wilson score interval of a proportion.
- `two_proportion_test(k1, n1, k2, n2, confidence=0.95)` runs a pooled z test. The interval of `k1/n1 - k2/n2` is Newcombe's hybrid score method 10.
- `welch_t_test(a, b, confidence=0.95)` runs Welch's t test with Welch-Satterthwaite degrees of freedom.
- `permutation_test(a, b, paired=False, iterations=10_000, seed=0)` runs a two-sided seeded permutation test. Its interval fields are NaN.
- `min_sample_size(p1, p2, alpha=0.05, power=0.8)` returns the per-arm sample size of Fleiss without continuity correction.

All functions return a frozen `SignificanceResult` (except the interval and size helpers) and raise `InvalidInput` (code `invalid_input`, a `ValueError`) on bad input.

## Input limits

- Samples hold at most 1,000,000 values, each finite with magnitude at most 1e150.
- `iterations` is between 1 and 1,000,000.
- Counts are plain `int` values (not `bool`) up to 10**15.

## Edge cases

- Welch with zero variance in both samples: `statistic` is `inf`, `df` is `None`, `p_value` is 1.0 when the means are equal and 0.0 otherwise.
- A permutation p-value is `(count + 1) / (iterations + 1)`, so it is never 0.
- Only `random.Random(seed).random()` is used, so results repeat for a given seed.

## Reference values

The reference values in the test suite were computed once with SciPy 1.x and statsmodels. Neither is a dependency.
