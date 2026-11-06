# ab-significance

[![PyPI version](https://img.shields.io/pypi/v/trysac-ab-significance)](https://pypi.org/project/trysac-ab-significance/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-ab-significance)](LICENSE)
[![CI](https://github.com/Trysac/u034-ab-significance/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u034-ab-significance/actions/workflows/ci.yml)

Pure-Python significance tests for prompt or model variants: Wilson and Newcombe intervals, two-proportion z, Welch t, seeded permutation tests.

## Problem

Teams choose a prompt or model because it scored a few points higher on a small evaluation set, without checking whether the gap could be noise. Installing SciPy for a couple of tests is heavy for small tools, CI scripts and browser notebooks. This library provides the common tests and intervals using only the Python standard library, with seeded, reproducible resampling.

## Install

```sh
pip install trysac-ab-significance
# or
uv add trysac-ab-significance
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
from ab_significance import two_proportion_test, wilson_interval

result = two_proportion_test(45, 100, 60, 100)
print(round(result.p_value, 4))
print([round(x, 4) for x in (result.ci_low, result.ci_high)])
print(wilson_interval(7, 20))
```

## Public API

- `wilson_interval(k, n, confidence=0.95) -> (low, high)`: Wilson score interval for k/n.
- `two_proportion_test(k1, n1, k2, n2, confidence=0.95) -> SignificanceResult`: pooled z test; `effect` is k1/n1 - k2/n2; interval by Newcombe's hybrid score method (1998).
- `welch_t_test(a, b, confidence=0.95) -> SignificanceResult`: Welch t with Welch-Satterthwaite degrees of freedom; `effect` is mean(a) - mean(b).
- `permutation_test(a, b, *, paired=False, iterations=10_000, seed=0) -> SignificanceResult`: two-sided seeded permutation test on the difference of means; `ci_low` and `ci_high` are NaN.
- `min_sample_size(p1, p2, alpha=0.05, power=0.8) -> int`: per-arm sample size (Fleiss, no continuity correction).
- `SignificanceResult`: frozen dataclass with `statistic`, `p_value`, `ci_low`, `ci_high`, `effect`, `df`, `method`.
- Errors: `AbSignificanceError(ValueError)` with `code`; `InvalidInput` has `code == "invalid_input"`. Messages name the parameter and value.
- Limits: at most 1,000,000 values per sample and 1,000,000 iterations; iterations times total sample size at most 500,000,000; counts at most 10**15; sample magnitudes at most 1e150; seed in [-2**64, 2**64].
- Options: `confidence` must be strictly between 0 and 1; `paired=True` requires equal lengths.

## Why not SciPy?

`scipy.stats` provides these tests but is a large compiled dependency. This library runs anywhere, including Pyodide, at the cost of covering only the tests listed above.

## Security

See [docs/SECURITY.md](docs/SECURITY.md) for the threat model and limits.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
