# Architecture

## Purpose

`trysac-ab-significance` (import name `ab_significance`) is a small, dependency-free Python library that runs significance tests and confidence intervals for A/B comparisons of prompt or model variants.

## Shape

```text
caller code --> ab_significance (public API in __init__.py)
                    |
                    +-- _core.py   tests, intervals, input validation
                    +-- _dist.py   normal and Student t distributions
                    +-- errors.py  AbSignificanceError, InvalidInput
```

- **Public API**: the names listed in `__all__` of `src/ab_significance/__init__.py`: `wilson_interval`, `two_proportion_test`, `welch_t_test`, `permutation_test`, `min_sample_size`, `SignificanceResult`, `InvalidInput`, `AbSignificanceError`, `MAX_SAMPLE_SIZE`, `MAX_ITERATIONS`, `__version__`.
- **Math**: normal CDF via `math.erfc`; inverse normal by Acklam's approximation with one Halley step; Student t through the regularized incomplete beta function (Lentz continued fraction) and `math.lgamma`; inverse t by bisection to 1e-12.
- **Errors**: invalid input raises `InvalidInput`, a subclass of `AbSignificanceError(ValueError)`.
- **Side effects**: none. Functions perform no I/O and hold no state. Logging uses `logging.getLogger(__name__)` with a `NullHandler`; the library never configures logging.

## Quality contract

See [TESTING.md](TESTING.md) for the test suite and the coverage policy.

## Decisions

No architecture decision records are present in this repository yet.
