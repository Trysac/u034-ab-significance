# Testing

## Running

```sh
uv run pytest --cov=ab_significance   # tests with coverage report
uv run python scripts/check.py      # every gate, as CI and the factory run it
```

## Test files

- `tests/test_ab_significance.py`: the only test module. It imports the public API and checks the reference values from the specification for `two_proportion_test`, `wilson_interval`, `welch_t_test`, `permutation_test` (seed reproducibility, NaN interval fields, paired and unpaired p-values) and `min_sample_size`, plus the `InvalidInput` error cases. The reference values were computed with SciPy and statsmodels, which are not dependencies of the library.

## Policy

- Branch coverage of `src/ab_significance` must stay at or above 80 %; the gate fails below it.
- Tests exercise the public API (`import ab_significance`) and assert return values, raised exceptions and their messages.
- Tests are deterministic: no network, no wall clock, no dependence on test order.

## Layout

`tests/` contains `test_ab_significance.py`. No `conftest.py` is present.
