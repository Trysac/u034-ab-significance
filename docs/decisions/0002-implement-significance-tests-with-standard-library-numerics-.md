# Implement significance tests with standard-library numerics and a documented resampling algorithm

Status: accepted (generated, pending human review)

## Context

Teams compare prompt or model variants on small samples and need p-values and intervals in small tools, CI scripts and browser notebooks (Pyodide). SciPy is a large compiled dependency, and resampling results must be reproducible across Python versions.

## Decision

Implement the tests as pure functions using only the standard library. The normal CDF uses math.erfc and the inverse normal uses Acklam's approximation plus one Halley step. The Student t distribution uses the regularized incomplete beta function (Lentz continued fraction, math.lgamma), and quantiles are found by bisection. Sums use math.fsum. Permutation tests use only random.Random(seed).random(), a hand-written Fisher-Yates shuffle and sign flips, so results do not depend on the Python version. Every input is validated against named limits and raises InvalidInput, a subclass of AbSignificanceError(ValueError) with a stable code.

## Consequences

The library runs anywhere Python runs, has no dependencies and gives reproducible results. Accuracy is about 1e-10 against the reference values rather than SciPy's last digit. Permutation tests are slower than vectorized SciPy code. The scope is limited to the five functions: no Bayesian, sequential, multiple-comparison or bootstrap methods. Input limits (1,000,000 values, 1e150 magnitude, 10**15 counts) are deliberate and documented.
