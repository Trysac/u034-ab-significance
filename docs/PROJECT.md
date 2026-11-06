# Project record

## Objective

Provide pure-Python significance tests and confidence intervals for comparing two prompt or model variants, so that a reported difference can be checked against sampling noise without installing SciPy. Reference values are checked against SciPy and statsmodels.

## Scope

- In scope: Wilson score intervals, two-proportion z test with Newcombe hybrid score interval, Welch t test, seeded two-sided permutation tests (paired and unpaired), and per-arm sample size by Fleiss without continuity correction.
- Out of scope: Bayesian A/B testing, sequential testing, multiple-comparison corrections, bootstrap intervals, plotting, and any runtime dependency or I/O without an accepted ADR.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | pending measurement |
| Hours (human) | pending measurement |
| Cost (USD) | pending measurement |
