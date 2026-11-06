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
| Tokens (input / output) | 343446 / 75713 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.6627 |

## Factory run u034 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | claude-sonnet-5-5 | yes | 2502 | 65391 | 0 | 4246 | 0.3299 |
| fix-1 | claude-sonnet-5-5 | no | 4 | 352 | 5718 | 24416 | 0.0203 |
| fix-2 | claude-sonnet-5-5 | no | 6 | 743 | 7584 | 41880 | 0.0306 |
| audit | claude-sonnet-5-5 | no | 10 | 3082 | 47559 | 86596 | 0.1584 |
| fix-3 | claude-sonnet-5-5 | no | 8 | 1791 | 10913 | 61005 | 0.0513 |
| docs | claude-haiku-5-5 | no | 122773 | 4354 | 0 | 0 | 0.0723 |

Total: $0.6627. Prices: Anthropic list prices.
