# Security notes

## Threat model

`trysac-ab-significance` is a pure library without I/O. It does not open files, sockets or processes, does not deserialize data and does not call `eval` or `exec`.

### Defended

- **Invalid input**: arguments are type-checked and range-checked; failures raise `InvalidInput` with a message naming the parameter and value. Samples and whole inputs are not echoed; values are bounded in length.
- **Resource exhaustion**: samples are limited to 1,000,000 values, iterations to 1,000,000, and `iterations * (len(a) + len(b))` to 500,000,000. Counts are limited to 10**15 and magnitudes to 1e150.
- **Non-finite values**: NaN and infinity inputs are rejected.
- **Supply chain**: no runtime dependencies; Dependabot weekly updates; `gitleaks` in local gates and CI.

### Not defended

- **Cryptographic randomness**: permutation tests use `random.Random(seed)`, a seeded non-cryptographic generator, by design for reproducibility. Do not use it for secrets.
- **Numerical accuracy at extremes**: p-values and intervals are floating-point approximations; the t quantile is found by bisection and the normal quantile by an approximation refined once. Results near the limits of the input ranges are not guaranteed beyond the stated tolerances.
- **Side channels and timing**: no protection is provided.
- **Malicious callers with access to the process**: the library trusts its host process.

## Publishing

Releases are published to PyPI with trusted publishing (OIDC) when available, never with a long-lived token stored in the repository.

To report a vulnerability, see [SECURITY.md](../SECURITY.md).
