# ab-significance

[![PyPI version](https://img.shields.io/pypi/v/trysac-ab-significance)](https://pypi.org/project/trysac-ab-significance/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-ab-significance)](LICENSE)
[![CI](https://github.com/Trysac/u034-ab-significance/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u034-ab-significance/actions/workflows/ci.yml)

Pure-Python significance tests for prompt or model variants: Wilson and Newcombe intervals, two-proportion z, Welch t, seeded permutation tests

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-ab-significance
# or
uv add trysac-ab-significance
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import ab_significance

# normalize() trims surrounding whitespace and lower-cases the text.
print(ab_significance.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    ab_significance.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
