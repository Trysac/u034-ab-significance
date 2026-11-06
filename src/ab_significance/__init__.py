"""Pure-Python significance tests for prompt or model variants.

Wilson and Newcombe intervals, two-proportion z, Welch t and seeded permutation tests,
using only the standard library.
"""

from __future__ import annotations

import logging

from ._core import (
    MAX_ITERATIONS,
    MAX_SAMPLE_SIZE,
    SignificanceResult,
    min_sample_size,
    permutation_test,
    two_proportion_test,
    welch_t_test,
    wilson_interval,
)
from .errors import AbSignificanceError, InvalidInput

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "MAX_ITERATIONS",
    "MAX_SAMPLE_SIZE",
    "AbSignificanceError",
    "InvalidInput",
    "SignificanceResult",
    "__version__",
    "min_sample_size",
    "permutation_test",
    "two_proportion_test",
    "welch_t_test",
    "wilson_interval",
]
