"""Typed errors raised by ab_significance."""

from __future__ import annotations

_MAX_MESSAGE_LENGTH = 200


class AbSignificanceError(ValueError):
    """Base class of every error raised by this library.

    Attributes:
        code: stable machine-readable identifier of the error kind.
    """

    code: str = "ab_significance_error"

    def __init__(self, message: str) -> None:
        super().__init__(message[:_MAX_MESSAGE_LENGTH])


class InvalidInput(AbSignificanceError):
    """An argument is of the wrong type, out of range, too large or not finite."""

    code = "invalid_input"
