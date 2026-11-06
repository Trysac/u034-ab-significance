"""Behavior tests for the public API of ab_significance."""

from __future__ import annotations

import pytest

import ab_significance


def test_normalize_trims_and_lowercases() -> None:
    assert ab_significance.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ab_significance.normalize(value)


def test_version_is_semver() -> None:
    parts = ab_significance.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
