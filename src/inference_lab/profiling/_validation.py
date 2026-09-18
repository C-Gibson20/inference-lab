"""Shared primitive validation for profiling record contracts."""

import re
from datetime import datetime, timedelta
from typing import Final

_SNAKE_CASE_PATTERN: Final = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")
_CORRELATION_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")


def is_snake_case(value: str) -> bool:
    """Return whether a string is canonical snake case."""
    return _SNAKE_CASE_PATTERN.fullmatch(value) is not None


def validate_utc_datetime(value: datetime, field_name: str) -> None:
    """Require a timezone-aware UTC datetime."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be in UTC")


def validate_optional_correlation(value: object, field_name: str) -> None:
    """Validate an opaque, payload-free correlation value when present."""
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    if value == "":
        raise ValueError(f"{field_name} must not be empty")
    if _CORRELATION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a non-empty opaque correlation value")
