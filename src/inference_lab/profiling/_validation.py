"""Shared primitive validation for profiling record contracts."""

import re
from datetime import datetime, timedelta
from typing import Final

_SNAKE_CASE_PATTERN: Final = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")


def is_snake_case(value: str) -> bool:
    """Return whether a string is canonical snake case."""
    return _SNAKE_CASE_PATTERN.fullmatch(value) is not None


def validate_utc_datetime(value: datetime, field_name: str) -> None:
    """Require a timezone-aware UTC datetime."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be in UTC")
