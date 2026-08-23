"""Utility functions for profiling."""

from datetime import datetime, UTC
import re


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)


def is_snake_case(s: str) -> bool:
    """Return True if the string is in snake_case, False otherwise."""
    return bool(re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", s))


def validate_record_schema_version(value: object) -> None:
    """Validate a positive, non-boolean profiling record schema version."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("record_schema_version must be a positive integer")
