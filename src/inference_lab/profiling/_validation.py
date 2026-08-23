"""Shared primitive validation for profiling record contracts."""

from datetime import datetime, timedelta


def validate_utc_datetime(value: datetime, field_name: str) -> None:
    """Require a timezone-aware UTC datetime."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be in UTC")
