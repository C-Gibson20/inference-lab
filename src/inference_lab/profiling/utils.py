"""Utility functions for profiling."""

from datetime import datetime, UTC


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)
