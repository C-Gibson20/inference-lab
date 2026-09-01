"""Shared, side-effect-free validation helpers for provenance values."""

import re
from typing import Final

_LOWERCASE_SLUG_PATTERN: Final = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SHA256_HEX_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


def is_lowercase_slug(value: str) -> bool:
    """Return whether a string is a canonical lowercase hyphenated slug."""
    return _LOWERCASE_SLUG_PATTERN.fullmatch(value) is not None


def is_sha256_hex_digest(value: str) -> bool:
    """Return whether a string is a canonical lowercase SHA-256 digest."""
    return _SHA256_HEX_PATTERN.fullmatch(value) is not None
