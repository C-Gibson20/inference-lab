"""Primitive validation predicates for provenance values."""

import re
from typing import Final

_SHA256_HEX_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


def is_sha256_hex_digest(value: str) -> bool:
    """Return whether a string is a canonical lowercase SHA-256 digest."""
    return _SHA256_HEX_PATTERN.fullmatch(value) is not None
