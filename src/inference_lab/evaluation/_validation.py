"""Primitive validation predicates for evaluation contracts."""

import re
from typing import Final

_SHA256_CHECKSUM_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")


def is_sha256_checksum(value: str) -> bool:
    """Return whether a string is a canonical prefixed SHA-256 checksum."""
    return _SHA256_CHECKSUM_PATTERN.fullmatch(value) is not None
