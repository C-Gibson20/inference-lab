"""Shared validation predicates for project-wide scalar formats."""

import re
from typing import Final

_SEMANTIC_VERSION_PATTERN: Final = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def is_canonical_semantic_version(value: object) -> bool:
    """Return whether a value is a canonical Semantic Version string."""
    return (
        isinstance(value, str)
        and _SEMANTIC_VERSION_PATTERN.fullmatch(value) is not None
    )
