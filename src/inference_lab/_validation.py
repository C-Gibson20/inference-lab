"""Shared validation predicates for project-wide scalar formats."""

import os
import re
from pathlib import Path
from typing import Final

_URI_SCHEME_PATTERN: Final = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE_PATTERN: Final = re.compile(r"^[A-Za-z]:[\\/]")
_LOWERCASE_SLUG_PATTERN: Final = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_CANONICAL_MEDIA_TYPE_PATTERN: Final = re.compile(
    r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+"
)
CANONICAL_SEMANTIC_VERSION_PATTERN: Final = (
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_SEMANTIC_VERSION_PATTERN: Final = re.compile(CANONICAL_SEMANTIC_VERSION_PATTERN)


def is_strict_integer(value: object) -> bool:
    """Return whether a runtime value is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_lowercase_slug(value: object) -> bool:
    """Return whether a value is a canonical lowercase hyphenated slug."""
    return (
        isinstance(value, str) and _LOWERCASE_SLUG_PATTERN.fullmatch(value) is not None
    )


def is_canonical_media_type(value: object) -> bool:
    """Return whether a value is a canonical parameter-free media type."""
    return (
        isinstance(value, str)
        and _CANONICAL_MEDIA_TYPE_PATTERN.fullmatch(value) is not None
    )


def paths_overlap(first: Path, second: Path) -> bool:
    """Return whether two paths are equal or contain one another."""
    if not isinstance(first, Path):
        raise TypeError("first must be a Path")
    if not isinstance(second, Path):
        raise TypeError("second must be a Path")
    return first == second or first in second.parents or second in first.parents


def is_local_path_shape(value: object) -> bool:
    """Return whether a value names a local path on the current host.

    This predicate checks syntax only. It does not resolve the path, inspect the
    filesystem, or create directories.
    """
    if not isinstance(value, (str, Path)):
        return False

    raw_path = os.fspath(value)
    if not raw_path.strip() or "\x00" in raw_path:
        return False
    if raw_path.startswith(("//", "\\\\")):
        return False
    if _WINDOWS_DRIVE_PATTERN.match(raw_path) is not None:
        return os.name == "nt"
    return _URI_SCHEME_PATTERN.match(raw_path) is None


def is_canonical_semantic_version(value: object) -> bool:
    """Return whether a value is a canonical Semantic Version string."""
    return (
        isinstance(value, str)
        and _SEMANTIC_VERSION_PATTERN.fullmatch(value) is not None
    )
