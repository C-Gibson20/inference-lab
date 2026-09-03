"""Shared validation predicates for project-wide scalar formats."""

import os
import re
from pathlib import Path
from typing import Final

_URI_SCHEME_PATTERN: Final = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE_PATTERN: Final = re.compile(r"^[A-Za-z]:[\\/]")
_SEMANTIC_VERSION_PATTERN: Final = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


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
