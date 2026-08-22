"""Evaluation schema and dataset version policy."""

from typing import Final

CURRENT_SCHEMA_VERSION: Final[str] = "1.0.0"
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({CURRENT_SCHEMA_VERSION})


def validate_schema_version(value: str) -> str:
    """Return a supported schema version or raise ValueError."""
    if value not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ValueError(
            f"Unsupported evaluation schema version {value!r}; "
            f"supported versions: {supported}"
        )
    return value
