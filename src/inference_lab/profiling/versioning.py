"""Schema-version policy for profiling records."""

from typing import Final

CURRENT_RECORD_SCHEMA_VERSION: Final[int] = 2
SUPPORTED_RECORD_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset(
    {CURRENT_RECORD_SCHEMA_VERSION}
)


def validate_record_schema_version(value: object) -> None:
    """Reject profiling record schema versions this build cannot interpret."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("record_schema_version must be an integer")
    if value not in SUPPORTED_RECORD_SCHEMA_VERSIONS:
        supported = ", ".join(
            str(version) for version in sorted(SUPPORTED_RECORD_SCHEMA_VERSIONS)
        )
        raise ValueError(
            f"unsupported record_schema_version {value!r}; supported versions: "
            f"{supported}"
        )
