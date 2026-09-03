"""Schema-version policy for profiling records."""

from typing import Final

from inference_lab._validation import is_canonical_semantic_version

CURRENT_RECORD_SCHEMA_VERSION: Final[str] = "3.0.0"
SUPPORTED_RECORD_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    {CURRENT_RECORD_SCHEMA_VERSION}
)


def validate_record_schema_version(value: object) -> None:
    """Reject profiling record schema versions this build cannot interpret."""
    if not isinstance(value, str):
        raise TypeError("record_schema_version must be a string")
    if not is_canonical_semantic_version(value):
        raise ValueError("record_schema_version must be a semantic version string")
    if value not in SUPPORTED_RECORD_SCHEMA_VERSIONS:
        supported = ", ".join(
            str(version) for version in sorted(SUPPORTED_RECORD_SCHEMA_VERSIONS)
        )
        raise ValueError(
            f"unsupported record_schema_version {value!r}; supported versions: "
            f"{supported}"
        )
