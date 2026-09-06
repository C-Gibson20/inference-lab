"""Schema-version policy for durable source and parse records."""

from typing import Final

CURRENT_SOURCE_RECORD_SCHEMA_VERSION: Final[str] = "1.0.0"
SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    {CURRENT_SOURCE_RECORD_SCHEMA_VERSION}
)


def validate_source_record_schema_version(value: object) -> str:
    """Return a supported source-record schema version after strict validation."""
    if not isinstance(value, str):
        raise TypeError("record_schema_version must be a string.")
    if value not in SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS:
        raise ValueError("record_schema_version is not supported by this build.")
    return value
