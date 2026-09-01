"""Schema-version policy for derived-record lineage contracts."""

from typing import Final

from inference_lab.provenance.errors import UnsupportedLineageSchemaVersionError

CURRENT_LINEAGE_SCHEMA_VERSION: Final[str] = "1.0.0"
SUPPORTED_LINEAGE_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    {CURRENT_LINEAGE_SCHEMA_VERSION}
)


def validate_lineage_schema_version(value: object) -> str:
    """Return a supported lineage schema version after strict validation."""
    if not isinstance(value, str):
        raise TypeError("lineage_schema_version must be a string.")

    if value not in SUPPORTED_LINEAGE_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_LINEAGE_SCHEMA_VERSIONS))
        raise UnsupportedLineageSchemaVersionError(
            f"unsupported lineage_schema_version; supported versions: {supported}."
        )

    return value
