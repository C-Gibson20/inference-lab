"""Version policy for the retrieval serialisation contract."""

from typing import Final

from inference_lab.retrieval.errors import UnsupportedRetrievalSchemaVersionError

CURRENT_RETRIEVAL_SCHEMA_VERSION: Final[str] = "1.0.0"
SUPPORTED_RETRIEVAL_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    {CURRENT_RETRIEVAL_SCHEMA_VERSION}
)


def validate_retrieval_schema_version(value: object) -> str:
    """Return a supported schema version or reject it before nested parsing."""
    if not isinstance(value, str):
        raise TypeError("retrieval_schema_version must be a string.")
    if value not in SUPPORTED_RETRIEVAL_SCHEMA_VERSIONS:
        raise UnsupportedRetrievalSchemaVersionError(
            "unsupported retrieval schema version."
        )
    return value
