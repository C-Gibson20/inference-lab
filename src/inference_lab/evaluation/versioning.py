"""Evaluation dataset and split-algorithm version policy."""

from typing import Any, Final


CURRENT_SCHEMA_VERSION: Final[str] = "2.0.0"
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({CURRENT_SCHEMA_VERSION})

CURRENT_SPLIT_ALGORITHM_VERSION: Final[str] = "1.0.0"
SUPPORTED_SPLIT_ALGORITHM_VERSIONS: Final[frozenset[str]] = frozenset(
    {CURRENT_SPLIT_ALGORITHM_VERSION}
)


def validate_schema_version(value: object) -> str:
    """Return the current schema version or reject it."""
    if not isinstance(value, str) or value not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ValueError(
            f"Unsupported evaluation schema version {value!r}; "
            f"supported versions: {supported}"
        )
    return value


def validate_split_algorithm_version(value: object) -> str:
    """Return a supported deterministic split-algorithm version."""
    if not isinstance(value, str) or value not in SUPPORTED_SPLIT_ALGORITHM_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SPLIT_ALGORITHM_VERSIONS))
        raise ValueError(
            f"Unsupported split algorithm version {value!r}; "
            f"supported versions: {supported}"
        )
    return value


def validate_case_payload_version(payload: object) -> dict[str, Any]:
    """Require a case payload using the current schema version."""
    return _validate_payload_version(payload, "case")


def validate_manifest_payload_version(payload: object) -> dict[str, Any]:
    """Require a manifest payload using the current schema version."""
    return _validate_payload_version(payload, "manifest")


def _validate_payload_version(
    payload: object,
    payload_kind: str,
) -> dict[str, Any]:
    """Validate one JSON-object payload without implying migration support."""
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation {payload_kind} payload must be a JSON object")

    validate_schema_version(payload.get("schema_version"))
    return payload
