"""Evaluation schema and dataset version policy."""

from typing import Any, Final

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


def migrate_case_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate a case payload to the current schema version."""

    return _migrate_payload(payload, "case")


def migrate_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate a manifest payload to the current schema version."""

    return _migrate_payload(payload, "manifest")


def _migrate_payload(payload: dict[str, Any], payload_kind: str) -> dict[str, Any]:
    """Apply the registered migration for one payload kind."""

    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation {payload_kind} payload must be a JSON object")

    version = payload.get("schema_version")

    if version == CURRENT_SCHEMA_VERSION:
        return payload

    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported evaluation schema version: {version!r}")

    raise NotImplementedError(
        f"No migration is registered for {payload_kind} schema version {version!r}"
    )
