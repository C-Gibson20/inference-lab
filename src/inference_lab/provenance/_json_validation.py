"""Strict JSON validation primitives for lineage serialisation."""

import json
from collections.abc import Callable
from typing import TypeVar, cast

from inference_lab.json_types import JsonObject, JsonValue
from inference_lab.provenance.errors import InvalidLineageSerialisationError
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.versioning import validate_lineage_schema_version

_IdentifierT = TypeVar("_IdentifierT", DocumentVersionId, RecordId, SourceSpanId)


def reject_json_constant(_: str) -> JsonValue:
    """Reject non-finite constants accepted by Python's JSON extension."""
    raise InvalidLineageSerialisationError(
        "lineage JSON must not contain non-finite values."
    )


def unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    """Build one JSON object while rejecting duplicate member names."""
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise InvalidLineageSerialisationError(
                "lineage JSON must not contain duplicate object keys."
            )
        result[key] = value
    return result


def load_json(data: bytes) -> JsonObject:
    """Decode strict UTF-8 JSON without exposing rejected payload content."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")
    try:
        text = data.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise InvalidLineageSerialisationError(
            "lineage data must be valid UTF-8 JSON."
        ) from None
    if not isinstance(payload, dict):
        raise InvalidLineageSerialisationError(
            "lineage JSON payload must be an object."
        )
    return cast(JsonObject, payload)


def require_object(
    value: JsonValue,
    *,
    field_name: str,
    keys: frozenset[str],
) -> JsonObject:
    """Require one object containing exactly the canonical fields."""
    if not isinstance(value, dict):
        raise InvalidLineageSerialisationError(f"{field_name} must be an object.")
    if frozenset(value) != keys:
        raise InvalidLineageSerialisationError(
            f"{field_name} must contain exactly the canonical fields."
        )
    return cast(JsonObject, value)


def require_array(value: JsonValue, *, field_name: str) -> list[JsonValue]:
    """Require one JSON array without implicit container coercion."""
    if not isinstance(value, list):
        raise InvalidLineageSerialisationError(f"{field_name} must be an array.")
    return value


def require_string(value: JsonValue, *, field_name: str) -> str:
    """Require one JSON string without implicit scalar coercion."""
    if not isinstance(value, str):
        raise InvalidLineageSerialisationError(f"{field_name} must be a string.")
    return value


def require_optional_string(value: JsonValue, *, field_name: str) -> str | None:
    """Require a JSON string or null."""
    if value is not None and not isinstance(value, str):
        raise InvalidLineageSerialisationError(
            f"{field_name} must be a string or null."
        )
    return value


def require_optional_integer(value: JsonValue, *, field_name: str) -> int | None:
    """Require a strict JSON integer or null."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise InvalidLineageSerialisationError(
            f"{field_name} must be an integer or null."
        )
    return value


def require_schema_version(payload: JsonObject) -> str:
    """Validate schema compatibility before deeper payload parsing."""
    if "lineage_schema_version" not in payload:
        raise InvalidLineageSerialisationError(
            "lineage payload requires lineage_schema_version."
        )
    value = payload["lineage_schema_version"]
    if not isinstance(value, str):
        raise TypeError("lineage_schema_version must be a string.")
    return validate_lineage_schema_version(value)


def parse_identifier(
    value: JsonValue,
    *,
    field_name: str,
    identifier_type: Callable[[str], _IdentifierT],
) -> _IdentifierT:
    """Rebuild one domain-specific identifier from canonical JSON text."""
    text = require_string(value, field_name=field_name)
    try:
        return identifier_type(text)
    except ValueError:
        raise InvalidLineageSerialisationError(
            f"{field_name} must be a canonical provenance identifier."
        ) from None
