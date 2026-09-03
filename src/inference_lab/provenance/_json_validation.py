"""Lineage-specific adapters for shared strict JSON primitives."""

from collections.abc import Callable
from typing import TypeVar

from inference_lab._json import (
    JsonDecodingError,
    JsonDecodingFailure,
    JsonObject,
    JsonStructureError,
    JsonValue,
    load_json_object,
    require_exact_json_fields,
    require_json_array,
    require_json_object,
    require_json_string,
    require_optional_json_integer,
    require_optional_json_string,
)
from inference_lab.provenance.errors import InvalidLineageSerialisationError
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.versioning import validate_lineage_schema_version

_IdentifierT = TypeVar("_IdentifierT", DocumentVersionId, RecordId, SourceSpanId)

_DECODING_MESSAGES = {
    JsonDecodingFailure.INVALID_DATA: "lineage data must be valid UTF-8 JSON.",
    JsonDecodingFailure.DUPLICATE_KEY: (
        "lineage JSON must not contain duplicate object keys."
    ),
    JsonDecodingFailure.NON_FINITE_VALUE: (
        "lineage JSON must not contain non-finite values."
    ),
    JsonDecodingFailure.ROOT_NOT_OBJECT: ("lineage JSON payload must be an object."),
}


def load_json(data: bytes) -> JsonObject:
    """Decode strict UTF-8 JSON without exposing rejected payload content."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")
    try:
        return load_json_object(data)
    except JsonDecodingError as error:
        raise InvalidLineageSerialisationError(
            _DECODING_MESSAGES[error.failure]
        ) from None


def require_object(
    value: JsonValue,
    *,
    field_name: str,
    keys: frozenset[str] | None = None,
) -> JsonObject:
    """Require an object and, when supplied, its exact canonical fields."""
    try:
        payload = require_json_object(value)
    except JsonStructureError:
        raise InvalidLineageSerialisationError(
            f"{field_name} must be an object."
        ) from None
    if keys is None:
        return payload
    try:
        require_exact_json_fields(payload, keys)
    except JsonStructureError:
        raise InvalidLineageSerialisationError(
            f"{field_name} must contain exactly the canonical fields."
        ) from None
    return payload


def require_array(value: JsonValue, *, field_name: str) -> list[JsonValue]:
    """Require one JSON array without implicit container coercion."""
    try:
        return require_json_array(value)
    except JsonStructureError:
        raise InvalidLineageSerialisationError(
            f"{field_name} must be an array."
        ) from None


def require_string(value: JsonValue, *, field_name: str) -> str:
    """Require one JSON string without implicit scalar coercion."""
    try:
        return require_json_string(value)
    except JsonStructureError:
        raise InvalidLineageSerialisationError(
            f"{field_name} must be a string."
        ) from None


def require_optional_string(value: JsonValue, *, field_name: str) -> str | None:
    """Require a JSON string or null."""
    try:
        return require_optional_json_string(value)
    except JsonStructureError:
        raise InvalidLineageSerialisationError(
            f"{field_name} must be a string or null."
        ) from None


def require_optional_integer(value: JsonValue, *, field_name: str) -> int | None:
    """Require a strict JSON integer or null."""
    try:
        return require_optional_json_integer(value)
    except JsonStructureError:
        raise InvalidLineageSerialisationError(
            f"{field_name} must be an integer or null."
        ) from None


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
