"""Retrieval-specific adapters for shared strict JSON primitives."""

from collections.abc import Callable

from inference_lab._json import (
    JsonDecodingError,
    JsonDecodingFailure,
    JsonObject,
    JsonStructureError,
    JsonValue,
    load_json_object,
    require_exact_json_fields,
    require_json_array,
    require_json_boolean,
    require_json_integer,
    require_json_object,
    require_json_string,
)
from inference_lab.retrieval.errors import InvalidRetrievalSerialisationError
from inference_lab.retrieval.versioning import validate_retrieval_schema_version

_DECODING_MESSAGES = {
    JsonDecodingFailure.INVALID_DATA: "retrieval data must be valid UTF-8 JSON.",
    JsonDecodingFailure.DUPLICATE_KEY: (
        "retrieval JSON must not contain duplicate object keys."
    ),
    JsonDecodingFailure.NON_FINITE_VALUE: (
        "retrieval JSON must not contain non-finite values."
    ),
    JsonDecodingFailure.ROOT_NOT_OBJECT: ("retrieval JSON payload must be an object."),
}


def load_json(data: bytes) -> JsonObject:
    """Decode strict UTF-8 JSON without exposing rejected payload content."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")
    try:
        return load_json_object(data)
    except JsonDecodingError as error:
        raise InvalidRetrievalSerialisationError(
            _DECODING_MESSAGES[error.failure]
        ) from None


def require_object(
    value: JsonValue,
    *,
    name: str,
    keys: frozenset[str] | None = None,
) -> JsonObject:
    """Require an object and, when supplied, its exact fields."""
    try:
        payload = require_json_object(value)
    except JsonStructureError:
        raise InvalidRetrievalSerialisationError(f"{name} must be an object.") from None
    if keys is None:
        return payload
    try:
        require_exact_json_fields(payload, keys)
    except JsonStructureError:
        raise InvalidRetrievalSerialisationError(
            f"{name} must contain exactly the canonical fields."
        ) from None
    return payload


def require_array(value: JsonValue, *, name: str) -> list[JsonValue]:
    """Require one JSON array."""
    try:
        return require_json_array(value)
    except JsonStructureError:
        raise InvalidRetrievalSerialisationError(f"{name} must be an array.") from None


def require_string(value: JsonValue, *, name: str) -> str:
    """Require one JSON string."""
    try:
        return require_json_string(value)
    except JsonStructureError:
        raise InvalidRetrievalSerialisationError(f"{name} must be a string.") from None


def require_boolean(value: JsonValue, *, name: str) -> bool:
    """Require one JSON boolean."""
    try:
        return require_json_boolean(value)
    except JsonStructureError:
        raise InvalidRetrievalSerialisationError(f"{name} must be a boolean.") from None


def require_integer(value: JsonValue, *, name: str) -> int:
    """Require one strict JSON integer."""
    try:
        return require_json_integer(value)
    except JsonStructureError:
        raise InvalidRetrievalSerialisationError(
            f"{name} must be an integer."
        ) from None


def parse_enum(value: JsonValue, *, name: str, enum_type: type[object]) -> object:
    """Parse one closed string enum without leaking rejected values."""
    text = require_string(value, name=name)
    try:
        return enum_type(text)  # type: ignore[call-arg]
    except ValueError:
        raise InvalidRetrievalSerialisationError(f"{name} is not supported.") from None


def parse_identifier(
    value: JsonValue,
    *,
    name: str,
    constructor: Callable[[str], object],
) -> object:
    """Parse one canonical provenance identifier."""
    try:
        return constructor(require_string(value, name=name))
    except (InvalidRetrievalSerialisationError, ValueError, TypeError):
        raise InvalidRetrievalSerialisationError(
            f"{name} must be a canonical provenance identifier."
        ) from None


def validate_schema(payload: JsonObject) -> None:
    """Validate the schema field before interpreting nested data."""
    if "retrieval_schema_version" not in payload:
        raise InvalidRetrievalSerialisationError(
            "retrieval payload requires retrieval_schema_version."
        )
    try:
        validate_retrieval_schema_version(payload["retrieval_schema_version"])
    except TypeError as error:
        raise InvalidRetrievalSerialisationError(str(error)) from None
