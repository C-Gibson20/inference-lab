"""Shared strict JSON primitives for retrieval serialisation."""

import json
from collections.abc import Callable
from typing import cast

from inference_lab.json_types import JsonObject, JsonValue
from inference_lab.retrieval.errors import InvalidRetrievalSerialisationError
from inference_lab.retrieval.versioning import validate_retrieval_schema_version


def canonical_json_bytes(payload: JsonObject) -> bytes:
    """Encode one canonical newline-terminated JSON object."""
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    """Build one JSON object while rejecting duplicate keys."""
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise InvalidRetrievalSerialisationError(
                "retrieval JSON must not contain duplicate object keys."
            )
        result[key] = value
    return result


def _reject_constant(_: str) -> JsonValue:
    """Reject non-finite JSON constants."""
    raise InvalidRetrievalSerialisationError(
        "retrieval JSON must not contain non-finite values."
    )


def load_json(data: bytes) -> JsonObject:
    """Decode strict UTF-8 JSON without exposing rejected payload content."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise InvalidRetrievalSerialisationError(
            "retrieval data must be valid UTF-8 JSON."
        ) from None
    if not isinstance(payload, dict):
        raise InvalidRetrievalSerialisationError(
            "retrieval JSON payload must be an object."
        )
    return cast(JsonObject, payload)


def require_object(
    value: JsonValue,
    *,
    name: str,
    keys: frozenset[str],
) -> JsonObject:
    """Require exactly the fields belonging to one decoded object."""
    if not isinstance(value, dict):
        raise InvalidRetrievalSerialisationError(f"{name} must be an object.")
    if frozenset(value) != keys:
        raise InvalidRetrievalSerialisationError(
            f"{name} must contain exactly the canonical fields."
        )
    return cast(JsonObject, value)


def require_array(value: JsonValue, *, name: str) -> list[JsonValue]:
    """Require one JSON array."""
    if not isinstance(value, list):
        raise InvalidRetrievalSerialisationError(f"{name} must be an array.")
    return value


def require_string(value: JsonValue, *, name: str) -> str:
    """Require one JSON string."""
    if not isinstance(value, str):
        raise InvalidRetrievalSerialisationError(f"{name} must be a string.")
    return value


def require_boolean(value: JsonValue, *, name: str) -> bool:
    """Require one JSON boolean."""
    if not isinstance(value, bool):
        raise InvalidRetrievalSerialisationError(f"{name} must be a boolean.")
    return value


def require_integer(value: JsonValue, *, name: str) -> int:
    """Require one strict JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRetrievalSerialisationError(f"{name} must be an integer.")
    return value


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
