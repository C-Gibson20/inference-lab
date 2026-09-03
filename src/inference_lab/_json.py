"""Shared strict JSON types, encoding, and structural primitives."""

import json
from enum import StrEnum
from typing import cast

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class JsonDecodingFailure(StrEnum):
    """Payload-private reasons why strict JSON object decoding failed."""

    INVALID_DATA = "invalid_data"
    DUPLICATE_KEY = "duplicate_key"
    NON_FINITE_VALUE = "non_finite_value"
    ROOT_NOT_OBJECT = "root_not_object"


class JsonDecodingError(ValueError):
    """Report a strict JSON decoding failure without retaining input data."""

    def __init__(self, failure: JsonDecodingFailure) -> None:
        super().__init__(failure.value)
        self.failure = failure


class JsonStructureError(ValueError):
    """Report that a decoded value does not have a required JSON shape."""


class JsonFieldsError(JsonStructureError):
    """Report missing or unknown fields without retaining their values."""

    def __init__(
        self,
        *,
        missing: frozenset[str],
        unknown: frozenset[str],
    ) -> None:
        super().__init__("JSON object fields do not match")
        self.missing = missing
        self.unknown = unknown


def canonical_json_text(value: JsonValue) -> str:
    """Encode one value as compact canonical JSON text."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(payload: JsonObject) -> bytes:
    """Encode one object as canonical UTF-8 JSON with a trailing newline."""
    return f"{canonical_json_text(payload)}\n".encode("utf-8")


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    """Build an object while rejecting duplicate member names."""
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise JsonDecodingError(JsonDecodingFailure.DUPLICATE_KEY)
        result[key] = value
    return result


def _reject_constant(_: str) -> JsonValue:
    """Reject non-finite constants accepted by Python's JSON extension."""
    raise JsonDecodingError(JsonDecodingFailure.NON_FINITE_VALUE)


def load_json_object(data: bytes) -> JsonObject:
    """Decode strict UTF-8 JSON whose root is an object."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise JsonDecodingError(JsonDecodingFailure.INVALID_DATA) from None
    if not isinstance(value, dict):
        raise JsonDecodingError(JsonDecodingFailure.ROOT_NOT_OBJECT)
    return cast(JsonObject, value)


def require_json_object(value: JsonValue) -> JsonObject:
    """Require one decoded JSON object."""
    if not isinstance(value, dict):
        raise JsonStructureError("value must be an object")
    return value


def require_exact_json_fields(
    payload: JsonObject,
    expected: frozenset[str],
) -> None:
    """Require exactly the expected fields in one decoded object."""
    actual = frozenset(payload)
    if actual != expected:
        raise JsonFieldsError(
            missing=expected - actual,
            unknown=actual - expected,
        )


def require_json_array(value: JsonValue) -> list[JsonValue]:
    """Require one decoded JSON array."""
    if not isinstance(value, list):
        raise JsonStructureError("value must be an array")
    return value


def require_json_string(value: JsonValue) -> str:
    """Require one decoded JSON string."""
    if not isinstance(value, str):
        raise JsonStructureError("value must be a string")
    return value


def require_optional_json_string(value: JsonValue) -> str | None:
    """Require one decoded JSON string or null."""
    if value is not None and not isinstance(value, str):
        raise JsonStructureError("value must be a string or null")
    return value


def require_json_integer(value: JsonValue) -> int:
    """Require one decoded JSON integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise JsonStructureError("value must be an integer")
    return value


def require_optional_json_integer(value: JsonValue) -> int | None:
    """Require one decoded JSON integer or null, excluding booleans."""
    if value is not None:
        return require_json_integer(value)
    return None


def require_json_boolean(value: JsonValue) -> bool:
    """Require one decoded JSON boolean."""
    if not isinstance(value, bool):
        raise JsonStructureError("value must be a boolean")
    return value
