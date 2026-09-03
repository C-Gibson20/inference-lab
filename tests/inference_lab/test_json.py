"""Tests for shared strict JSON mechanics."""

from collections.abc import Callable

import pytest

from inference_lab._json import (
    JsonDecodingError,
    JsonDecodingFailure,
    JsonFieldsError,
    JsonObject,
    JsonStructureError,
    JsonValue,
    canonical_json_bytes,
    canonical_json_text,
    load_json_object,
    require_exact_json_fields,
    require_json_array,
    require_json_boolean,
    require_json_integer,
    require_json_object,
    require_json_string,
    require_optional_json_integer,
    require_optional_json_string,
)

type _StructureRequirement = Callable[[JsonValue], object]


def test_canonical_json_is_compact_sorted_unicode_with_optional_newline() -> None:
    payload: JsonObject = {"z": 1, "a": "α"}

    assert canonical_json_text(payload) == '{"a":"α","z":1}'
    assert canonical_json_bytes(payload) == b'{"a":"\xce\xb1","z":1}\n'


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json_text({"value": float("nan")})


@pytest.mark.parametrize(
    ("data", "failure"),
    [
        (b"{", JsonDecodingFailure.INVALID_DATA),
        (b"\xff", JsonDecodingFailure.INVALID_DATA),
        (b"[]", JsonDecodingFailure.ROOT_NOT_OBJECT),
        (b'{"value":NaN}', JsonDecodingFailure.NON_FINITE_VALUE),
        (b'{"value":1,"value":2}', JsonDecodingFailure.DUPLICATE_KEY),
    ],
)
def test_strict_object_decoding_reports_payload_private_failure(
    data: bytes,
    failure: JsonDecodingFailure,
) -> None:
    with pytest.raises(JsonDecodingError) as caught:
        load_json_object(data)

    assert caught.value.failure is failure
    assert str(caught.value) == failure.value


def test_strict_object_decoding_accepts_canonical_data() -> None:
    assert load_json_object(b'{"items":[1,true,null]}') == {"items": [1, True, None]}
    with pytest.raises(TypeError, match="data must be bytes"):
        load_json_object("{}")  # type: ignore[arg-type]


def test_structural_primitives_return_strict_json_types() -> None:
    payload: JsonObject = {"value": 1}

    assert require_json_object(payload) is payload
    assert require_json_array([1]) == [1]
    assert require_json_string("value") == "value"
    assert require_optional_json_string(None) is None
    assert require_optional_json_string("value") == "value"
    assert require_json_integer(1) == 1
    assert require_optional_json_integer(None) is None
    assert require_optional_json_integer(1) == 1
    assert require_json_boolean(True) is True
    require_exact_json_fields(payload, frozenset({"value"}))


@pytest.mark.parametrize(
    ("require", "value"),
    [
        (require_json_object, []),
        (require_json_array, {}),
        (require_json_string, 1),
        (require_optional_json_string, 1),
        (require_json_integer, True),
        (require_optional_json_integer, True),
        (require_json_boolean, 1),
    ],
)
def test_structural_primitives_reject_implicit_coercion(
    require: _StructureRequirement,
    value: JsonValue,
) -> None:
    with pytest.raises(JsonStructureError):
        require(value)


@pytest.mark.parametrize(
    ("expected", "missing", "unknown"),
    [
        (frozenset({"value", "missing"}), frozenset({"missing"}), frozenset()),
        (frozenset(), frozenset(), frozenset({"value"})),
    ],
)
def test_exact_fields_report_names_without_values(
    expected: frozenset[str],
    missing: frozenset[str],
    unknown: frozenset[str],
) -> None:
    payload: JsonObject = {"value": 1}
    with pytest.raises(JsonFieldsError) as caught:
        require_exact_json_fields(payload, expected)

    assert caught.value.missing == missing
    assert caught.value.unknown == unknown
