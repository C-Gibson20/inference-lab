"""Negative and shape-complete tests for retrieval query serialisation."""

import json

import pytest

from inference_lab.provenance import CorpusVersionId
from inference_lab.retrieval import (
    AnyFilter,
    ExactFilter,
    ExistsFilter,
    FilterExpression,
    InvalidRetrievalSerialisationError,
    MissingFilter,
    NotFilter,
    RangeFilter,
    RetrievalLimit,
    RetrievalQuery,
    deserialise_query,
    serialise_query,
)

_CORPUS_VERSION = "00000000-0000-4000-8000-000000000001"


def _query_data(filter_payload: object) -> bytes:
    """Encode a canonical query envelope around one raw filter payload."""
    return json.dumps(
        {
            "retrieval_schema_version": "2.0.0",
            "text": "question",
            "corpus_version": _CORPUS_VERSION,
            "limit": 5,
            "filter": filter_payload,
        }
    ).encode()


@pytest.mark.parametrize(
    "expression",
    [
        ExactFilter("integer", 3),
        ExactFilter("float", 1.5),
        ExactFilter("string", "value"),
        RangeFilter("year", 2020, 2025, False, True),
        RangeFilter("start-year", 2020, None),
        RangeFilter("end-year", None, 2025),
        ExistsFilter("doi"),
        MissingFilter("withdrawn-at"),
        AnyFilter((ExactFilter("kind", "paper"), ExactFilter("kind", "patent"))),
        NotFilter(ExistsFilter("retracted-at")),
    ],
)
def test_query_serialisation_round_trips_every_filter_shape(
    expression: FilterExpression,
) -> None:
    query = RetrievalQuery(
        "question",
        CorpusVersionId.from_str(_CORPUS_VERSION),
        RetrievalLimit(5),
        expression,
    )

    assert deserialise_query(serialise_query(query)) == query


def test_query_serialisation_rejects_wrong_envelope_type() -> None:
    with pytest.raises(TypeError, match="RetrievalQuery"):
        serialise_query("question")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes"):
        deserialise_query("question")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("filter_payload", "message"),
    [
        ({"kind": "unknown"}, "filter.kind"),
        (
            {
                "kind": "exact",
                "field": "date",
                "value": {"type": "date", "value": "not-a-date"},
            },
            "date",
        ),
        (
            {
                "kind": "exact",
                "field": "date",
                "value": {"type": "date", "value": "20240102"},
            },
            "canonical",
        ),
        (
            {
                "kind": "exact",
                "field": "score",
                "value": {"type": "float", "value": 1},
            },
            "float",
        ),
        (
            {
                "kind": "exact",
                "field": "score",
                "value": {"type": "unknown", "value": "x"},
            },
            "scalar.type",
        ),
        ({"kind": "one_of", "field": "kind", "values": "paper"}, "array"),
        (
            {
                "kind": "range",
                "field": "year",
                "lower": None,
                "upper": {"type": "integer", "value": 2025},
                "lower_inclusive": 1,
                "upper_inclusive": True,
            },
            "boolean",
        ),
        ({"kind": "exists", "field": 1}, "string"),
        ({"kind": "missing", "field": 1}, "string"),
        ({"kind": "any", "children": "not-an-array"}, "array"),
    ],
)
def test_query_deserialisation_rejects_invalid_filter_values(
    filter_payload: object,
    message: str,
) -> None:
    with pytest.raises(InvalidRetrievalSerialisationError, match=message):
        deserialise_query(_query_data(filter_payload))


@pytest.mark.parametrize(
    "data",
    [
        b"[]",
        b"{}",
        b'{"retrieval_schema_version":true}',
        _query_data([]),
        _query_data({"kind": "exact", "field": "x", "value": []}),
    ],
)
def test_query_deserialisation_rejects_wrong_json_structure(data: bytes) -> None:
    with pytest.raises(InvalidRetrievalSerialisationError):
        deserialise_query(data)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("text", " "), ("limit", 0)],
)
def test_query_deserialisation_translates_contract_failures(
    field_name: str,
    value: object,
) -> None:
    payload = json.loads(_query_data(None))
    payload[field_name] = value

    with pytest.raises(InvalidRetrievalSerialisationError, match="retrieval contract"):
        deserialise_query(json.dumps(payload).encode())
