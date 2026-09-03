"""Tests for strict retrieval serialisation."""

import json
from datetime import date

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import (
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    AllFilter,
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    ExactFilter,
    FailureCategory,
    InvalidRetrievalSerialisationError,
    OneOfFilter,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalFailure,
    RetrievalLimit,
    RetrievalQuery,
    UnsupportedRetrievalSchemaVersionError,
    deserialise_evidence_result,
    deserialise_query,
    serialise_evidence_result,
    serialise_query,
)


def test_query_serialisation_is_canonical_and_round_trips() -> None:
    query = RetrievalQuery(
        "  assay date?  ",
        CorpusVersionId.from_str("00000000-0000-4000-8000-000000000001"),
        RetrievalLimit(3),
        AllFilter(
            (
                # The nested shape makes the canonical ordering visible in the
                # golden bytes while retaining the caller's child order.
                ExactFilter("is-public", True),
                ExactFilter("priority-date", date(2024, 1, 2)),
            )
        ),
    )
    expected = (
        b'{"corpus_version":"00000000-0000-4000-8000-000000000001",'
        b'"filter":{"children":[{"field":"is-public","kind":"exact",'
        b'"value":{"type":"boolean","value":true}},{"field":"priority-date",'
        b'"kind":"exact","value":{"type":"date","value":"2024-01-02"}}],'
        b'"kind":"all"},"limit":3,"retrieval_schema_version":"2.0.0",'
        b'"text":"  assay date?  "}\n'
    )
    assert serialise_query(query) == expected
    assert deserialise_query(expected) == query


def test_set_like_filters_have_order_independent_canonical_bytes() -> None:
    version = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000001")
    first = RetrievalQuery(
        "question",
        version,
        RetrievalLimit(3),
        OneOfFilter("document-type", ("paper", "patent")),
    )
    second = RetrievalQuery(
        "question",
        version,
        RetrievalLimit(3),
        OneOfFilter("document-type", ("patent", "paper")),
    )

    assert first == second
    assert serialise_query(first) == serialise_query(second)


def _result_fixture() -> EvidenceResult:
    """Return a representative result with distinct score semantics."""
    source = EvidenceSource(
        DocumentId.from_str("00000000-0000-4000-8000-000000000010"),
        DocumentVersionId.from_str("00000000-0000-4000-8000-000000000011"),
        SourceSpanId.from_str("00000000-0000-4000-8000-000000000012"),
    )
    candidate = RetrievalCandidate(
        RecordId.from_str("00000000-0000-4000-8000-000000000013"),
        CandidateKind.ORIGINAL_PASSAGE,
        "private source passage",
        (source,),
    )
    hit = EvidenceHit(
        candidate,
        (
            ChannelScore(RetrievalChannel.LEXICAL, 1, 4.25, "bm25"),
            ChannelScore(RetrievalChannel.DENSE, 2, None, None),
        ),
        1,
    )
    return EvidenceResult(
        CorpusVersionId.from_str("00000000-0000-4000-8000-000000000014"),
        RetrievalLimit(1),
        (hit,),
        ResultStatus.PARTIAL,
        (RetrievalFailure(FailureCategory.CHANNEL_UNAVAILABLE, True),),
    )


def _replace_result_path(path: tuple[str | int, ...], value: object) -> bytes:
    """Replace one nested result value and encode the resulting payload."""
    payload = json.loads(serialise_evidence_result(_result_fixture()))
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    return json.dumps(payload).encode()


def test_evidence_result_round_trips_full_provenance_and_failures() -> None:
    result = _result_fixture()
    encoded = serialise_evidence_result(result)

    assert b'"document_id"' in encoded
    assert b'"document_version_id"' in encoded
    assert b'"source_span_id"' in encoded
    assert json.loads(encoded)["requested_limit"] == result.requested_limit.value
    assert deserialise_evidence_result(encoded) == result


def test_evidence_result_serialisation_rejects_wrong_envelope_type() -> None:
    with pytest.raises(TypeError, match="EvidenceResult"):
        serialise_evidence_result("result")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("hits", 0, "candidate", "kind"), "unknown", "candidate.kind"),
        (("hits", 0, "candidate", "content"), 1, "candidate.content"),
        (("hits", 0, "candidate", "sources"), {}, "candidate.sources"),
        (("hits", 0, "contributions", 0, "channel"), "unknown", "score.channel"),
        (("hits", 0, "contributions", 0, "raw_score"), True, "raw_score"),
        (("hits", 0, "contributions", 0, "raw_score"), "4.25", "raw_score"),
        (("hits", 0, "contributions", 0, "score_label"), 1, "score_label"),
        (("hits", 0, "contributions"), {}, "hit.contributions"),
        (("failures", 0, "category"), "unknown", "failure.category"),
        (("failures", 0, "channel"), 1, "failure.channel"),
        (("failures", 0, "retryable"), 1, "failure.retryable"),
        (("hits",), {}, "result.hits"),
        (("failures",), {}, "result.failures"),
        (("status",), "unknown", "result.status"),
    ],
)
def test_result_deserialisation_rejects_invalid_nested_values(
    path: tuple[str | int, ...],
    value: object,
    message: str,
) -> None:
    with pytest.raises(InvalidRetrievalSerialisationError, match=message):
        deserialise_evidence_result(_replace_result_path(path, value))


@pytest.mark.parametrize(
    ("identifier_path", "value"),
    [
        (("corpus_version",), "not-an-id"),
        (("hits", 0, "candidate", "record_id"), "not-an-id"),
        (("hits", 0, "candidate", "sources", 0, "document_id"), "not-an-id"),
        (
            ("hits", 0, "candidate", "sources", 0, "document_version_id"),
            "not-an-id",
        ),
        (("hits", 0, "candidate", "sources", 0, "source_span_id"), "not-an-id"),
    ],
)
def test_result_deserialisation_rejects_noncanonical_identifiers(
    identifier_path: tuple[str | int, ...],
    value: str,
) -> None:
    with pytest.raises(InvalidRetrievalSerialisationError, match="identifier"):
        deserialise_evidence_result(_replace_result_path(identifier_path, value))


def test_result_deserialisation_rejects_limit_and_mismatch_contradictions() -> None:
    compatible_payload = json.loads(serialise_evidence_result(_result_fixture()))
    compatible_payload["failures"][0] = {
        "category": "corpus_version_mismatch",
        "channel": "metadata",
        "retryable": False,
    }
    compatible = deserialise_evidence_result(json.dumps(compatible_payload).encode())
    assert compatible.failures[0].channel is RetrievalChannel.METADATA

    payload = json.loads(serialise_evidence_result(_result_fixture()))
    second_hit = json.loads(json.dumps(payload["hits"][0]))
    second_hit["rank"] = 2
    second_hit["candidate"]["record_id"] = "00000000-0000-4000-8000-000000000099"
    payload["hits"].append(second_hit)
    with pytest.raises(InvalidRetrievalSerialisationError):
        deserialise_evidence_result(json.dumps(payload).encode())

    for channel in ("lexical", None):
        payload = json.loads(serialise_evidence_result(_result_fixture()))
        payload["failures"][0] = {
            "category": "corpus_version_mismatch",
            "channel": channel,
            "retryable": False,
        }
        with pytest.raises(InvalidRetrievalSerialisationError):
            deserialise_evidence_result(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "data",
    [
        b"not json",
        b'\xff{"retrieval_schema_version":"2.0.0"}',
        b'{"retrieval_schema_version":"2.0.0","retrieval_schema_version":"2.0.0"}',
        b'{"retrieval_schema_version":"2.0.0","text":"x"}',
        b'{"retrieval_schema_version":"2.0.0","text":"x","corpus_version":"00000000-0000-4000-8000-000000000001","limit":1,"filter":null,"extra":true}',
        b'{"retrieval_schema_version":"2.0.0","text":"x","corpus_version":"00000000-0000-4000-8000-000000000001","limit":1.0,"filter":null}',
        b'{"retrieval_schema_version":"2.0.0","text":"x","corpus_version":"00000000-0000-4000-8000-000000000001","limit":1,"filter":{"kind":"exact","field":"x","value":{"type":"integer","value":true}}}',
    ],
)
def test_query_deserialisation_rejects_non_canonical_or_malformed_payloads(
    data: bytes,
) -> None:
    with pytest.raises((InvalidRetrievalSerialisationError, TypeError)):
        deserialise_query(data)


@pytest.mark.parametrize(
    "filter_payload",
    [
        {},
        {"field": "document-type"},
        {"kind": "all", "children": [{}]},
        {"kind": "not", "child": {}},
    ],
)
def test_nested_filters_without_discriminators_use_serialisation_error(
    filter_payload: object,
) -> None:
    data = json.dumps(
        {
            "retrieval_schema_version": "2.0.0",
            "text": "question",
            "corpus_version": "00000000-0000-4000-8000-000000000001",
            "limit": 1,
            "filter": filter_payload,
        }
    ).encode()

    with pytest.raises(InvalidRetrievalSerialisationError, match="filter"):
        deserialise_query(data)


@pytest.mark.parametrize("version", ["1.0.0", "9.0.0"])
def test_deserialisation_rejects_unsupported_and_non_finite_versions(
    version: str,
) -> None:
    with pytest.raises(UnsupportedRetrievalSchemaVersionError):
        deserialise_query(
            json.dumps(
                {
                    "retrieval_schema_version": version,
                    "text": "x",
                    "corpus_version": "00000000-0000-4000-8000-000000000001",
                    "limit": 1,
                    "filter": None,
                }
            ).encode()
        )
    with pytest.raises(InvalidRetrievalSerialisationError):
        deserialise_query(
            b'{"retrieval_schema_version":"2.0.0","text":"x","corpus_version":"00000000-0000-4000-8000-000000000001","limit":1,"filter":{"kind":"exact","field":"x","value":{"type":"float","value":NaN}}}'
        )


def test_serialisation_rejection_does_not_echo_private_payload_or_log() -> None:
    marker = "private-query-and-candidate-marker"
    data = json.dumps(
        {
            "retrieval_schema_version": "2.0.0",
            "text": marker,
            "corpus_version": "not-an-id",
            "limit": 1,
            "filter": None,
        }
    ).encode()
    with capture_logs() as logs:
        with pytest.raises(InvalidRetrievalSerialisationError) as error:
            deserialise_query(data)
    assert marker not in str(error.value)
    assert logs == []
