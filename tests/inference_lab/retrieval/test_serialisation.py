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
    """Encode tagged scalars deterministically and recover exact values."""
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
        b'"kind":"all"},"limit":3,"retrieval_schema_version":"1.0.0",'
        b'"text":"  assay date?  "}\n'
    )
    assert serialise_query(query) == expected
    assert deserialise_query(expected) == query


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
        (hit,),
        ResultStatus.PARTIAL,
        (RetrievalFailure(FailureCategory.CHANNEL_UNAVAILABLE, True),),
    )


def test_evidence_result_round_trips_full_provenance_and_failures() -> None:
    """Round-trip candidates, source links, scores, status, and failures."""
    result = _result_fixture()
    encoded = serialise_evidence_result(result)

    assert b'"document_id"' in encoded
    assert b'"document_version_id"' in encoded
    assert b'"source_span_id"' in encoded
    assert deserialise_evidence_result(encoded) == result


@pytest.mark.parametrize(
    "data",
    [
        b"not json",
        b'\xff{"retrieval_schema_version":"1.0.0"}',
        b'{"retrieval_schema_version":"1.0.0","retrieval_schema_version":"1.0.0"}',
        b'{"retrieval_schema_version":"1.0.0","text":"x"}',
        b'{"retrieval_schema_version":"1.0.0","text":"x","corpus_version":"00000000-0000-4000-8000-000000000001","limit":1,"filter":null,"extra":true}',
        b'{"retrieval_schema_version":"1.0.0","text":"x","corpus_version":"00000000-0000-4000-8000-000000000001","limit":1.0,"filter":null}',
        b'{"retrieval_schema_version":"1.0.0","text":"x","corpus_version":"00000000-0000-4000-8000-000000000001","limit":1,"filter":{"kind":"exact","field":"x","value":{"type":"integer","value":true}}}',
    ],
)
def test_query_deserialisation_rejects_non_canonical_or_malformed_payloads(
    data: bytes,
) -> None:
    """Reject malformed JSON, duplicate/unknown fields, and coercion."""
    with pytest.raises((InvalidRetrievalSerialisationError, TypeError)):
        deserialise_query(data)


def test_deserialisation_rejects_unsupported_and_non_finite_versions() -> None:
    """Reject unsupported schema versions and non-finite numeric values."""
    with pytest.raises(UnsupportedRetrievalSchemaVersionError):
        deserialise_query(
            b'{"retrieval_schema_version":"9.0.0","text":"x","corpus_version":"00000000-0000-4000-8000-000000000001","limit":1,"filter":null}'
        )
    with pytest.raises(InvalidRetrievalSerialisationError):
        deserialise_query(
            b'{"retrieval_schema_version":"1.0.0","text":"x","corpus_version":"00000000-0000-4000-8000-000000000001","limit":1,"filter":{"kind":"exact","field":"x","value":{"type":"float","value":NaN}}}'
        )


def test_serialisation_rejection_does_not_echo_private_payload_or_log() -> None:
    """Keep query and candidate content out of errors and operational logs."""
    marker = "private-query-and-candidate-marker"
    data = json.dumps(
        {
            "retrieval_schema_version": "1.0.0",
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
