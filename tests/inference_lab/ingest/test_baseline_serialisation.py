"""Tests for the baseline serialisation module."""

import pytest

from inference_lab import ingest
from inference_lab.ingest.baseline_contracts import (
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineExecutionStatus,
    BaselineFailure,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineIngestSummary,
    BaselinePublicationStatus,
    BaselineStage,
    BaselineStageCounts,
    BaselineStageDisposition,
)
from inference_lab.ingest.baseline_serialisation import (
    serialise_baseline_ingest_summary,
)
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_PROPOSED_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_EXPECTED_VERSION = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_DOCUMENT_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")


def _unchanged_outcome() -> BaselineDocumentOutcome:
    return BaselineDocumentOutcome(
        ordinal=0,
        kind=BaselineDocumentOutcomeKind.UNCHANGED_REUSED,
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        parse_disposition=BaselineStageDisposition.REUSED,
        chunk_disposition=BaselineStageDisposition.REUSED,
    )


def test_summary_serialisation_is_byte_identical() -> None:
    summary = BaselineIngestSummary(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=_EXPECTED_VERSION,
        execution_status=BaselineExecutionStatus.COMPLETED,
        publication_status=BaselinePublicationStatus.COMMITTED,
        document_outcomes=(_unchanged_outcome(),),
        stage_counts=(
            BaselineStageCounts(
                stage=BaselineStage.PARSE,
                reused=1,
            ),
            BaselineStageCounts(
                stage=BaselineStage.CHUNK,
                reused=1,
            ),
        ),
    )

    first = serialise_baseline_ingest_summary(summary)
    second = serialise_baseline_ingest_summary(summary)

    assert first == second
    assert first.endswith(b"\n")


def test_summary_serialisation_contains_only_safe_fields() -> None:
    summary = BaselineIngestSummary(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=None,
        execution_status=BaselineExecutionStatus.FAILED,
        publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
        failures=(
            BaselineFailure(
                stage=BaselineStage.DISCOVERY,
                category=BaselineFailureCategory.DISCOVERY_FAILED,
                scope=BaselineFailureScope.REQUEST,
                severity=BaselineFailureSeverity.FATAL,
                retryable=False,
            ),
        ),
    )

    encoded = serialise_baseline_ingest_summary(summary)

    assert b"/private" not in encoded
    assert b"source_text" not in encoded
    assert b"exception" not in encoded
    assert b"raw_message" not in encoded


def test_summary_serialisation_rejects_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="BaselineIngestSummary"):
        serialise_baseline_ingest_summary(None)  # type: ignore[arg-type]


def test_baseline_contracts_are_public_ingest_api() -> None:

    public_name = "serialise_baseline_ingest_summary"

    assert public_name in ingest.__all__
    assert hasattr(ingest, public_name)


def test_public_serialisation_matches_payload_encoding() -> None:
    from inference_lab._json import canonical_json_bytes

    summary = BaselineIngestSummary(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=None,
        execution_status=BaselineExecutionStatus.FAILED,
        publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
        failures=(
            BaselineFailure(
                stage=BaselineStage.DISCOVERY,
                category=BaselineFailureCategory.DISCOVERY_FAILED,
                scope=BaselineFailureScope.REQUEST,
                severity=BaselineFailureSeverity.FATAL,
                retryable=False,
            ),
        ),
    )

    payload = ingest.baseline_ingest_summary_payload(summary)
    encoded = ingest.serialise_baseline_ingest_summary(summary)

    assert encoded == canonical_json_bytes(payload)
