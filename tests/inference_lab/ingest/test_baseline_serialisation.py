"""Phase-level contract for deterministic, payload-free ingest summaries."""

from inference_lab.ingest import (
    BaselineExecutionStatus,
    BaselineFailure,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineIngestSummary,
    BaselinePublicationStatus,
    BaselineStage,
    serialise_baseline_ingest_summary,
)
from inference_lab.provenance import CorpusId, CorpusVersionId


def test_failed_ingest_summary_serialises_deterministically_without_payloads() -> None:
    summary = BaselineIngestSummary(
        corpus_id=CorpusId("11111111-1111-4111-8111-111111111111"),
        proposed_corpus_version=CorpusVersionId("33333333-3333-4333-8333-333333333333"),
        expected_corpus_version=CorpusVersionId("44444444-4444-4444-8444-444444444444"),
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

    assert encoded == serialise_baseline_ingest_summary(summary)
    assert encoded.endswith(b"\n")
    assert b"/private" not in encoded
    assert b"source_text" not in encoded
    assert b"exception" not in encoded
    assert b"raw_message" not in encoded
