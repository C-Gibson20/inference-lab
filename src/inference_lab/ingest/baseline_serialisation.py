"""Canonical serialisation for baseline-ingest summaries."""

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.ingest.baseline_contracts import (
    BaselineDocumentOutcome,
    BaselineFailure,
    BaselineIngestSummary,
    BaselineStageCounts,
)


def _failure_payload(failure: BaselineFailure) -> JsonObject:
    """Convert one bounded failure to canonical JSON-compatible data."""
    if not isinstance(failure, BaselineFailure):
        raise TypeError("failure must be a BaselineFailure.")

    return {
        "stage": failure.stage.value,
        "category": failure.category.value,
        "scope": failure.scope.value,
        "severity": failure.severity.value,
        "retryable": failure.retryable,
    }


def _document_outcome_payload(
    outcome: BaselineDocumentOutcome,
) -> JsonObject:
    """Convert one document outcome to canonical JSON-compatible data."""
    if not isinstance(outcome, BaselineDocumentOutcome):
        raise TypeError("outcome must be a BaselineDocumentOutcome.")

    return {
        "ordinal": outcome.ordinal,
        "kind": outcome.kind.value,
        "document_id": (
            outcome.document_id.as_str() if outcome.document_id is not None else None
        ),
        "document_version_id": (
            outcome.document_version_id.as_str()
            if outcome.document_version_id is not None
            else None
        ),
        "parse_disposition": outcome.parse_disposition.value,
        "chunk_disposition": outcome.chunk_disposition.value,
        "failures": [_failure_payload(failure) for failure in outcome.failures],
    }


def _stage_counts_payload(
    counts: BaselineStageCounts,
) -> JsonObject:
    """Convert stage counters to canonical JSON-compatible data."""
    if not isinstance(counts, BaselineStageCounts):
        raise TypeError("counts must be a BaselineStageCounts.")

    return {
        "stage": counts.stage.value,
        "reused": counts.reused,
        "worked": counts.worked,
        "failed": counts.failed,
    }


def baseline_ingest_summary_payload(
    summary: BaselineIngestSummary,
) -> JsonObject:
    """Return canonical JSON-compatible data for one ingest summary."""
    if not isinstance(summary, BaselineIngestSummary):
        raise TypeError("summary must be a BaselineIngestSummary.")

    return {
        "corpus_id": summary.corpus_id.as_str(),
        "proposed_corpus_version": (summary.proposed_corpus_version.as_str()),
        "expected_corpus_version": (
            summary.expected_corpus_version.as_str()
            if summary.expected_corpus_version is not None
            else None
        ),
        "execution_status": summary.execution_status.value,
        "publication_status": summary.publication_status.value,
        "document_outcomes": [
            _document_outcome_payload(outcome) for outcome in summary.document_outcomes
        ],
        "stage_counts": [
            _stage_counts_payload(counts) for counts in summary.stage_counts
        ],
        "failures": [_failure_payload(failure) for failure in summary.failures],
    }


def serialise_baseline_ingest_summary(
    summary: BaselineIngestSummary,
) -> bytes:
    """Serialise one summary as canonical UTF-8 JSON with a newline."""
    return canonical_json_bytes(baseline_ingest_summary_payload(summary))
