"""Tests for privacy-safe corpus-audit lifecycle observation."""

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

import inference_lab.evaluation.audit_observability as audit_observability
from inference_lab.evaluation import (
    AuditAggregationError,
    AuditContractError,
    AuditErrorCategory,
    CorpusAuditReport,
    run_corpus_audit,
)

from .test_audit_aggregation import (  # noqa: TID252
    _DOCUMENT_ONE,
    _VERSION_ONE,
    _membership,
    _observation,
    _request,
)


def _audit_request():
    membership = _membership(_DOCUMENT_ONE, _VERSION_ONE)
    return _request((membership,), (_observation(membership),))


def test_run_corpus_audit_logs_bounded_lifecycle_and_binds_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _audit_request()
    real_build_report = audit_observability._build_report
    observed_context: dict[str, object] = {}

    def build_report(request_to_build):
        observed_context.update(get_contextvars())
        return real_build_report(request_to_build)

    monkeypatch.setattr(audit_observability, "_build_report", build_report)
    clock_values = iter((1_000_000_000, 1_025_000_000))

    with capture_logs() as logs:
        report = run_corpus_audit(request, clock=clock_values.__next__)

    assert isinstance(report, CorpusAuditReport)
    assert observed_context == {
        "corpus_id": request.scope.corpus_id.as_str(),
        "corpus_version": request.scope.corpus_version.as_str(),
        "document_id": None,
    }
    assert get_contextvars() == {}
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("corpus_audit_started", "info"),
        ("corpus_audit_completed", "info"),
    ]
    assert logs[0]["pipeline_stage"] == "corpus-audit"
    assert (
        logs[0]["source_file_count"]
        == request.source_inventory.discovery_counts.files_seen
    )
    assert logs[0]["member_count"] == 1
    assert logs[0]["parse_observation_count"] == 1
    assert logs[1]["overall_status"] == report.overall_status.value
    assert logs[1]["duration_ms"] == 25.0
    assert isinstance(logs[1]["known_count"], int)
    assert isinstance(logs[1]["missing_count"], int)
    assert isinstance(logs[1]["failed_count"], int)
    assert logs[1]["limitation_count"] == len(report.limitations)
    rendered_logs = repr(logs)
    assert report.input_digest not in rendered_logs
    assert report.report_digest not in rendered_logs


def test_run_corpus_audit_wraps_unexpected_failure_without_private_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_marker = "/private/audit/source-payload"

    def fail(_request) -> CorpusAuditReport:
        raise RuntimeError(private_marker)

    monkeypatch.setattr(audit_observability, "_build_report", fail)

    with capture_logs() as logs:
        with pytest.raises(AuditAggregationError) as captured:
            run_corpus_audit(_audit_request(), clock=iter((1, 2)).__next__)

    assert captured.value.category is AuditErrorCategory.AGGREGATION_FAILURE
    assert captured.value.stage == "aggregation"
    assert captured.value.retryable is False
    assert [record["event"] for record in logs] == [
        "corpus_audit_started",
        "corpus_audit_failed",
    ]
    assert logs[-1]["error_category"] == "aggregation_failure"
    assert logs[-1]["failure_stage"] == "aggregation"
    assert logs[-1]["retryable"] is False
    assert private_marker not in repr(logs)
    assert private_marker not in str(captured.value)


def test_run_corpus_audit_preserves_typed_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = AuditContractError(AuditErrorCategory.INVALID_INPUT)

    def fail(_request) -> CorpusAuditReport:
        raise failure

    monkeypatch.setattr(audit_observability, "_build_report", fail)

    with capture_logs() as logs:
        with pytest.raises(AuditContractError) as captured:
            run_corpus_audit(_audit_request(), clock=iter((1, 2)).__next__)

    assert captured.value is failure
    assert logs[-1]["error_category"] == "invalid_input"
    assert logs[-1]["failure_stage"] == "contract"
    assert logs[-1]["retryable"] is False
