"""Privacy-safe lifecycle observation for corpus-audit operations."""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog
from structlog.contextvars import bound_contextvars

from inference_lab._validation import is_strict_integer
from inference_lab.evaluation.audit_contracts import (
    AUDIT_CONTRACT_VERSION,
    CorpusAuditRequest,
)
from inference_lab.evaluation.audit_errors import (
    AuditAggregationError,
    AuditContractError,
)
from inference_lab.evaluation.audit_reports import (
    AUDIT_IMPLEMENTATION,
    AUDIT_IMPLEMENTATION_VERSION,
    AUDIT_REPORT_SCHEMA_VERSION,
    CorpusAuditReport,
)

Clock = Callable[[], int]

_LOGGER = structlog.get_logger(__name__)


def _read_clock(clock: Clock) -> int:
    value = clock()
    if not is_strict_integer(value):
        raise TypeError("clock must return an integer nanosecond value.")
    return value


def _emit_event(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded audit metadata without changing the audit outcome."""
    try:
        getattr(_LOGGER, method_name)(event, pipeline_stage="corpus-audit", **fields)
    except Exception:
        pass


def _base_log_fields(request: CorpusAuditRequest) -> dict[str, object]:
    return {
        "report_schema_version": AUDIT_REPORT_SCHEMA_VERSION,
        "audit_implementation": AUDIT_IMPLEMENTATION,
        "audit_implementation_version": AUDIT_IMPLEMENTATION_VERSION,
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "source_file_count": request.source_inventory.discovery_counts.files_seen,
        "member_count": len(request.scope.memberships),
        "parse_observation_count": len(request.parse_observations),
    }


def _failure_log_fields(error: Exception) -> dict[str, object]:
    if isinstance(error, (AuditContractError, AuditAggregationError)):
        return {
            "error_category": error.category.value,
            "failure_stage": error.stage,
            "retryable": error.retryable,
        }
    return {
        "error_category": AuditAggregationError().category.value,
        "failure_stage": AuditAggregationError().stage,
        "retryable": False,
    }


def _build_report(request: CorpusAuditRequest) -> CorpusAuditReport:
    return CorpusAuditReport.from_request(request)


def run_corpus_audit(
    request: CorpusAuditRequest,
    *,
    clock: Clock = time.monotonic_ns,
) -> CorpusAuditReport:
    """Run one corpus audit with privacy-safe lifecycle observability."""
    if not isinstance(request, CorpusAuditRequest):
        raise TypeError("request must be a CorpusAuditRequest.")
    if not callable(clock):
        raise TypeError("clock must be callable.")

    started_ns = _read_clock(clock)
    base_fields = _base_log_fields(request)
    with bound_contextvars(
        corpus_id=request.scope.corpus_id.as_str(),
        corpus_version=request.scope.corpus_version.as_str(),
        document_id=None,
    ):
        _emit_event("info", "corpus_audit_started", **base_fields)
        try:
            report = _build_report(request)
            finished_ns = _read_clock(clock)
            if finished_ns < started_ns:
                raise ValueError("clock must not move backwards during an audit.")
            duration_ms = (finished_ns - started_ns) / 1_000_000
        except AuditContractError as error:
            _emit_event(
                "error",
                "corpus_audit_failed",
                **base_fields,
                **_failure_log_fields(error),
            )
            raise
        except Exception as error:
            failure = (
                error
                if isinstance(error, AuditAggregationError)
                else AuditAggregationError()
            )
            _emit_event(
                "error",
                "corpus_audit_failed",
                **base_fields,
                **_failure_log_fields(failure),
            )
            if failure is not error:
                raise failure from None
            raise

        acquisition_coverage = report.acquisition.coverage.coverage
        _emit_event(
            "info",
            "corpus_audit_completed",
            **base_fields,
            overall_status=report.overall_status.value,
            known_count=acquisition_coverage.observed
            + acquisition_coverage.observed_absent,
            missing_count=acquisition_coverage.missing,
            failed_count=acquisition_coverage.failed,
            limitation_count=len(report.limitations),
            duration_ms=duration_ms,
        )
        return report
