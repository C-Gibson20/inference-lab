"""Tests for privacy-safe failure-event records."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from inference_lab.profiling import FailureEvent


def fixed_timestamp() -> datetime:
    """Return a deterministic UTC timestamp for record tests."""
    return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def test_failed_event_serializes_operational_fields() -> None:
    record = FailureEvent(
        failure_category="input",
        failure_code="malformed_document",
        module="ingest",
        pipeline_stage="parse",
        stage_id="parse-1",
        status="failed",
        level="error",
        recoverable=False,
        retry_count=1,
        partial=True,
        exception_type="ValueError",
        timestamp=fixed_timestamp(),
    )

    assert record.to_dict() == {
        "event": "unit_failed",
        "failure_category": "input",
        "failure_code": "malformed_document",
        "module": "ingest",
        "pipeline_stage": "parse",
        "stage_id": "parse-1",
        "run_id": None,
        "request_id": None,
        "document_id": None,
        "corpus_version": None,
        "status": "failed",
        "level": "error",
        "recoverable": False,
        "retry_count": 1,
        "partial": True,
        "exception_type": "ValueError",
        "timestamp": "2026-08-21T12:00:00+00:00",
        "record_schema_version": 1,
    }


def test_degraded_measurement_event_is_warning() -> None:
    record = FailureEvent(
        failure_category="measurement_unavailable",
        failure_code="gpu_sampler_unavailable",
        event="unit_degraded",
        status="degraded",
        level="warning",
        recoverable=True,
        timestamp=fixed_timestamp(),
    )

    assert record.event == "unit_degraded"
    assert record.status == "degraded"
    assert record.level == "warning"


def test_cancelled_event_requires_cancelled_status() -> None:
    record = FailureEvent(
        failure_category="cancelled",
        failure_code="request_cancelled",
        status="cancelled",
        level="warning",
        timestamp=fixed_timestamp(),
    )

    assert record.to_dict()["status"] == "cancelled"


@pytest.mark.parametrize("code", ["", "MalformedDocument", "bad-code"])
def test_failure_code_must_be_snake_case(code: str) -> None:
    with pytest.raises(ValueError, match="failure_code"):
        FailureEvent(failure_category="internal", failure_code=code)


def test_failure_status_and_event_must_agree() -> None:
    with pytest.raises(ValueError, match="unit_failed"):
        FailureEvent(
            failure_category="dependency",
            failure_code="backend_unavailable",
            event="unit_degraded",
            status="failed",
        )


def test_measurement_unavailable_must_be_degraded() -> None:
    with pytest.raises(ValueError, match="measurement_unavailable"):
        FailureEvent(
            failure_category="measurement_unavailable",
            failure_code="sampler_unavailable",
        )


@pytest.mark.parametrize("retry_count", [-1, True, 1.5])
def test_retry_count_must_be_non_negative_integer(retry_count: object) -> None:
    with pytest.raises(ValueError, match="retry_count"):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            retry_count=retry_count,  # type: ignore[arg-type]
        )


def test_exception_metadata_must_not_contain_newlines() -> None:
    with pytest.raises(ValueError, match="exception_type"):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            exception_type="ValueError\nsecret payload",
        )


def test_naive_and_non_utc_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            timestamp=datetime(2026, 8, 21, 12, 0),
        )

    non_utc = datetime(
        2026,
        8,
        21,
        12,
        0,
        tzinfo=timezone(timedelta(hours=1)),
    )
    with pytest.raises(ValueError, match="timestamp must be in UTC"):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            timestamp=non_utc,
        )


def test_failure_event_rejects_invalid_schema_version() -> None:
    with pytest.raises(ValueError, match="record_schema_version"):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            record_schema_version=0,
        )
