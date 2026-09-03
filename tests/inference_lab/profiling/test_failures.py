"""Tests for privacy-safe failure-event records."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from inference_lab.profiling import FailureEvent


def fixed_timestamp() -> datetime:
    """Return a deterministic UTC timestamp for record tests."""
    return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def test_failed_event_serialises_operational_fields() -> None:
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
        observed_at=fixed_timestamp(),
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
        "corpus_id": None,
        "corpus_version": None,
        "status": "failed",
        "level": "error",
        "recoverable": False,
        "retry_count": 1,
        "partial": True,
        "exception_type": "ValueError",
        "observed_at": "2026-08-21T12:00:00+00:00",
        "record_schema_version": "3.0.0",
    }


def test_degraded_measurement_event_is_warning() -> None:
    record = FailureEvent(
        failure_category="measurement_unavailable",
        failure_code="gpu_sampler_unavailable",
        event="unit_degraded",
        status="degraded",
        level="warning",
        recoverable=True,
        observed_at=fixed_timestamp(),
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
        observed_at=fixed_timestamp(),
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"event": "failed"}, "valid failure event"),
        ({"failure_category": "unknown"}, "failure_category must be supported"),
        ({"status": "unknown"}, "status must be valid"),
        ({"level": "info"}, "failure level must be warning or error"),
    ],
)
def test_failure_discriminants_must_be_supported(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "failure_category": "internal",
        "failure_code": "unexpected_error",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        FailureEvent(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"module": ""},
        {"pipeline_stage": ""},
        {"stage_id": ""},
    ],
)
def test_failure_context_fields_reject_empty_strings(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "failure_category": "internal",
        "failure_code": "unexpected_error",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="must not be empty"):
        FailureEvent(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["recoverable", "partial"])
def test_failure_boolean_flags_require_booleans(field_name: str) -> None:
    values: dict[str, object] = {
        "failure_category": "internal",
        "failure_code": "unexpected_error",
        field_name: 1,
    }

    with pytest.raises(ValueError, match=f"{field_name} must be a boolean"):
        FailureEvent(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status", "event", "level", "message"),
    [
        ("degraded", "unit_failed", "warning", "must use unit_degraded"),
        ("failed", "unit_failed", "warning", "failed records must use error"),
        ("cancelled", "unit_failed", "error", "cancelled records must use warning"),
    ],
)
def test_failure_status_controls_event_and_level(
    status: str,
    event: str,
    level: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            status=status,  # type: ignore[arg-type]
            event=event,  # type: ignore[arg-type]
            level=level,  # type: ignore[arg-type]
        )


def test_cancelled_category_requires_cancelled_status() -> None:
    with pytest.raises(ValueError, match="cancelled category"):
        FailureEvent(
            failure_category="cancelled",
            failure_code="request_cancelled",
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


@pytest.mark.parametrize(
    "exception_type",
    ["", "ValueError: secret", "ValueError secret", "package/Class", "Error\nsecret"],
)
def test_exception_metadata_must_be_qualified_type_name(
    exception_type: str,
) -> None:
    with pytest.raises(ValueError, match="qualified exception class name"):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            exception_type=exception_type,
        )


def test_qualified_exception_type_is_accepted() -> None:
    record = FailureEvent(
        failure_category="internal",
        failure_code="unexpected_error",
        exception_type="builtins.RuntimeError",
    )

    assert record.exception_type == "builtins.RuntimeError"


def test_naive_and_non_utc_observation_times_are_rejected() -> None:
    with pytest.raises(ValueError, match="observed_at must be timezone-aware UTC"):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            observed_at=datetime(2026, 8, 21, 12, 0),
        )

    non_utc = datetime(
        2026,
        8,
        21,
        12,
        0,
        tzinfo=timezone(timedelta(hours=1)),
    )
    with pytest.raises(ValueError, match="observed_at must be in UTC"):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            observed_at=non_utc,
        )


def test_failure_event_rejects_invalid_schema_version() -> None:
    with pytest.raises(ValueError, match="record_schema_version"):
        FailureEvent(
            failure_category="internal",
            failure_code="unexpected_error",
            record_schema_version="2.0.0",
        )
