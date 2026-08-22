"""Tests for privacy-safe evaluation records."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from inference_lab.evaluation.records import (
    EvaluationRecord,
    EvaluationRecorder,
    evaluation_record_id,
)


RUN_ID = "run-" + "a" * 64
CASE_ID = "case-001"
TIMESTAMP = datetime(2026, 8, 22, tzinfo=timezone.utc)


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def warning(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def test_record_id_is_stable_and_record_contains_no_content_fields() -> None:
    record = EvaluationRecord(
        record_id=evaluation_record_id(RUN_ID, CASE_ID),
        run_id=RUN_ID,
        case_id=CASE_ID,
        status="passed",
        duration_ms=12.5,
        timestamp=TIMESTAMP,
    )

    assert record.record_id == evaluation_record_id(RUN_ID, CASE_ID)
    assert set(record.model_dump()) <= {
        "record_id",
        "run_id",
        "case_id",
        "status",
        "duration_ms",
        "warning_code",
        "error_code",
        "timestamp",
        "record_schema_version",
    }


def test_failed_record_requires_machine_readable_error_code() -> None:
    with pytest.raises(ValidationError, match="error_code"):
        EvaluationRecord(
            record_id=evaluation_record_id(RUN_ID, CASE_ID),
            run_id=RUN_ID,
            case_id=CASE_ID,
            status="failed",
            timestamp=TIMESTAMP,
        )


def test_recorder_emits_warning_and_error_without_payloads() -> None:
    logger = FakeLogger()
    recorder = EvaluationRecorder(logger)

    recorder.invalid_input(issue_code="manifest_invalid", timestamp=TIMESTAMP)
    recorder.record(
        run_id=RUN_ID,
        case_id=CASE_ID,
        status="failed",
        error_code="backend_unavailable",
        timestamp=TIMESTAMP,
    )

    assert [event for event, _ in logger.events] == [
        "evaluation_input_rejected",
        "evaluation_unit_failed",
    ]
    assert all("question" not in fields for _, fields in logger.events)
    assert all("output" not in fields for _, fields in logger.events)
