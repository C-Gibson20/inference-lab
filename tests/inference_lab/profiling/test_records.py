"""Tests for profiling record contracts."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from inference_lab.profiling import ProfilingRecord


def test_valid_record_serializes_all_fields() -> None:
    timestamp = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)

    record = ProfilingRecord(
        event="stage_completed",
        module="profiling",
        pipeline_stage="parse",
        run_id="run-1",
        timestamp=timestamp,
    )

    assert record.to_dict() == {
        "event": "stage_completed",
        "module": "profiling",
        "level": "info",
        "pipeline_stage": "parse",
        "document_id": None,
        "corpus_version": None,
        "run_id": "run-1",
        "request_id": None,
        "stage_id": None,
        "parent_stage_id": None,
        "timestamp": "2026-08-21T12:00:00+00:00",
        "record_schema_version": 1,
    }


@pytest.mark.parametrize("event", ["", "StageCompleted", "stage-completed"])
def test_invalid_event_is_rejected(event: str) -> None:
    with pytest.raises(ValueError):
        ProfilingRecord(event=event, module="profiling")


@pytest.mark.parametrize("field", ["event", "module"])
def test_required_text_fields_cannot_be_empty(field: str) -> None:
    values = {"event": "stage_completed", "module": "profiling"}
    values[field] = ""

    with pytest.raises(ValueError):
        ProfilingRecord(**values)


def test_invalid_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported log level"):
        ProfilingRecord(
            event="stage_completed",
            module="profiling",
            level="notice",  # type: ignore[arg-type]
        )


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        ProfilingRecord(
            event="stage_completed",
            module="profiling",
            timestamp=datetime(2026, 8, 21, 12, 0),
        )


def test_non_utc_timestamp_is_rejected() -> None:
    timestamp = datetime(
        2026,
        8,
        21,
        12,
        0,
        tzinfo=timezone(timedelta(hours=1)),
    )

    with pytest.raises(ValueError, match="timestamp must be in UTC"):
        ProfilingRecord(
            event="stage_completed",
            module="profiling",
            timestamp=timestamp,
        )


@pytest.mark.parametrize("version", [0, -1])
def test_non_positive_schema_version_is_rejected(version: int) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        ProfilingRecord(
            event="stage_completed",
            module="profiling",
            record_schema_version=version,
        )


def test_record_is_immutable() -> None:
    record = ProfilingRecord(event="stage_completed", module="profiling")

    with pytest.raises(AttributeError):
        record.event = "changed"  # type: ignore[misc]
