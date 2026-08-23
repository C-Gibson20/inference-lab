"""Tests for deterministic stage timing instrumentation."""

import asyncio
from datetime import UTC, datetime

import pytest

from inference_lab.profiling import StageTimer, StageTiming


class FakeClock:
    """Return predetermined nanosecond values for deterministic tests."""

    def __init__(self, *values: int) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def fixed_timestamp() -> datetime:
    """Return a deterministic UTC timestamp for record tests."""
    return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def test_successful_stage_emits_start_and_completion() -> None:
    records: list[StageTiming] = []

    with StageTimer(
        module="ingest",
        pipeline_stage="parse",
        stage_id="parse-1",
        emit=records.append,
        clock=FakeClock(1_000_000_000, 1_025_000_000),
        observed_at_factory=fixed_timestamp,
    ) as timer:
        timer.set_counts(items_processed=1, records_emitted=4)

    assert [record.event for record in records] == [
        "stage_started",
        "stage_completed",
    ]
    assert records[0].status == "started"
    assert records[1].status == "completed"
    assert records[1].duration_ms == 25.0
    assert records[1].items_processed == 1
    assert records[1].records_emitted == 4


def test_stage_exception_is_recorded_and_re_raised() -> None:
    records: list[StageTiming] = []

    with pytest.raises(RuntimeError, match="parse failed"):
        with StageTimer(
            module="ingest",
            pipeline_stage="parse",
            stage_id="parse-1",
            emit=records.append,
            clock=FakeClock(10, 1_000_010),
            observed_at_factory=fixed_timestamp,
        ):
            raise RuntimeError("parse failed")

    assert records[-1].event == "stage_completed"
    assert records[-1].status == "failed"
    assert records[-1].duration_ms == 1.0


def test_completion_emission_failure_does_not_mask_stage_failure() -> None:
    def emit(record: StageTiming) -> None:
        if record.event == "stage_completed":
            raise RuntimeError("profiling emission failed")

    with pytest.raises(ValueError, match="stage failed"):
        with StageTimer(
            module="ingest",
            pipeline_stage="parse",
            stage_id="parse-1",
            emit=emit,
            clock=FakeClock(10, 20),
            observed_at_factory=fixed_timestamp,
        ):
            raise ValueError("stage failed")


def test_completion_emission_failure_surfaces_after_successful_stage() -> None:
    def emit(record: StageTiming) -> None:
        if record.event == "stage_completed":
            raise RuntimeError("profiling emission failed")

    with pytest.raises(RuntimeError, match="profiling emission failed"):
        with StageTimer(
            module="ingest",
            pipeline_stage="parse",
            stage_id="parse-1",
            emit=emit,
            clock=FakeClock(10, 20),
            observed_at_factory=fixed_timestamp,
        ):
            pass


def test_cancelled_stage_is_recorded() -> None:
    records: list[StageTiming] = []

    with pytest.raises(asyncio.CancelledError):
        with StageTimer(
            module="runtime",
            pipeline_stage="generate",
            stage_id="generate-1",
            emit=records.append,
            clock=FakeClock(100, 200),
            observed_at_factory=fixed_timestamp,
        ):
            raise asyncio.CancelledError

    assert records[-1].status == "cancelled"


def test_nested_stage_context_preserves_parent_id() -> None:
    records: list[StageTiming] = []

    with StageTimer(
        module="ingest",
        pipeline_stage="parse",
        stage_id="parse-1",
        emit=records.append,
        clock=FakeClock(0, 30),
        observed_at_factory=fixed_timestamp,
    ):
        with StageTimer(
            module="ingest",
            pipeline_stage="ocr",
            stage_id="ocr-1",
            parent_stage_id="parse-1",
            emit=records.append,
            clock=FakeClock(10, 20),
            observed_at_factory=fixed_timestamp,
        ):
            pass

    assert records[2].parent_stage_id == "parse-1"
    assert records[3].parent_stage_id is None


def test_timer_cannot_be_reused() -> None:
    records: list[StageTiming] = []
    timer = StageTimer(
        module="ingest",
        pipeline_stage="parse",
        stage_id="parse-1",
        emit=records.append,
        clock=FakeClock(0, 1),
        observed_at_factory=fixed_timestamp,
    )

    with timer:
        pass

    with pytest.raises(RuntimeError, match="cannot be reused"):
        timer.__enter__()


@pytest.mark.parametrize("field", ["items_processed", "records_emitted"])
def test_negative_counter_is_rejected(field: str) -> None:
    records: list[StageTiming] = []

    with StageTimer(
        module="ingest",
        pipeline_stage="parse",
        stage_id="parse-1",
        emit=records.append,
        clock=FakeClock(0, 1),
        observed_at_factory=fixed_timestamp,
    ) as timer:
        with pytest.raises(ValueError, match="non-negative integer"):
            timer.set_counts(**{field: -1})


def test_timing_record_serialises_nullable_fields() -> None:
    record = StageTiming(
        event="stage_completed",
        module="profiling",
        pipeline_stage="parse",
        stage_id="parse-1",
        observed_at=fixed_timestamp(),
        duration_ms=2.5,
        status="completed",
    )

    payload = record.to_dict()

    assert payload["duration_ms"] == 2.5
    assert payload["items_processed"] is None
    assert payload["records_emitted"] is None
    assert payload["observed_at"] == "2026-08-21T12:00:00+00:00"
    assert payload["record_schema_version"] == 2


@pytest.mark.parametrize(
    "duration_ms",
    [float("nan"), float("inf"), float("-inf"), True, "1.0"],
)
def test_timing_record_rejects_invalid_duration(duration_ms: object) -> None:
    with pytest.raises(ValueError, match="finite non-negative number"):
        StageTiming(
            event="stage_completed",
            module="profiling",
            pipeline_stage="parse",
            stage_id="parse-1",
            duration_ms=duration_ms,  # type: ignore[arg-type]
            status="completed",
        )


def test_timing_record_rejects_invalid_schema_version() -> None:
    with pytest.raises(ValueError, match="record_schema_version"):
        StageTiming(
            event="stage_completed",
            module="profiling",
            pipeline_stage="parse",
            stage_id="parse-1",
            duration_ms=1.0,
            status="completed",
            record_schema_version=0,
        )
