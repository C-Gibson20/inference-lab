"""Tests for baseline observed-stage instrumentation and sink fan-out."""

import asyncio
from datetime import UTC, datetime

import pytest

from inference_lab.profiling import (
    FailureEvent,
    FailureMapping,
    FanoutSink,
    ObservedStage,
    StageTiming,
)


class FakeClock:
    """Return predetermined nanosecond values for deterministic tests."""

    def __init__(self, *values: int) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


class CollectingSink:
    """Collect immutable records in delivery order."""

    def __init__(self) -> None:
        self.records: list[object] = []

    def emit(self, record: object) -> None:
        self.records.append(record)


def fixed_timestamp() -> datetime:
    """Return a deterministic UTC timestamp for record tests."""
    return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def observed_stage(
    sink: CollectingSink,
    *,
    stage_id: str = "parse-1",
    condition: str = "uncontrolled",
    clock: FakeClock | None = None,
    failure_mapper=None,
) -> ObservedStage:
    """Build one observed stage with deterministic clocks and identity."""
    return ObservedStage(
        module="ingest",
        pipeline_stage="parse",
        stage_id=stage_id,
        sink=sink,
        condition=condition,  # type: ignore[arg-type]
        clock=clock or FakeClock(1_000_000_000, 1_025_000_000),
        observed_at_factory=fixed_timestamp,
        run_id="run-1",
        request_id="request-1",
        document_id="document-1",
        corpus_id="corpus-1",
        corpus_version="version-1",
        parent_stage_id="discovery-1",
        failure_mapper=failure_mapper,
    )


def timing_records(sink: CollectingSink) -> list[StageTiming]:
    """Return timing records from a collected stream."""
    return [record for record in sink.records if isinstance(record, StageTiming)]


def failure_records(sink: CollectingSink) -> list[FailureEvent]:
    """Return failure records from a collected stream."""
    return [record for record in sink.records if isinstance(record, FailureEvent)]


def test_successful_observation_has_one_start_and_one_terminal_record() -> None:
    sink = CollectingSink()

    with observed_stage(sink, condition="cold") as stage:
        stage.set_counts(items_processed=1, records_emitted=4)
        stage.mark_success()

    records = timing_records(sink)
    assert [(record.event, record.status) for record in records] == [
        ("stage_started", "started"),
        ("stage_completed", "completed"),
    ]
    assert records[1].condition == "cold"
    assert records[1].duration_ms == 25.0
    assert records[1].items_processed == 1
    assert records[1].records_emitted == 4
    assert records[1].run_id == "run-1"
    assert records[1].parent_stage_id == "discovery-1"
    assert failure_records(sink) == []


@pytest.mark.parametrize("disposition", ["reuse", "empty", "no_change"])
def test_success_dispositions_complete_without_failure_records(
    disposition: str,
) -> None:
    sink = CollectingSink()

    with observed_stage(sink) as stage:
        stage.mark_success()

    assert timing_records(sink)[-1].status == "completed"
    assert failure_records(sink) == []
    assert disposition in {"reuse", "empty", "no_change"}


def test_returned_failure_emits_terminal_and_failure_records_without_raising() -> None:
    sink = CollectingSink()

    with observed_stage(sink) as stage:
        stage.set_counts(items_processed=3, records_emitted=1)
        stage.mark_failure(
            FailureMapping(
                failure_category="dependency",
                failure_code="parser_unavailable",
                recoverable=True,
                retry_count=2,
                partial=True,
            )
        )

    completion = timing_records(sink)[-1]
    failure = failure_records(sink)[0]
    assert completion.status == "failed"
    assert completion.items_processed == 3
    assert completion.records_emitted == 1
    assert failure.failure_category == "dependency"
    assert failure.failure_code == "parser_unavailable"
    assert failure.recoverable is True
    assert failure.retry_count == 2
    assert failure.partial is True
    assert len(failure_records(sink)) == 1


def test_failure_mapping_rejects_payload_oriented_exception_metadata() -> None:
    with pytest.raises(ValueError, match="qualified exception class name"):
        FailureMapping(
            failure_category="input",
            failure_code="bad_input",
            exception_type="ValueError: secret",
        )


def test_failure_mapping_rejects_measurement_only_category() -> None:
    with pytest.raises(ValueError, match="for a stage"):
        FailureMapping(
            failure_category="measurement_unavailable",  # type: ignore[arg-type]
            failure_code="sampler_unavailable",
        )


def test_raised_failure_is_mapped_and_primary_exception_is_preserved() -> None:
    sink = CollectingSink()
    primary = ValueError("sentinel document payload")

    def map_failure(error: BaseException) -> FailureMapping:
        assert error is primary
        return FailureMapping(
            failure_category="input",
            failure_code="invalid_source",
            exception_type="builtins.ValueError",
        )

    with pytest.raises(ValueError) as raised:
        with observed_stage(sink, failure_mapper=map_failure):
            raise primary

    assert raised.value is primary
    assert timing_records(sink)[-1].status == "failed"
    assert failure_records(sink)[0].failure_code == "invalid_source"
    assert failure_records(sink)[0].exception_type == "builtins.ValueError"
    assert "sentinel document payload" not in repr(sink.records)


def test_cancelled_stage_emits_one_cancelled_failure_and_preserves_cancellation() -> (
    None
):
    sink = CollectingSink()

    with pytest.raises(asyncio.CancelledError):
        with observed_stage(sink):
            raise asyncio.CancelledError

    assert timing_records(sink)[-1].status == "cancelled"
    failure = failure_records(sink)[0]
    assert failure.status == "cancelled"
    assert failure.failure_category == "cancelled"
    assert failure.level == "warning"
    assert len(failure_records(sink)) == 1


def test_sink_failure_does_not_replace_a_returned_failure() -> None:
    class FailingSink:
        def __init__(self) -> None:
            self.calls = 0

        def emit(self, record: object) -> None:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("sentinel sink failure")

    sink = FailingSink()
    with observed_stage(sink) as stage:  # type: ignore[arg-type]
        stage.mark_failure(
            FailureMapping(
                failure_category="internal",
                failure_code="returned_failure",
            )
        )

    assert sink.calls == 3


def test_successful_stage_surfaces_sink_failure() -> None:
    class FailingSink:
        def __init__(self) -> None:
            self.calls = 0

        def emit(self, record: object) -> None:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("sentinel sink failure")

    sink = FailingSink()
    with pytest.raises(RuntimeError, match="sentinel sink failure"):
        with observed_stage(sink):  # type: ignore[arg-type]
            pass


def test_fanout_delivers_the_same_record_to_all_sinks_in_order() -> None:
    first = CollectingSink()
    second = CollectingSink()
    fanout = FanoutSink(first, second)
    record = StageTiming(
        event="stage_started",
        module="profiling",
        pipeline_stage="parse",
        stage_id="parse-1",
        observed_at=fixed_timestamp(),
    )

    fanout.emit(record)

    assert first.records == [record]
    assert second.records == [record]
    assert first.records[0] is second.records[0]


def test_fanout_continues_after_a_consumer_failure_and_raises_first_error() -> None:
    calls: list[str] = []

    class FailingSink:
        def emit(self, record: object) -> None:
            calls.append("first")
            raise ValueError("first sink failure")

    class RecordingSink:
        def emit(self, record: object) -> None:
            calls.append("second")

    with pytest.raises(ValueError, match="first sink failure"):
        FanoutSink(FailingSink(), RecordingSink()).emit(
            StageTiming(
                event="stage_started",
                module="profiling",
                pipeline_stage="parse",
                stage_id="parse-1",
                observed_at=fixed_timestamp(),
            )
        )

    assert calls == ["first", "second"]
