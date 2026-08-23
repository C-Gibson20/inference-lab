import asyncio
import traceback

import pytest

from inference_lab.backends.runtime import (
    BackendExecutionError,
    BackendProtocolError,
    GenerationEvent,
    GenerationEventType,
    validate_generation_stream,
)

BACKEND_SECRET = "PRIVATE BACKEND FAILURE DETAIL"


async def events(*items: GenerationEvent):
    for item in items:
        yield item


def delta(sequence_number: int, text: str) -> GenerationEvent:
    return GenerationEvent(
        event_type=GenerationEventType.DELTA,
        sequence_number=sequence_number,
        text_delta=text,
    )


def completed(sequence_number: int) -> GenerationEvent:
    return GenerationEvent(
        event_type=GenerationEventType.COMPLETED,
        sequence_number=sequence_number,
        finish_reason="stop",
    )


async def collect(stream) -> list[GenerationEvent]:
    return [event async for event in stream]


def test_valid_stream_forwards_deltas_and_completion() -> None:
    result = asyncio.run(
        collect(validate_generation_stream(events(delta(0, "hello"), completed(1))))
    )

    assert [event.text_delta for event in result] == ["hello", None]


def test_stream_rejects_non_contiguous_sequence_numbers() -> None:
    with pytest.raises(BackendProtocolError, match="contiguous"):
        asyncio.run(
            collect(validate_generation_stream(events(delta(1, "hello"), completed(2))))
        )


def test_stream_requires_completion() -> None:
    with pytest.raises(BackendProtocolError, match="without completion"):
        asyncio.run(collect(validate_generation_stream(events(delta(0, "hello")))))


async def invalid_events():
    yield object()


def test_stream_classifies_invalid_event_objects_as_protocol_errors() -> None:
    with pytest.raises(BackendProtocolError, match="GenerationEvent values"):
        asyncio.run(
            collect(
                validate_generation_stream(invalid_events())  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize("trailing_event", [delta(2, "late"), completed(2)])
def test_stream_rejects_events_after_completion(
    trailing_event: GenerationEvent,
) -> None:
    async def consume() -> list[GenerationEvent]:
        forwarded = []
        with pytest.raises(BackendProtocolError, match="after completion"):
            async for event in validate_generation_stream(
                events(delta(0, "hello"), completed(1), trailing_event)
            ):
                forwarded.append(event)
        return forwarded

    forwarded = asyncio.run(consume())

    assert [event.event_type for event in forwarded] == [GenerationEventType.DELTA]


async def failing_events():
    yield delta(0, "partial")
    raise RuntimeError(BACKEND_SECRET)


def test_stream_preserves_partial_output_before_backend_failure() -> None:
    async def consume() -> list[str | None]:
        output = []
        with pytest.raises(BackendExecutionError) as captured:
            async for event in validate_generation_stream(failing_events()):
                output.append(event.text_delta)
        rendered_exception = "".join(traceback.format_exception(captured.value))
        assert BACKEND_SECRET not in rendered_exception
        assert captured.value.__cause__ is None
        assert captured.value.__suppress_context__ is True
        return output

    assert asyncio.run(consume()) == ["partial"]


async def cancelled_events():
    yield delta(0, "partial")
    raise asyncio.CancelledError


def test_stream_propagates_cancellation_without_completion() -> None:
    async def consume() -> list[str | None]:
        output = []
        with pytest.raises(asyncio.CancelledError):
            async for event in validate_generation_stream(cancelled_events()):
                output.append(event.text_delta)
        return output

    assert asyncio.run(consume()) == ["partial"]
