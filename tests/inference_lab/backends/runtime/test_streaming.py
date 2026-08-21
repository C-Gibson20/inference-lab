import asyncio

import pytest

from inference_lab.backends.runtime.errors import (
    BackendExecutionError,
    BackendProtocolError,
)
from inference_lab.backends.runtime.streaming import validate_generation_stream
from inference_lab.backends.runtime.types import GenerationEvent, GenerationEventType


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


async def failing_events():
    yield delta(0, "partial")
    raise RuntimeError("backend implementation detail")


def test_stream_preserves_partial_output_before_backend_failure() -> None:
    async def consume() -> list[str | None]:
        output = []
        with pytest.raises(BackendExecutionError):
            async for event in validate_generation_stream(failing_events()):
                output.append(event.text_delta)
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
