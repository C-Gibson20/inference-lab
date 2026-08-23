"""Validation helpers for generator backend streams."""

from collections.abc import AsyncIterable, AsyncIterator

from .errors import BackendError, BackendExecutionError, BackendProtocolError
from .types import GenerationEvent, GenerationEventType


async def validate_generation_stream(
    events: AsyncIterable[GenerationEvent],
) -> AsyncIterator[GenerationEvent]:
    """Validate and forward a generator's ordered event stream.

    The stream must start at sequence zero, increment by one for each event,
    and end with one completion event. Backend exceptions and cancellation are
    propagated unchanged. Unexpected implementation exceptions are converted to
    ``BackendExecutionError`` without exposing backend payloads. The completion
    event is forwarded only after the upstream stream is confirmed exhausted.
    """
    expected_sequence = 0
    completion: GenerationEvent | None = None

    try:
        async for event in events:
            if not isinstance(event, GenerationEvent):
                raise BackendProtocolError(
                    "generation stream must contain GenerationEvent values"
                )
            if completion is not None:
                raise BackendProtocolError(
                    "generation stream cannot contain events after completion"
                )
            if event.sequence_number != expected_sequence:
                raise BackendProtocolError(
                    "generation stream sequence numbers must be contiguous"
                )
            expected_sequence += 1
            if event.event_type is GenerationEventType.COMPLETED:
                completion = event
            else:
                yield event
    except BackendError:
        raise
    except Exception:
        raise BackendExecutionError() from None

    if completion is None:
        raise BackendProtocolError("generation stream ended without completion")
    yield completion
