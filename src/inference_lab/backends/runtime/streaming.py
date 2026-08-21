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
    ``BackendExecutionError`` without exposing backend payloads.
    """
    expected_sequence = 0

    try:
        async for event in events:
            if event.sequence_number != expected_sequence:
                raise BackendProtocolError(
                    "generation stream sequence numbers must be contiguous"
                )
            expected_sequence += 1
            yield event
            if event.event_type is GenerationEventType.COMPLETED:
                return
    except BackendError:
        raise
    except Exception as error:
        raise BackendExecutionError() from error

    raise BackendProtocolError("generation stream ended without completion")
