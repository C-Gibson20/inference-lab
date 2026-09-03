"""Autoregressive stream execution for the local causal-model backend."""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Protocol

import structlog
import torch
from transformers.cache_utils import DynamicCache

from inference_lab.backends.runtime.errors import (
    BackendError,
    BackendExecutionError,
    BackendResourceError,
)
from inference_lab.backends.runtime.types import (
    BackendMetadata,
    BackendOperation,
    GenerationEvent,
    GenerationEventType,
    GenerationRequest,
    GenerationUsage,
)

_LOGGER = structlog.get_logger(__name__)


class _ModelOutput(Protocol):
    """Model output required by local greedy generation."""

    logits: torch.Tensor


class _CausalModel(Protocol):
    """Causal-model surface required by local greedy generation."""

    config: object

    def __call__(self, **kwargs: object) -> _ModelOutput:
        """Run one cached or uncached model evaluation."""
        ...


class _Tokeniser(Protocol):
    """Tokeniser surface required by local greedy generation."""

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        """Decode generated token IDs into text."""
        ...


@dataclass(frozen=True, slots=True)
class _PreparedGeneration:
    """Validated prompt and budget prepared for one generation request."""

    rendered_prompt: str
    input_ids: tuple[int, ...]
    input_tokens: int
    available_output_tokens: int


@dataclass(slots=True)
class _GenerationState:
    """Transient model and emission state for one generation stream."""

    attention_mask: torch.Tensor | None
    generated_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""
    sequence_number: int = 0
    finish_reason: str = "length"
    cache: DynamicCache | None = None
    outputs: _ModelOutput | None = None

    @classmethod
    def for_prompt(cls, input_tokens: int) -> "_GenerationState":
        """Create empty generation state for a prepared prompt."""
        return cls(
            attention_mask=torch.ones((1, input_tokens), dtype=torch.long),
        )

    @property
    def output_tokens(self) -> int:
        """Return the number of selected output tokens."""
        return len(self.generated_ids)

    @property
    def should_stop(self) -> bool:
        """Return whether token or text stopping has completed generation."""
        return self.finish_reason == "stop"

    def start(self, model: _CausalModel, input_ids: tuple[int, ...]) -> None:
        """Create native cache state and evaluate the complete prompt."""
        attention_mask = self._require_attention_mask()
        self.cache = DynamicCache()
        self.outputs = model(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            attention_mask=attention_mask,
            past_key_values=self.cache,
            use_cache=True,
        )

    def select_token(self, model: _CausalModel) -> int:
        """Select and record the next greedy token from current logits."""
        outputs = self._require_outputs()
        token_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1))
        self.generated_ids.append(token_id)
        if _is_end_token(model, token_id):
            self.finish_reason = "stop"
        return token_id

    def advance(self, model: _CausalModel, token_id: int) -> None:
        """Advance the native cache by evaluating one selected token."""
        attention_mask = self._require_attention_mask()
        cache = self._require_cache()
        self.attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=torch.long)),
            dim=1,
        )
        self.outputs = model(
            input_ids=torch.tensor([[token_id]], dtype=torch.long),
            attention_mask=self.attention_mask,
            past_key_values=cache,
            use_cache=True,
        )

    def emit_visible_delta(
        self,
        tokeniser: _Tokeniser,
        stop_sequences: tuple[str, ...],
    ) -> GenerationEvent | None:
        """Return newly safe decoded text, retaining possible stop prefixes."""
        decoded = _decode(tokeniser, self.generated_ids)
        stop_position = _stop_position(decoded, stop_sequences)
        visible_text = decoded
        if stop_position is not None:
            visible_text = decoded[:stop_position]
            self.finish_reason = "stop"
        safe_length = _safe_decoded_length(
            visible_text,
            stop_sequences,
            stop_matched=stop_position is not None,
        )
        return self._emit(visible_text[:safe_length])

    def flush_visible_text(
        self,
        tokeniser: _Tokeniser,
        stop_sequences: tuple[str, ...],
    ) -> GenerationEvent | None:
        """Emit text buffered only for an unmatched stop-sequence prefix."""
        decoded = _decode(tokeniser, self.generated_ids)
        stop_position = _stop_position(decoded, stop_sequences)
        visible_text = decoded[:stop_position] if stop_position is not None else decoded
        return self._emit(visible_text)

    def clear(self) -> None:
        """Release all model and generated-token state held by the stream."""
        self.cache = None
        self.outputs = None
        self.attention_mask = None
        self.generated_ids.clear()
        self.emitted_text = ""

    def _emit(self, visible_text: str) -> GenerationEvent | None:
        delta = visible_text[len(self.emitted_text) :]
        if not delta:
            return None
        self.emitted_text = visible_text
        event = GenerationEvent(
            event_type=GenerationEventType.DELTA,
            sequence_number=self.sequence_number,
            text_delta=delta,
        )
        self.sequence_number += 1
        return event

    def _require_attention_mask(self) -> torch.Tensor:
        if self.attention_mask is None:
            raise BackendExecutionError()
        return self.attention_mask

    def _require_cache(self) -> DynamicCache:
        if self.cache is None:
            raise BackendExecutionError()
        return self.cache

    def _require_outputs(self) -> _ModelOutput:
        if self.outputs is None:
            raise BackendExecutionError()
        return self.outputs


async def _generate_local_events(
    request: GenerationRequest,
    prepared: _PreparedGeneration,
    *,
    model: _CausalModel,
    tokeniser: _Tokeniser,
    ensure_available: Callable[[], None],
    metadata: BackendMetadata,
) -> AsyncIterator[GenerationEvent]:
    """Generate greedy tokens and expose payload-free backend failures."""
    state = _GenerationState.for_prompt(prepared.input_tokens)
    output_tokens = 0
    sequence_number = 0
    finish_reason = "length"

    try:
        with torch.inference_mode():
            await asyncio.sleep(0)
            state.start(model, prepared.input_ids)
            for _ in range(prepared.available_output_tokens):
                ensure_available()
                token_id = state.select_token(model)
                event = state.emit_visible_delta(
                    tokeniser,
                    request.options.stop_sequences,
                )
                if event is not None:
                    yield event
                if state.should_stop or (
                    state.output_tokens == prepared.available_output_tokens
                ):
                    break
                await asyncio.sleep(0)
                state.advance(model, token_id)

            if not state.should_stop:
                event = state.flush_visible_text(
                    tokeniser,
                    request.options.stop_sequences,
                )
                if event is not None:
                    yield event
    except asyncio.CancelledError:
        raise
    except BackendError:
        raise
    except MemoryError:
        _log_generation_failure(metadata, failure_category="resource", retryable=True)
        raise BackendResourceError() from None
    except Exception:
        _log_generation_failure(
            metadata,
            failure_category="execution",
            retryable=False,
        )
        raise BackendExecutionError() from None
    finally:
        output_tokens = state.output_tokens
        sequence_number = state.sequence_number
        finish_reason = state.finish_reason
        state.clear()

    yield _completion_event(
        prepared,
        output_tokens=output_tokens,
        sequence_number=sequence_number,
        finish_reason=finish_reason,
    )


def _completion_event(
    prepared: _PreparedGeneration,
    *,
    output_tokens: int,
    sequence_number: int,
    finish_reason: str,
) -> GenerationEvent:
    """Build the terminal event from retained scalar accounting state."""
    return GenerationEvent(
        event_type=GenerationEventType.COMPLETED,
        sequence_number=sequence_number,
        finish_reason=finish_reason,
        usage=GenerationUsage(
            input_tokens=prepared.input_tokens,
            output_tokens=output_tokens,
            total_tokens=prepared.input_tokens + output_tokens,
        ),
    )


def _is_end_token(model: _CausalModel, token_id: int) -> bool:
    """Return whether a generated token is one of the model end tokens."""
    end_token_ids = getattr(model.config, "eos_token_id", None)
    if isinstance(end_token_ids, int):
        return token_id == end_token_ids
    return isinstance(end_token_ids, list) and token_id in end_token_ids


def _decode(tokeniser: _Tokeniser, token_ids: list[int]) -> str:
    """Decode generated token IDs without exposing backend failures."""
    try:
        decoded = tokeniser.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        raise BackendExecutionError() from None
    if not isinstance(decoded, str):
        raise BackendExecutionError()
    return decoded


def _stop_position(text: str, stop_sequences: tuple[str, ...]) -> int | None:
    """Return the earliest matched stop-sequence position, if any."""
    positions = tuple(
        position
        for sequence in stop_sequences
        if (position := text.find(sequence)) >= 0
    )
    return min(positions) if positions else None


def _safe_decoded_length(
    text: str,
    stop_sequences: tuple[str, ...],
    *,
    stop_matched: bool,
) -> int:
    """Keep a suffix buffered while it could become a stop sequence."""
    if stop_matched or not stop_sequences:
        return len(text)
    buffered_length = 0
    for sequence in stop_sequences:
        for length in range(1, min(len(sequence), len(text)) + 1):
            if text.endswith(sequence[:length]):
                buffered_length = max(buffered_length, length)
    return len(text) - buffered_length


def _log_generation_failure(
    metadata: BackendMetadata,
    *,
    failure_category: str,
    retryable: bool,
) -> None:
    """Log a payload-free local generation failure."""
    _LOGGER.error(
        "backend_generation_failed",
        pipeline_stage="runtime",
        backend_name=metadata.backend_name,
        operation=BackendOperation.GENERATE.value,
        failure_category=failure_category,
        retryable=retryable,
    )
