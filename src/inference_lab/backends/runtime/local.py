"""Local causal-model backend boundary for runtime mechanics fixtures."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import structlog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from inference_lab.backends.runtime._validation import validate_integer
from inference_lab.backends.runtime.errors import (
    BackendClosedError,
    BackendError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendRequestError,
    BackendResourceError,
)
from inference_lab.backends.runtime.streaming import validate_generation_stream
from inference_lab.backends.runtime.types import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    GenerationEvent,
    GenerationEventType,
    GenerationRequest,
    GenerationUsage,
)

_LOGGER = structlog.get_logger(__name__)
_REQUIRED_ARTEFACTS = frozenset(
    {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
)
_MODEL_REVISION = (
    "fixture-20260902-"
    "model-b00e17e66e7913bf56c0d9d063f8c905fd124c8939c2ad7230a8741c708bd60a-"
    "tokeniser-6f4399da12154ba12fa4c43f689201cce5945c4b4ac544e92762d3004b35e794-"
    "template-fb9aa64d5765e07f84beacb7326c5947bc7d39ab187e808dee99e2c6fa07b56d-"
    "prompt-v1"
)


class _Tokeniser(Protocol):
    """Minimal tokeniser surface required by the streaming implementation."""

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


def _validate_local_path(artefact_dir: str | Path) -> Path:
    """Validate and convert an explicit local artefact directory."""
    if not isinstance(artefact_dir, (str, Path)):
        raise TypeError("artefact_dir must be a local path")
    raw_path = str(artefact_dir)
    if not raw_path or "\x00" in raw_path:
        raise BackendRequestError()
    parsed = urlsplit(raw_path)
    if parsed.scheme or parsed.netloc or raw_path.startswith(("//", "\\\\")):
        raise BackendRequestError()
    return Path(artefact_dir)


class LocalCausalBackend:
    """Generation-only boundary for a local causal model fixture."""

    def __init__(
        self, artefact_dir: str | Path, context_limit_tokens: int = 8192
    ) -> None:
        """Create an unloaded backend for one local artefact directory."""
        self._artefact_dir = _validate_local_path(artefact_dir)
        validate_integer(context_limit_tokens, "context_limit_tokens")
        if context_limit_tokens <= 0:
            raise ValueError("context_limit_tokens must be positive")
        self._context_limit_tokens = context_limit_tokens
        self._effective_context_limit_tokens: int | None = None
        self._model = None
        self._tokeniser = None
        self._is_ready = False
        self._is_closed = False

    @property
    def metadata(self) -> BackendMetadata:
        """Return stable identity and generation-only capabilities."""
        return BackendMetadata(
            backend_name="transformers-local",
            model_name="run02-causal-fixture",
            model_revision=_MODEL_REVISION,
            supported_operations=(BackendOperation.GENERATE,),
            capabilities=(
                BackendCapability.STREAMING,
                BackendCapability.CANCELLATION,
                BackendCapability.TOKEN_USAGE,
            ),
        )

    @property
    def is_ready(self) -> bool:
        """Return whether model resources are loaded and usable."""
        return self._is_ready

    @property
    def is_closed(self) -> bool:
        """Return whether this backend has been permanently closed."""
        return self._is_closed

    def load(self) -> None:
        """Load and validate the local model and tokeniser exactly once."""
        if self._is_closed:
            raise BackendClosedError()
        if self._is_ready:
            return

        try:
            if not self._artefact_dir.is_dir():
                raise ValueError("artefact directory is unavailable")
            present = {path.name for path in self._artefact_dir.iterdir()}
            if not _REQUIRED_ARTEFACTS.issubset(present):
                raise ValueError("required artefacts are unavailable")
            tokeniser = AutoTokenizer.from_pretrained(
                self._artefact_dir,
                local_files_only=True,
                use_fast=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self._artefact_dir,
                local_files_only=True,
            )
            model.eval()
            if model.config.vocab_size != len(tokeniser):
                raise ValueError("model and tokeniser vocabularies are incompatible")
            model_context_limit = getattr(model.config, "max_position_embeddings", None)
            if isinstance(model_context_limit, bool) or not isinstance(
                model_context_limit, int
            ):
                raise ValueError("model context limit must be an integer")
            if model_context_limit <= 0:
                raise ValueError("model context limit must be positive")
        except Exception:
            self._model = None
            self._tokeniser = None
            self._is_ready = False
            _LOGGER.error(
                "backend_load_failed",
                pipeline_stage="runtime",
                backend_name=self.metadata.backend_name,
                operation=BackendOperation.GENERATE.value,
                failure_category="load",
                retryable=False,
            )
            raise BackendExecutionError() from None

        self._model = model
        self._tokeniser = tokeniser
        self._effective_context_limit_tokens = min(
            self._context_limit_tokens, model_context_limit
        )
        self._is_ready = True
        _LOGGER.info(
            "backend_load_completed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            model_name=self.metadata.model_name,
            operation=BackendOperation.GENERATE.value,
        )

    def close(self) -> None:
        """Release model resources and permanently close this backend."""
        self._model = None
        self._tokeniser = None
        self._effective_context_limit_tokens = None
        self._is_ready = False
        if self._is_closed:
            return
        self._is_closed = True
        _LOGGER.info(
            "backend_close_completed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            operation=BackendOperation.GENERATE.value,
        )

    def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationEvent]:
        """Return a validated asynchronous stream for one generation request."""
        self._ensure_available()
        prepared = self._prepare_generation(request)
        _LOGGER.debug(
            "backend_generation_started",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            operation=BackendOperation.GENERATE.value,
            input_tokens=prepared.input_tokens,
            output_token_budget=prepared.available_output_tokens,
        )
        return validate_generation_stream(self._generate_events(request, prepared))

    async def _generate_events(
        self,
        request: GenerationRequest,
        prepared: _PreparedGeneration,
    ) -> AsyncIterator[GenerationEvent]:
        """Generate greedy tokens and expose only validated stream events."""
        model = self._model
        tokeniser = self._tokeniser
        if model is None or tokeniser is None:
            raise BackendNotReadyError()

        generated_ids: list[int] = []
        emitted_text = ""
        sequence_number = 0
        finish_reason = "length"
        attention_mask = torch.ones((1, prepared.input_tokens), dtype=torch.long)
        input_ids = torch.tensor([prepared.input_ids], dtype=torch.long)
        outputs = None
        past_key_values = None
        output_tokens = 0

        try:
            with torch.inference_mode():
                await asyncio.sleep(0)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                past_key_values = DynamicCache.from_legacy_cache(
                    outputs.past_key_values
                )
                for _ in range(prepared.available_output_tokens):
                    self._ensure_available()
                    next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1))
                    generated_ids.append(next_token)
                    finish_reason = (
                        "stop" if self._is_end_token(model, next_token) else "length"
                    )

                    decoded = self._decode(tokeniser, generated_ids)
                    stop_position = self._stop_position(
                        decoded, request.options.stop_sequences
                    )
                    if stop_position is not None:
                        visible_text = decoded[:stop_position]
                        finish_reason = "stop"
                    else:
                        visible_text = decoded

                    safe_length = self._safe_decoded_length(
                        visible_text,
                        request.options.stop_sequences,
                        stop_position is not None,
                    )
                    safe_text = visible_text[:safe_length]
                    delta = safe_text[len(emitted_text) :]
                    if delta:
                        emitted_text = safe_text
                        yield GenerationEvent(
                            event_type=GenerationEventType.DELTA,
                            sequence_number=sequence_number,
                            text_delta=delta,
                        )
                        sequence_number += 1

                    if stop_position is not None or finish_reason == "stop":
                        break

                    await asyncio.sleep(0)
                    input_ids = torch.tensor([[next_token]], dtype=torch.long)
                    attention_mask = torch.cat(
                        (attention_mask, torch.ones((1, 1), dtype=torch.long)),
                        dim=1,
                    )
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values

                if finish_reason == "length":
                    decoded = self._decode(tokeniser, generated_ids)
                    stop_position = self._stop_position(
                        decoded, request.options.stop_sequences
                    )
                    visible_text = (
                        decoded[:stop_position]
                        if stop_position is not None
                        else decoded
                    )
                    delta = visible_text[len(emitted_text) :]
                    if delta:
                        emitted_text = visible_text
                        yield GenerationEvent(
                            event_type=GenerationEventType.DELTA,
                            sequence_number=sequence_number,
                            text_delta=delta,
                        )
                        sequence_number += 1
        except asyncio.CancelledError:
            raise
        except BackendError:
            raise
        except MemoryError:
            _LOGGER.error(
                "backend_generation_failed",
                pipeline_stage="runtime",
                backend_name=self.metadata.backend_name,
                operation=BackendOperation.GENERATE.value,
                failure_category="resource",
                retryable=True,
            )
            raise BackendResourceError() from None
        except Exception:
            _LOGGER.error(
                "backend_generation_failed",
                pipeline_stage="runtime",
                backend_name=self.metadata.backend_name,
                operation=BackendOperation.GENERATE.value,
                failure_category="execution",
                retryable=False,
            )
            raise BackendExecutionError() from None
        finally:
            output_tokens = len(generated_ids)
            del past_key_values
            del outputs
            del input_ids
            del attention_mask
            generated_ids.clear()

        yield GenerationEvent(
            event_type=GenerationEventType.COMPLETED,
            sequence_number=sequence_number,
            finish_reason=finish_reason,
            usage=GenerationUsage(
                input_tokens=prepared.input_tokens,
                output_tokens=output_tokens,
                total_tokens=prepared.input_tokens + output_tokens,
            ),
        )

    @staticmethod
    def _is_end_token(model: object, token_id: int) -> bool:
        """Return whether a generated token is one of the model end tokens."""
        end_token_ids = getattr(getattr(model, "config", None), "eos_token_id", None)
        if isinstance(end_token_ids, int):
            return token_id == end_token_ids
        return isinstance(end_token_ids, list) and token_id in end_token_ids

    @staticmethod
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

    @staticmethod
    def _stop_position(text: str, stop_sequences: tuple[str, ...]) -> int | None:
        """Return the earliest matched stop-sequence position, if any."""
        positions = tuple(
            position
            for sequence in stop_sequences
            if (position := text.find(sequence)) >= 0
        )
        return min(positions) if positions else None

    @staticmethod
    def _safe_decoded_length(
        text: str,
        stop_sequences: tuple[str, ...],
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

    @property
    def effective_context_limit_tokens(self) -> int:
        """Return the effective context limit after loading the model."""
        self._ensure_available()
        if self._effective_context_limit_tokens is None:
            raise BackendExecutionError()
        return self._effective_context_limit_tokens

    def _render_prompt(self, request: GenerationRequest) -> str:
        """Render messages through the fixture's versioned chat template."""
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        if self._tokeniser is None:
            raise BackendNotReadyError()
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]
        try:
            rendered = self._tokeniser.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            raise BackendRequestError() from None
        if not isinstance(rendered, str) or not rendered:
            raise BackendExecutionError()
        return rendered

    def _prepare_generation(self, request: GenerationRequest) -> _PreparedGeneration:
        """Render, tokenise, and budget a validated generation request."""
        self._ensure_available()
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        options = request.options
        if (
            options.temperature is not None
            or options.top_p is not None
            or options.seed is not None
        ):
            raise BackendRequestError()

        rendered_prompt = self._render_prompt(request)
        if self._tokeniser is None:
            raise BackendNotReadyError()
        try:
            encoded = self._tokeniser(
                rendered_prompt,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            input_ids = tuple(encoded["input_ids"])
        except Exception:
            raise BackendRequestError() from None
        input_tokens = len(input_ids)
        context_limit = self.effective_context_limit_tokens
        if input_tokens >= context_limit:
            raise BackendRequestError()
        available_output_tokens = context_limit - input_tokens
        if options.maximum_output_tokens is not None:
            available_output_tokens = min(
                available_output_tokens, options.maximum_output_tokens
            )
        if available_output_tokens <= 0:
            raise BackendRequestError()
        return _PreparedGeneration(
            rendered_prompt=rendered_prompt,
            input_ids=input_ids,
            input_tokens=input_tokens,
            available_output_tokens=available_output_tokens,
        )

    def _ensure_available(self) -> None:
        """Raise the lifecycle error appropriate to the current state."""
        if self._is_closed:
            raise BackendClosedError()
        if not self._is_ready:
            raise BackendNotReadyError()
