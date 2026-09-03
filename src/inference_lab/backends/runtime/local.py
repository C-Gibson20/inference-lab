"""Local causal-model backend boundary for runtime mechanics fixtures."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

import structlog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache

from inference_lab._validation import is_local_path_shape
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
_EXPECTED_ARTEFACT_DIGESTS: Final = (
    (
        "chat_template.jinja",
        "fb9aa64d5765e07f84beacb7326c5947bc7d39ab187e808dee99e2c6fa07b56d",
    ),
    (
        "config.json",
        "626c98a02b75a97adbb5fe74133c45c3331e8960a459fb13f105124f8d7ab350",
    ),
    (
        "generation_config.json",
        "2a7400bd649bc65cbd19ff7e64d57bdc223077c762bc60bde7e567fb0fb42b6d",
    ),
    (
        "model.safetensors",
        "b00e17e66e7913bf56c0d9d063f8c905fd124c8939c2ad7230a8741c708bd60a",
    ),
    (
        "special_tokens_map.json",
        "2d6145f8078c474ab0a72c09cbdce3a10cd02232862b75502544c0ec8d1c66ee",
    ),
    (
        "tokenizer.json",
        "6f4399da12154ba12fa4c43f689201cce5945c4b4ac544e92762d3004b35e794",
    ),
    (
        "tokenizer_config.json",
        "a928f2a7c43097e56ced8dee60a7de9963a0091bc2d785ff6d15ab83b9f72378",
    ),
)
_EXPECTED_ARTEFACT_NAMES: Final = frozenset(
    name for name, _ in _EXPECTED_ARTEFACT_DIGESTS
)
_TOKENISER_ARTEFACT_NAMES: Final = frozenset(
    {
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


def _manifest_digest(artefacts: tuple[tuple[str, str], ...]) -> str:
    digest = hashlib.sha256()
    for name, expected_digest in artefacts:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(expected_digest))
    return digest.hexdigest()


_MODEL_REVISION: Final = (
    f"fixture-20260902-manifest-{_manifest_digest(_EXPECTED_ARTEFACT_DIGESTS)}-"
    "prompt-v1"
)
_TOKENISER_REVISION: Final = "tokeniser-manifest-" + _manifest_digest(
    tuple(
        artefact
        for artefact in _EXPECTED_ARTEFACT_DIGESTS
        if artefact[0] in _TOKENISER_ARTEFACT_NAMES
    )
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


class CacheObservationOutcome(StrEnum):
    """Lifecycle outcome observed by the implementation mechanics harness."""

    CREATED = "created"
    REUSED = "reused"
    REJECTED = "rejected"
    CLEARED = "cleared"


@dataclass(frozen=True, slots=True)
class LocalMechanicsObservation:
    """Payload-free observation of one native prefix-cache comparison."""

    rendered_input_tokens: int
    common_prefix_tokens: int
    tokens_evaluated_without_cache: int
    tokens_evaluated_to_create_cache: int
    tokens_evaluated_after_cache: int
    cache_outcomes: tuple[CacheObservationOutcome, ...]
    logits_equivalent: bool | None
    selection_equivalent: bool | None
    backend_name: str
    model_name: str
    model_revision: str
    tokeniser_revision: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate immutable mechanics observations."""
        for field_name in (
            "rendered_input_tokens",
            "common_prefix_tokens",
            "tokens_evaluated_without_cache",
            "tokens_evaluated_to_create_cache",
            "tokens_evaluated_after_cache",
        ):
            value = getattr(self, field_name)
            validate_integer(value, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if not isinstance(self.cache_outcomes, tuple):
            raise TypeError("cache_outcomes must be a tuple")
        if not self.cache_outcomes or any(
            not isinstance(outcome, CacheObservationOutcome)
            for outcome in self.cache_outcomes
        ):
            raise ValueError("cache_outcomes must contain lifecycle outcomes")
        for field_name in ("logits_equivalent", "selection_equivalent"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a boolean or None")
        for field_name in (
            "backend_name",
            "model_name",
            "model_revision",
            "tokeniser_revision",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value:
                raise ValueError(f"{field_name} must not be empty")


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
    if not is_local_path_shape(artefact_dir):
        raise BackendRequestError()
    return Path(artefact_dir)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artefact:
        while chunk := artefact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_artefacts(artefact_dir: Path) -> None:
    present = {path.name for path in artefact_dir.iterdir()}
    if present != _EXPECTED_ARTEFACT_NAMES:
        raise ValueError("artefact inventory does not match the fixture")
    for name, expected_digest in _EXPECTED_ARTEFACT_DIGESTS:
        path = artefact_dir / name
        if not path.is_file() or _file_sha256(path) != expected_digest:
            raise ValueError("artefact checksum does not match the fixture")


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
        self._active_cache_state: Cache | None = None
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
            _verify_local_artefacts(self._artefact_dir)
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
            _verify_local_artefacts(self._artefact_dir)
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
        self._active_cache_state = None
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


class LocalMechanicsHarness:
    """Implementation-only seam for observing native prefix-cache behaviour."""

    def __init__(self, backend: LocalCausalBackend) -> None:
        """Create a mechanics harness bound to one local backend."""
        if not isinstance(backend, LocalCausalBackend):
            raise TypeError("backend must be LocalCausalBackend")
        self._backend = backend

    def observe_prefixes(
        self,
        first_request: GenerationRequest,
        second_request: GenerationRequest,
        *,
        cache_revision: str | None = None,
    ) -> LocalMechanicsObservation:
        """Compare a full prompt with a compatible cached-prefix continuation."""
        first = self._backend._prepare_generation(first_request)
        second = self._backend._prepare_generation(second_request)
        return self.observe_token_prefixes(
            first.input_ids,
            second.input_ids,
            rendered_input_tokens=second.input_tokens,
            cache_revision=cache_revision,
        )

    def observe_token_prefixes(
        self,
        first_token_ids: tuple[int, ...],
        second_token_ids: tuple[int, ...],
        *,
        rendered_input_tokens: int | None = None,
        cache_revision: str | None = None,
    ) -> LocalMechanicsObservation:
        """Observe prefix reuse for token IDs produced by the local tokeniser."""
        self._backend._ensure_available()
        self._validate_token_ids(first_token_ids, "first_token_ids")
        self._validate_token_ids(second_token_ids, "second_token_ids")
        if rendered_input_tokens is None:
            rendered_input_tokens = len(second_token_ids)
        validate_integer(rendered_input_tokens, "rendered_input_tokens")
        if rendered_input_tokens != len(second_token_ids):
            raise ValueError("rendered_input_tokens must match second token IDs")
        if cache_revision is not None and not isinstance(cache_revision, str):
            raise TypeError("cache_revision must be a string or None")

        model = self._backend._model
        if model is None:
            raise BackendNotReadyError()
        common_prefix_tokens = self._common_prefix_length(
            first_token_ids, second_token_ids
        )
        cache_compatible = (
            cache_revision is None
            or cache_revision == self._backend.metadata.model_revision
        )
        can_reuse = common_prefix_tokens > 0 and cache_compatible
        cache_outcomes: list[CacheObservationOutcome] = []
        logits_equivalent: bool | None = None
        selection_equivalent: bool | None = None
        tokens_evaluated_to_create_cache = 0
        tokens_evaluated_after_cache = 0

        try:
            with torch.inference_mode():
                cache: Cache | None = None
                cached_logits = None
                if can_reuse:
                    first_outputs = model(
                        input_ids=torch.tensor([first_token_ids], dtype=torch.long),
                        attention_mask=torch.ones(
                            (1, len(first_token_ids)), dtype=torch.long
                        ),
                        use_cache=True,
                    )
                    cache = DynamicCache.from_legacy_cache(
                        first_outputs.past_key_values
                    )
                    self._backend._active_cache_state = cache
                    cache_outcomes.append(CacheObservationOutcome.CREATED)
                    tokens_evaluated_to_create_cache = len(first_token_ids)
                    cached_logits = first_outputs.logits[:, common_prefix_tokens - 1, :]

                full_outputs = model(
                    input_ids=torch.tensor([second_token_ids], dtype=torch.long),
                    attention_mask=torch.ones(
                        (1, len(second_token_ids)), dtype=torch.long
                    ),
                    use_cache=False,
                )
                if can_reuse:
                    if cache is None or cached_logits is None:
                        raise BackendExecutionError()
                    cache.crop(common_prefix_tokens)
                    if cache.get_seq_length() != common_prefix_tokens:
                        raise BackendExecutionError()
                    suffix = second_token_ids[common_prefix_tokens:]
                    tokens_evaluated_after_cache = len(suffix)
                    if suffix:
                        cached_outputs = model(
                            input_ids=torch.tensor([suffix], dtype=torch.long),
                            attention_mask=torch.ones(
                                (1, len(second_token_ids)), dtype=torch.long
                            ),
                            past_key_values=cache,
                            use_cache=True,
                        )
                        cached_logits = cached_outputs.logits[:, -1, :]
                    cache_outcomes.append(CacheObservationOutcome.REUSED)
                    full_logits = full_outputs.logits[:, -1, :]
                    logits_equivalent = bool(
                        torch.allclose(full_logits, cached_logits, rtol=0, atol=1e-6)
                    )
                    selection_equivalent = bool(
                        torch.equal(
                            torch.argmax(full_logits, dim=-1),
                            torch.argmax(cached_logits, dim=-1),
                        )
                    )
                else:
                    cache_outcomes.append(CacheObservationOutcome.REJECTED)
        except BackendError:
            raise
        except MemoryError:
            raise BackendResourceError() from None
        except Exception:
            raise BackendExecutionError() from None
        finally:
            self._backend._active_cache_state = None
            cache_outcomes.append(CacheObservationOutcome.CLEARED)

        _LOGGER.debug(
            "backend_cache_observation_completed",
            pipeline_stage="runtime",
            backend_name=self._backend.metadata.backend_name,
            operation=BackendOperation.GENERATE.value,
            rendered_input_tokens=rendered_input_tokens,
            common_prefix_tokens=common_prefix_tokens,
            tokens_evaluated_without_cache=len(second_token_ids),
            tokens_evaluated_to_create_cache=tokens_evaluated_to_create_cache,
            tokens_evaluated_after_cache=tokens_evaluated_after_cache,
            cache_outcome=cache_outcomes[-2].value,
        )
        return LocalMechanicsObservation(
            rendered_input_tokens=rendered_input_tokens,
            common_prefix_tokens=common_prefix_tokens,
            tokens_evaluated_without_cache=len(second_token_ids),
            tokens_evaluated_to_create_cache=tokens_evaluated_to_create_cache,
            tokens_evaluated_after_cache=tokens_evaluated_after_cache,
            cache_outcomes=tuple(cache_outcomes),
            logits_equivalent=logits_equivalent,
            selection_equivalent=selection_equivalent,
            backend_name=self._backend.metadata.backend_name,
            model_name=self._backend.metadata.model_name,
            model_revision=self._backend.metadata.model_revision,
            tokeniser_revision=_TOKENISER_REVISION,
        )

    @staticmethod
    def _validate_token_ids(token_ids: tuple[int, ...], field_name: str) -> None:
        """Validate a token-ID sequence without retaining its values."""
        if not isinstance(token_ids, tuple):
            raise TypeError(f"{field_name} must be a tuple")
        if not token_ids:
            raise ValueError(f"{field_name} must not be empty")
        for token_id in token_ids:
            validate_integer(token_id, f"{field_name} values")
            if token_id < 0:
                raise ValueError(f"{field_name} values must not be negative")

    @staticmethod
    def _common_prefix_length(
        first_token_ids: tuple[int, ...], second_token_ids: tuple[int, ...]
    ) -> int:
        """Count equal token IDs from the beginning without exposing them."""
        for index, (first, second) in enumerate(
            zip(first_token_ids, second_token_ids, strict=False)
        ):
            if first != second:
                return index
        return min(len(first_token_ids), len(second_token_ids))
