"""Local causal-model backend boundary for runtime mechanics fixtures."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import structlog
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_lab.backends.runtime._validation import validate_integer
from inference_lab.backends.runtime.errors import (
    BackendClosedError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendRequestError,
)
from inference_lab.backends.runtime.types import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    GenerationEvent,
    GenerationRequest,
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
        """Reserve the generation boundary for the following RUN-02 sub-PR."""
        self._ensure_available()
        self._prepare_generation(request)

        async def unsupported_generation() -> AsyncIterator[GenerationEvent]:
            raise BackendRequestError()
            yield  # pragma: no cover

        return unsupported_generation()

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
