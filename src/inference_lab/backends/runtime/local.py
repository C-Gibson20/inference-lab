"""Lifecycle and request boundary for the local causal-model fixture."""

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final, cast

import structlog
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_lab._validation import is_local_path_shape
from inference_lab.backends.runtime._local_generation import (
    _CausalModel,
    _generate_local_events,
    _PreparedGeneration,
    _Tokeniser,
)
from inference_lab.backends.runtime._validation import validate_integer
from inference_lab.backends.runtime.errors import (
    BackendClosedError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendRequestError,
)
from inference_lab.backends.runtime.streaming import validate_generation_stream
from inference_lab.backends.runtime.types import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    GenerationEvent,
    GenerationRequest,
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
        """Delegate stream execution with resources checked at iteration time."""
        self._ensure_available()
        model = self._model
        tokeniser = self._tokeniser
        if model is None or tokeniser is None:
            raise BackendNotReadyError()
        async for event in _generate_local_events(
            request,
            prepared,
            model=cast(_CausalModel, model),
            tokeniser=cast(_Tokeniser, tokeniser),
            ensure_available=self._ensure_available,
            metadata=self.metadata,
        ):
            yield event

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
