"""Local-only Qwen3 embedding backend for scientific literature retrieval."""

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Final, Protocol, cast

import structlog
import torch
from transformers import AutoModel, AutoTokenizer

from inference_lab._validation import is_local_path_shape
from inference_lab.backends.runtime._memory_values import (
    MemoryComponentRole,
    MemoryMeasurement,
)
from inference_lab.backends.runtime.errors import (
    BackendClosedError,
    BackendError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendRequestError,
    BackendResourceError,
)
from inference_lab.backends.runtime.memory_pytorch import (
    PyTorchMemoryCalibrationAdapter,
)
from inference_lab.backends.runtime.types import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingProfile,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingUsage,
)

_LOGGER = structlog.get_logger(__name__)
_MODEL_NAME: Final = "Qwen/Qwen3-Embedding-0.6B"
_MODEL_REVISION: Final = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
_EMBEDDING_DIMENSIONS: Final = 1024
_MIN_EMBEDDING_DIMENSIONS: Final = 32
_TOKENIZER_VOCAB_SIZE: Final = 151669
_MAX_INPUT_TOKENS: Final = 4096
_INTERNAL_BATCH_SIZE: Final = 2
_QUERY_PREFIX: Final = (
    "Instruct: Given a scientific literature search query, retrieve passages "
    "that answer the query\nQuery: "
)
_DOCUMENT_PREFIX: Final = ""
_ARTEFACT_DIGESTS: Final = (
    (
        "config.json",
        "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
    ),
    (
        "merges.txt",
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    (
        "model.safetensors",
        "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    ),
    (
        "tokenizer.json",
        "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    ),
    (
        "tokenizer_config.json",
        "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    ),
    (
        "vocab.json",
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)


class _Tokenizer(Protocol):
    """Subset of the Transformers tokeniser API used by this backend."""

    pad_token_id: int | None
    padding_side: str

    def __len__(self) -> int: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...

    def build_inputs_with_special_tokens(self, token_ids_0: list[int]) -> list[int]: ...

    def pad(
        self,
        encoded_inputs: list[dict[str, list[int]]],
        *,
        padding: bool,
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]: ...


class _ModelOutput(Protocol):
    """Model output fields required for last-token pooling."""

    last_hidden_state: torch.Tensor


class _EmbeddingModel(Protocol):
    """Subset of a Transformers encoder model used during inference."""

    config: object

    def __call__(self, **inputs: torch.Tensor) -> _ModelOutput: ...

    def eval(self) -> object: ...

    def to(self, device: str) -> object: ...


def _validate_local_path(artefact_dir: str | Path) -> Path:
    """Validate and convert the explicit local artefact directory."""
    if not isinstance(artefact_dir, (str, Path)):
        raise TypeError("artefact_dir must be a local path")
    if not is_local_path_shape(artefact_dir):
        raise BackendRequestError()
    return Path(artefact_dir)


def _file_sha256(path: Path) -> str:
    """Return a streamed SHA-256 digest without loading an artefact in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as artefact:
        while chunk := artefact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_artefacts(artefact_dir: Path) -> None:
    """Verify the pinned model and tokeniser files used by this backend."""
    for name, expected_digest in _ARTEFACT_DIGESTS:
        path = artefact_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("required model artefact is unavailable")
        if _file_sha256(path) != expected_digest:
            raise ValueError("model artefact identity does not match")


def _build_profile(*, normalise: bool) -> EmbeddingProfile:
    """Return the complete path-free identity for one vector request."""
    return EmbeddingProfile(
        profile_version="1.0.0",
        tokenizer_id=_MODEL_NAME,
        tokenizer_revision=_MODEL_REVISION,
        query_prefix=_QUERY_PREFIX,
        document_prefix=_DOCUMENT_PREFIX,
        pooling_strategy="last-token-v1",
        input_normalisation="identity-v1",
        max_input_tokens=_MAX_INPUT_TOKENS,
        over_limit_policy="truncate_tail",
        l2_normalised=normalise,
        inference_dtype="float32",
        device_type="cpu",
    )


class QwenEmbeddingBackend:
    """Generate deterministic Qwen3 embedding vectors from pinned local files."""

    def __init__(self, artefact_dir: str | Path) -> None:
        """Create an unloaded backend bound to one explicit artefact directory."""
        self._artefact_dir = _validate_local_path(artefact_dir)
        self._model: _EmbeddingModel | None = None
        self._tokenizer: _Tokenizer | None = None
        self._is_ready = False
        self._is_closed = False

    @property
    def metadata(self) -> BackendMetadata:
        """Return pinned model identity and supported embedding capabilities."""
        return BackendMetadata(
            backend_name="transformers-local",
            model_name=_MODEL_NAME,
            model_revision=_MODEL_REVISION,
            supported_operations=(BackendOperation.EMBED,),
            capabilities=(
                BackendCapability.BATCHING,
                BackendCapability.TOKEN_USAGE,
            ),
            embedding_dimensions=_EMBEDDING_DIMENSIONS,
        )

    @property
    def is_ready(self) -> bool:
        """Return whether model and tokeniser resources are usable."""
        return self._is_ready

    @property
    def is_closed(self) -> bool:
        """Return whether this backend has been permanently closed."""
        return self._is_closed

    @property
    def output_profile(self) -> EmbeddingProfile:
        """Return the default path-free profile used for dense indexing."""
        return _build_profile(normalise=False)

    def load(self) -> None:
        """Load the verified model and tokeniser without registry access."""
        if self._is_closed:
            raise BackendClosedError()
        if self._is_ready:
            return

        try:
            if not self._artefact_dir.is_dir():
                raise ValueError("model directory is unavailable")
            _verify_local_artefacts(self._artefact_dir)
            tokenizer = cast(
                _Tokenizer,
                AutoTokenizer.from_pretrained(
                    self._artefact_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_fast=True,
                ),
            )
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                raise ValueError("tokeniser has no padding token")
            model = cast(
                _EmbeddingModel,
                AutoModel.from_pretrained(
                    self._artefact_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=torch.float32,
                ),
            )
            model.eval()
            model.to("cpu")
            model_type = getattr(model.config, "model_type", None)
            hidden_size = getattr(model.config, "hidden_size", None)
            if model_type != "qwen3" or hidden_size != _EMBEDDING_DIMENSIONS:
                raise ValueError("model configuration does not match the pinned model")
            if len(tokenizer) != _TOKENIZER_VOCAB_SIZE:
                raise ValueError("tokeniser identity is invalid")
            _verify_local_artefacts(self._artefact_dir)
        except (MemoryError, torch.OutOfMemoryError):
            self._model = None
            self._tokenizer = None
            self._is_ready = False
            _LOGGER.error(
                "backend_load_failed",
                pipeline_stage="runtime",
                backend_name=self.metadata.backend_name,
                model_name=self.metadata.model_name,
                operation=BackendOperation.EMBED.value,
                failure_category="resource",
                retryable=True,
            )
            raise BackendResourceError() from None
        except Exception:
            self._model = None
            self._tokenizer = None
            self._is_ready = False
            _LOGGER.error(
                "backend_load_failed",
                pipeline_stage="runtime",
                backend_name=self.metadata.backend_name,
                model_name=self.metadata.model_name,
                operation=BackendOperation.EMBED.value,
                failure_category="load",
                retryable=False,
            )
            raise BackendExecutionError() from None

        self._model = model
        self._tokenizer = tokenizer
        self._is_ready = True
        _LOGGER.info(
            "backend_load_completed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            model_name=self.metadata.model_name,
            operation=BackendOperation.EMBED.value,
            device_type="cpu",
            inference_dtype="float32",
        )

    def close(self) -> None:
        """Release loaded resources and permanently close this backend."""
        self._model = None
        self._tokenizer = None
        self._is_ready = False
        if self._is_closed:
            return
        self._is_closed = True
        _LOGGER.info(
            "backend_close_completed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            model_name=self.metadata.model_name,
            operation=BackendOperation.EMBED.value,
        )

    def measure_memory(self) -> tuple[MemoryMeasurement, ...]:
        """Return inspectable model tensor memory for the optional RUN-03 contract."""
        adapter = PyTorchMemoryCalibrationAdapter(
            cast(torch.nn.Module | None, self._model),
            component_id="embedder",
            component_role=MemoryComponentRole.EMBEDDER,
        )
        return tuple(
            measurement
            for measurement in adapter.measure_memory()
            if measurement.category == "model_weights"
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed request texts in order using fixed-size local inference batches."""
        self._ensure_available()
        if not isinstance(request, EmbeddingRequest):
            raise TypeError("request must be an EmbeddingRequest")
        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            raise BackendNotReadyError()
        if request.options.dimensions is not None and not (
            _MIN_EMBEDDING_DIMENSIONS
            <= request.options.dimensions
            <= _EMBEDDING_DIMENSIONS
        ):
            self._log_failure("request", retryable=False)
            raise BackendRequestError()
        if any(not text.strip() for text in request.input_texts):
            self._log_failure("request", retryable=False)
            raise BackendRequestError()

        try:
            prefix = _QUERY_PREFIX if request.purpose == "query" else _DOCUMENT_PREFIX
            vectors: list[tuple[float, ...]] = []
            total_input_tokens = 0
            affected_input_count = 0
            original_token_count = 0
            retained_token_count = 0
            for start in range(0, len(request.input_texts), _INTERNAL_BATCH_SIZE):
                prepared, token_counts, truncation = self._prepare_inputs(
                    tokenizer,
                    request.input_texts[start : start + _INTERNAL_BATCH_SIZE],
                    prefix=prefix,
                )
                total_input_tokens += sum(token_counts)
                affected_input_count += truncation[0]
                original_token_count += truncation[1]
                retained_token_count += truncation[2]
                vectors.extend(
                    self._run_batches(
                        model,
                        tokenizer,
                        prepared,
                        dimensions=request.options.dimensions or _EMBEDDING_DIMENSIONS,
                        normalise=request.options.normalise,
                    )
                )
            if affected_input_count:
                _LOGGER.warning(
                    "embedding_input_truncated",
                    pipeline_stage="runtime",
                    backend_name=self.metadata.backend_name,
                    model_name=self.metadata.model_name,
                    affected_input_count=affected_input_count,
                    original_token_count=original_token_count,
                    retained_token_count=retained_token_count,
                    token_limit=_MAX_INPUT_TOKENS,
                )
        except BackendRequestError:
            self._log_failure("request", retryable=False)
            raise
        except BackendError:
            self._log_failure("execution", retryable=False)
            raise
        except (MemoryError, torch.OutOfMemoryError):
            self._log_failure("resource", retryable=True)
            raise BackendResourceError() from None
        except Exception:
            self._log_failure("execution", retryable=False)
            raise BackendExecutionError() from None

        _LOGGER.debug(
            "backend_operation_completed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            operation=BackendOperation.EMBED.value,
            input_count=len(vectors),
            input_token_count=total_input_tokens,
            purpose=request.purpose,
        )
        return EmbeddingResult(
            vectors=tuple(vectors),
            model_metadata=replace(
                self.metadata,
                embedding_dimensions=len(vectors[0]),
            ),
            embedding_profile=_build_profile(normalise=request.options.normalise),
            usage=EmbeddingUsage(
                input_tokens=total_input_tokens,
                total_tokens=total_input_tokens,
            ),
        )

    def _prepare_inputs(
        self,
        tokenizer: _Tokenizer,
        input_texts: tuple[str, ...],
        *,
        prefix: str,
    ) -> tuple[list[list[int]], list[int], tuple[int, int, int]]:
        """Tokenise inputs and truncate only text tokens after each prefix."""
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        special_count = tokenizer.num_special_tokens_to_add(pair=False)
        available_text_tokens = _MAX_INPUT_TOKENS - special_count - len(prefix_ids)
        if available_text_tokens < 0:
            raise BackendExecutionError()

        prepared: list[list[int]] = []
        token_counts: list[int] = []
        affected_input_count = 0
        original_token_count = 0
        retained_token_count = 0
        for text in input_texts:
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            retained_text_ids = text_ids[:available_text_tokens]
            input_ids = tokenizer.build_inputs_with_special_tokens(
                prefix_ids + retained_text_ids
            )
            if len(input_ids) > _MAX_INPUT_TOKENS:
                raise BackendExecutionError()
            prepared.append(input_ids)
            token_counts.append(len(input_ids))
            if len(retained_text_ids) < len(text_ids):
                affected_input_count += 1
                original_token_count += len(prefix_ids) + len(text_ids) + special_count
                retained_token_count += len(input_ids)
        return (
            prepared,
            token_counts,
            (affected_input_count, original_token_count, retained_token_count),
        )

    def _run_batches(
        self,
        model: _EmbeddingModel,
        tokenizer: _Tokenizer,
        prepared: list[list[int]],
        *,
        dimensions: int,
        normalise: bool,
    ) -> tuple[tuple[float, ...], ...]:
        """Run bounded inference batches and return finite vectors in order."""
        output: list[tuple[float, ...]] = []
        with torch.inference_mode():
            for start in range(0, len(prepared), _INTERNAL_BATCH_SIZE):
                batch_inputs = [
                    {"input_ids": token_ids}
                    for token_ids in prepared[start : start + _INTERNAL_BATCH_SIZE]
                ]
                padded = tokenizer.pad(
                    batch_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = padded["input_ids"].to("cpu")
                attention_mask = padded["attention_mask"].to("cpu")
                if not bool(attention_mask[:, -1].all()):
                    raise BackendExecutionError()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                if (
                    hidden.ndim != 3
                    or hidden.shape[0] != len(batch_inputs)
                    or hidden.shape[1] != input_ids.shape[1]
                    or hidden.shape[2] != _EMBEDDING_DIMENSIONS
                ):
                    raise BackendExecutionError()
                vectors = hidden[:, -1, :dimensions]
                if normalise:
                    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
                if not bool(torch.isfinite(vectors).all()):
                    raise BackendExecutionError()
                output.extend(
                    tuple(float(value) for value in vector)
                    for vector in vectors.to(device="cpu", dtype=torch.float32).tolist()
                )
        if len(output) != len(prepared):
            raise BackendExecutionError()
        return tuple(output)

    def _ensure_available(self) -> None:
        """Raise the lifecycle-specific error when inference is unavailable."""
        if self._is_closed:
            raise BackendClosedError()
        if not self._is_ready:
            raise BackendNotReadyError()

    def _log_failure(self, category: str, *, retryable: bool) -> None:
        """Emit an error record without request text, paths, or exceptions."""
        _LOGGER.error(
            "backend_operation_failed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            model_name=self.metadata.model_name,
            operation=BackendOperation.EMBED.value,
            failure_category=category,
            retryable=retryable,
        )
