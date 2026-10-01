"""Local-only BGE reranker backend for the runtime reranking contract."""

import gc
import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import structlog
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    RerankCandidate,
    RerankRequest,
    RerankResult,
    ScoredCandidate,
)

_LOGGER = structlog.get_logger(__name__)
_MODEL_NAME: Final = "BAAI/bge-reranker-v2-m3"
_MODEL_REVISION: Final = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_TOKENIZER_VOCAB_SIZE: Final = 250002
_MODEL_HIDDEN_SIZE: Final = 1024
_MAX_PAIR_TOKENS: Final = 512
_MAX_QUERY_TOKENS: Final = 256
_MAX_CANDIDATES: Final = 100
_INTERNAL_BATCH_SIZE: Final = 4
_PREPROCESSING_PROFILE_VERSION: Final = "1.0.0"
_EXPECTED_ARTEFACT_DIGESTS: Final = (
    (
        ".gitattributes",
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    (
        "README.md",
        "c887aa6dd2598f908bf0582ca7068cc816585c7a1b6a07df305b631ede0cb174",
    ),
    (
        "assets/BEIR-bge-en-v1.5.png",
        "c1e57bfc0bb87408ac0d5084acce36a26914a2ea19f2f9f3560e71be9d4daef6",
    ),
    (
        "assets/BEIR-e5-mistral.png",
        "5119f2d1b364f79eaa2954bd5efb4a7e56364ee8748571964d800f85683280c4",
    ),
    (
        "assets/CMTEB-retrieval-bge-zh-v1.5.png",
        "ce02a58c566da5f733b6928cc968865e4d472c8a276546bfce5d0f4452050ed4",
    ),
    (
        "assets/llama-index.png",
        "62c4fbdeeb44296da80bdd2f0a7a6b5e44f3492072af84fdf3ed99d01a53e596",
    ),
    (
        "assets/miracl-bge-m3.png",
        "98f40bf0ba104f3efa52ef76da1434fb776834cdc732e6b29d2550672ea2df1b",
    ),
    (
        "config.json",
        "13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a",
    ),
    (
        "model.safetensors",
        "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
    ),
    (
        "sentencepiece.bpe.model",
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
    (
        "special_tokens_map.json",
        "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    ),
    (
        "tokenizer.json",
        "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",
    ),
    (
        "tokenizer_config.json",
        "7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4",
    ),
)


class _Tokenizer(Protocol):
    """Transformers tokeniser methods used for deterministic pair encoding."""

    def __len__(self) -> int: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]: ...

    def pad(
        self,
        encoded_inputs: list[dict[str, list[int]]],
        *,
        padding: bool,
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    """Tokenised pairs and bounded truncation counts for one request."""

    input_ids: tuple[tuple[int, ...], ...]
    affected_candidate_count: int
    original_candidate_tokens: int
    retained_candidate_tokens: int


def _validate_local_path(artefact_dir: str | Path) -> Path:
    """Validate and convert the explicit local model directory."""
    if not isinstance(artefact_dir, (str, Path)):
        raise TypeError("artefact_dir must be a local path")
    if not is_local_path_shape(artefact_dir):
        raise BackendRequestError()
    return Path(artefact_dir)


def _file_sha256(path: Path) -> str:
    """Hash one artefact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as artefact:
        while chunk := artefact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_artefacts(artefact_dir: Path) -> None:
    """Require the exact pinned file inventory and digest before loading."""
    expected = dict(_EXPECTED_ARTEFACT_DIGESTS)
    actual = {
        path.relative_to(artefact_dir).as_posix(): path
        for path in artefact_dir.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    if set(actual) != set(expected) or any(
        path.is_symlink() for path in actual.values()
    ):
        raise ValueError("model artefact inventory does not match the pinned revision")
    for name, path in actual.items():
        if _file_sha256(path) != expected[name]:
            raise ValueError("model artefact digest does not match the pinned revision")


def _prepare_pairs(
    tokenizer: _Tokenizer,
    query: str,
    candidates: Sequence[RerankCandidate],
) -> _PreparedBatch:
    """Preserve each query and tail-truncate candidate tokens to the pair limit."""
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    if len(query_ids) > _MAX_QUERY_TOKENS:
        raise BackendRequestError("The reranker query exceeds its token limit.")
    special_count = tokenizer.num_special_tokens_to_add(pair=True)
    available_candidate_tokens = _MAX_PAIR_TOKENS - len(query_ids) - special_count
    if available_candidate_tokens < 1:
        raise BackendRequestError("The reranker query leaves no room for a candidate.")

    prepared: list[tuple[int, ...]] = []
    affected = 0
    original_tokens = 0
    retained_tokens = 0
    for candidate in candidates:
        candidate_ids = tokenizer.encode(
            candidate.candidate_text, add_special_tokens=False
        )
        retained_ids = candidate_ids[:available_candidate_tokens]
        pair_ids = tokenizer.build_inputs_with_special_tokens(query_ids, retained_ids)
        if len(pair_ids) > _MAX_PAIR_TOKENS:
            raise BackendExecutionError()
        prepared.append(tuple(pair_ids))
        if len(retained_ids) < len(candidate_ids):
            affected += 1
            original_tokens += len(candidate_ids)
            retained_tokens += len(retained_ids)
    return _PreparedBatch(tuple(prepared), affected, original_tokens, retained_tokens)


class LocalRerankerBackend:
    """Score up to 100 candidates using the pinned local BGE-M3 reranker."""

    def __init__(self, artefact_dir: str | Path) -> None:
        """Create an unloaded backend bound to one explicit local artefact path."""
        self._artefact_dir = _validate_local_path(artefact_dir)
        self._model: torch.nn.Module | None = None
        self._tokenizer: _Tokenizer | None = None
        self._is_ready = False
        self._is_closed = False

    @property
    def metadata(self) -> BackendMetadata:
        """Return the pinned model identity and supported reranking capability."""
        return BackendMetadata(
            backend_name="transformers-local",
            model_name=_MODEL_NAME,
            model_revision=_MODEL_REVISION,
            preprocessing_profile_version=_PREPROCESSING_PROFILE_VERSION,
            supported_operations=(BackendOperation.RERANK,),
            capabilities=(BackendCapability.BATCHING,),
        )

    @property
    def is_ready(self) -> bool:
        """Return whether the verified model is ready for requests."""
        return self._is_ready

    @property
    def is_closed(self) -> bool:
        """Return whether this backend instance has been permanently closed."""
        return self._is_closed

    def load(self) -> None:
        """Verify and load the pinned model from local files only."""
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
            model = cast(
                torch.nn.Module,
                AutoModelForSequenceClassification.from_pretrained(
                    self._artefact_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=torch.float32,
                ),
            )
            model.to("cpu")
            model.eval()
            config = getattr(model, "config", None)
            if (
                getattr(config, "model_type", None) != "xlm-roberta"
                or getattr(config, "hidden_size", None) != _MODEL_HIDDEN_SIZE
                or getattr(config, "num_labels", None) != 1
            ):
                raise ValueError(
                    "model configuration does not match the pinned revision"
                )
            if len(tokenizer) != _TOKENIZER_VOCAB_SIZE:
                raise ValueError("tokeniser does not match the pinned revision")
            _verify_local_artefacts(self._artefact_dir)
        except (MemoryError, torch.OutOfMemoryError):
            self._model = None
            self._tokenizer = None
            self._is_ready = False
            _LOGGER.error(
                "backend_load_failed",
                pipeline_stage="runtime",
                backend_name=self.metadata.backend_name,
                model_name=_MODEL_NAME,
                operation=BackendOperation.RERANK.value,
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
                model_name=_MODEL_NAME,
                operation=BackendOperation.RERANK.value,
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
            model_name=_MODEL_NAME,
            operation=BackendOperation.RERANK.value,
            model_revision=_MODEL_REVISION,
            inference_dtype="float32",
            device_type="cpu",
        )

    def close(self) -> None:
        """Release model resources and permanently close this backend."""
        self._model = None
        self._tokenizer = None
        self._is_ready = False
        if self._is_closed:
            return
        self._is_closed = True
        gc.collect()
        _LOGGER.info(
            "backend_close_completed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            model_name=_MODEL_NAME,
            operation=BackendOperation.RERANK.value,
        )

    def measure_memory(self) -> tuple[MemoryMeasurement, ...]:
        """Report the inspectable reranker weight allocation."""
        measurements = PyTorchMemoryCalibrationAdapter(
            self._model,
            component_id="reranker",
            component_role=MemoryComponentRole.RERANKER,
        ).measure_memory()
        return tuple(item for item in measurements if item.category == "model_weights")

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Score every requested candidate, then apply stable ordering and top-k."""
        self._ensure_available()
        if not isinstance(request, RerankRequest):
            raise TypeError("request must be a RerankRequest")
        if len(request.candidates) > _MAX_CANDIDATES:
            self._log_failure("request", retryable=False)
            raise BackendRequestError("The reranker candidate limit is 100.")
        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            raise BackendNotReadyError()

        started = time.perf_counter_ns()
        try:
            prepared = _prepare_pairs(tokenizer, request.query, request.candidates)
            scores: list[tuple[str, float]] = []
            with torch.inference_mode():
                for start in range(0, len(request.candidates), _INTERNAL_BATCH_SIZE):
                    candidates = request.candidates[
                        start : start + _INTERNAL_BATCH_SIZE
                    ]
                    batch_ids = prepared.input_ids[start : start + _INTERNAL_BATCH_SIZE]
                    padded = tokenizer.pad(
                        [{"input_ids": list(input_ids)} for input_ids in batch_ids],
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = {key: value.to("cpu") for key, value in padded.items()}
                    logits = model(**inputs, return_dict=True).logits
                    values = logits.reshape(-1).float().tolist()
                    if len(values) != len(candidates) or any(
                        not math.isfinite(float(value)) for value in values
                    ):
                        raise BackendExecutionError()
                    scores.extend(
                        (candidate.candidate_id, float(score))
                        for candidate, score in zip(candidates, values, strict=True)
                    )
            if len(scores) != len(request.candidates):
                raise BackendExecutionError()
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

        if prepared.affected_candidate_count:
            _LOGGER.warning(
                "reranker_candidates_truncated",
                pipeline_stage="runtime",
                backend_name=self.metadata.backend_name,
                model_name=_MODEL_NAME,
                affected_candidate_count=prepared.affected_candidate_count,
                original_candidate_tokens=prepared.original_candidate_tokens,
                retained_candidate_tokens=prepared.retained_candidate_tokens,
                pair_token_limit=_MAX_PAIR_TOKENS,
            )
        ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
        if request.top_k is not None:
            ranked = ranked[: request.top_k]
        result = tuple(
            ScoredCandidate(candidate_id=candidate_id, score=score, rank=rank)
            for rank, (candidate_id, score) in enumerate(ranked, start=1)
        )
        _LOGGER.debug(
            "backend_operation_completed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            model_name=_MODEL_NAME,
            operation=BackendOperation.RERANK.value,
            candidate_count=len(request.candidates),
            result_count=len(result),
            duration_ms=(time.perf_counter_ns() - started) / 1_000_000,
            preprocessing_profile_version=_PREPROCESSING_PROFILE_VERSION,
        )
        return RerankResult(scored_candidates=result, model_metadata=self.metadata)

    def _ensure_available(self) -> None:
        """Raise a lifecycle-specific error before accessing model state."""
        if self._is_closed:
            raise BackendClosedError()
        if not self._is_ready:
            raise BackendNotReadyError()

    def _log_failure(self, category: str, *, retryable: bool) -> None:
        """Emit a bounded error record without request or dependency data."""
        _LOGGER.error(
            "backend_operation_failed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            model_name=_MODEL_NAME,
            operation=BackendOperation.RERANK.value,
            failure_category=category,
            retryable=retryable,
        )
