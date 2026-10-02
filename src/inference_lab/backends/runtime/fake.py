"""Deterministic fake runtime backend for contract and integration tests."""

import asyncio
import math
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum

import structlog

from inference_lab.backends.runtime._validation import validate_finite_number
from inference_lab.backends.runtime.errors import (
    BackendClosedError,
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
    EmbeddingProfile,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingUsage,
    GenerationEvent,
    GenerationEventType,
    GenerationRequest,
    GenerationUsage,
    RerankRequest,
    RerankResult,
    ScoredCandidate,
)

_LOGGER = structlog.get_logger(__name__)


class FakeFailureMode(StrEnum):
    """Failure category that the fake can inject."""

    REQUEST = "request"
    EXECUTION = "execution"
    RESOURCE = "resource"


SyncSleeper = Callable[[float], None]
AsyncSleeper = Callable[[float], Awaitable[None]]


def _default_metadata() -> BackendMetadata:
    return BackendMetadata(
        backend_name="fake",
        model_name="fake-model",
        model_revision="test",
        preprocessing_profile_version="1.0.0",
        supported_operations=(
            BackendOperation.EMBED,
            BackendOperation.RERANK,
            BackendOperation.GENERATE,
        ),
        capabilities=(
            BackendCapability.BATCHING,
            BackendCapability.STREAMING,
            BackendCapability.CANCELLATION,
            BackendCapability.TOKEN_USAGE,
        ),
        embedding_dimensions=3,
    )


@dataclass(frozen=True, slots=True)
class FakeBackendConfig:
    """Deterministic behaviour controls for :class:`FakeBackend`."""

    metadata: BackendMetadata = field(default_factory=_default_metadata)
    initial_delay_seconds: float = 0.0
    per_item_delay_seconds: float = 0.0
    per_chunk_delay_seconds: float = 0.0
    generation_chunks: tuple[str, ...] = ("fake", " response")
    embedding_vectors: tuple[tuple[float, ...], ...] | None = None
    rerank_scores: tuple[tuple[str, float], ...] = ()
    failure_mode: FakeFailureMode | None = None
    fail_operation: BackendOperation | None = None
    fail_after_generation_chunks: int | None = None
    sync_sleeper: SyncSleeper = time.sleep
    async_sleeper: AsyncSleeper = asyncio.sleep

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate all fake data, timing, and failure controls."""
        if not isinstance(self.metadata, BackendMetadata):
            raise TypeError("metadata must be BackendMetadata")
        for field_name, delay in (
            ("initial_delay_seconds", self.initial_delay_seconds),
            ("per_item_delay_seconds", self.per_item_delay_seconds),
            ("per_chunk_delay_seconds", self.per_chunk_delay_seconds),
        ):
            validate_finite_number(delay, field_name)
            if delay < 0:
                raise ValueError(f"{field_name} must not be negative")

        if not isinstance(self.generation_chunks, tuple):
            raise TypeError("generation_chunks must be a tuple")
        if any(not isinstance(chunk, str) for chunk in self.generation_chunks):
            raise TypeError("generation_chunks must contain strings")
        if any(not chunk for chunk in self.generation_chunks):
            raise ValueError("generation_chunks must not contain empty strings")

        self._validate_embedding_vectors()
        self._validate_rerank_scores()
        self._validate_failure_controls()

        if not callable(self.sync_sleeper):
            raise TypeError("sync_sleeper must be callable")
        if not callable(self.async_sleeper):
            raise TypeError("async_sleeper must be callable")

    def _validate_embedding_vectors(self) -> None:
        """Validate configured embedding values and dimensions."""
        if self.embedding_vectors is None:
            return
        if not isinstance(self.embedding_vectors, tuple):
            raise TypeError("embedding_vectors must be a tuple or None")
        if BackendOperation.EMBED not in self.metadata.supported_operations:
            raise ValueError("embedding_vectors requires the embed operation")
        if not self.embedding_vectors:
            raise ValueError("embedding_vectors must not be empty")
        if any(not isinstance(vector, tuple) for vector in self.embedding_vectors):
            raise TypeError("embedding_vectors must contain tuples")

        dimensions = len(self.embedding_vectors[0])
        if dimensions == 0:
            raise ValueError("embedding_vectors must not contain empty vectors")
        if any(len(vector) != dimensions for vector in self.embedding_vectors):
            raise ValueError("embedding_vectors must have one consistent dimension")
        for vector in self.embedding_vectors:
            for value in vector:
                validate_finite_number(value, "embedding vector values")

        declared_dimensions = self.metadata.embedding_dimensions
        if declared_dimensions is not None and declared_dimensions != dimensions:
            raise ValueError("embedding vector dimension must match backend metadata")

    def _validate_rerank_scores(self) -> None:
        """Validate configured candidate score overrides."""
        if not isinstance(self.rerank_scores, tuple):
            raise TypeError("rerank_scores must be a tuple")
        if self.rerank_scores and (
            BackendOperation.RERANK not in self.metadata.supported_operations
        ):
            raise ValueError("rerank_scores requires the rerank operation")
        if any(not isinstance(item, tuple) for item in self.rerank_scores):
            raise TypeError("rerank_scores must contain tuples")
        if any(len(item) != 2 for item in self.rerank_scores):
            raise ValueError("rerank_scores entries must contain an ID and score")

        candidate_ids: list[str] = []
        for candidate_id, score in self.rerank_scores:
            if not isinstance(candidate_id, str):
                raise TypeError("rerank score candidate IDs must be strings")
            if not candidate_id:
                raise ValueError("rerank score candidate IDs must not be empty")
            validate_finite_number(score, "rerank score values")
            candidate_ids.append(candidate_id)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("rerank score candidate IDs must be unique")

    def _validate_failure_controls(self) -> None:
        """Validate failure injection as one coherent configuration."""
        if self.failure_mode is not None and not isinstance(
            self.failure_mode, FakeFailureMode
        ):
            raise TypeError("failure_mode must be FakeFailureMode or None")
        if self.fail_operation is not None and not isinstance(
            self.fail_operation, BackendOperation
        ):
            raise TypeError("fail_operation must be BackendOperation or None")
        if (self.failure_mode is None) != (self.fail_operation is None):
            raise ValueError(
                "failure_mode and fail_operation must be configured together"
            )
        if (
            self.fail_operation is not None
            and self.fail_operation not in self.metadata.supported_operations
        ):
            raise ValueError("fail_operation must be advertised by backend metadata")

        if self.fail_after_generation_chunks is None:
            return
        if isinstance(self.fail_after_generation_chunks, bool) or not isinstance(
            self.fail_after_generation_chunks, int
        ):
            raise TypeError("fail_after_generation_chunks must be an integer or None")
        if self.fail_after_generation_chunks < 0:
            raise ValueError("fail_after_generation_chunks must not be negative")
        if self.fail_operation is not BackendOperation.GENERATE:
            raise ValueError(
                "fail_after_generation_chunks requires a generate failure operation"
            )
        if self.fail_after_generation_chunks >= len(self.generation_chunks):
            raise ValueError(
                "fail_after_generation_chunks must identify an emitted chunk boundary"
            )


@dataclass(frozen=True, slots=True)
class _GenerationPlan:
    """Deterministic output selected for one generation request."""

    chunks: tuple[str, ...]
    output_text: str
    finish_reason: str


class FakeBackend:
    """Deterministic backend implementing embedding, reranking, and generation."""

    def __init__(self, config: FakeBackendConfig | None = None) -> None:
        """Create an unloaded fake backend with the supplied behaviour."""
        if config is not None and not isinstance(config, FakeBackendConfig):
            raise TypeError("config must be FakeBackendConfig or None")
        self._config = FakeBackendConfig() if config is None else config
        self._is_ready = False
        self._is_closed = False

    @property
    def metadata(self) -> BackendMetadata:
        """Return configured fake backend metadata."""
        return self._config.metadata

    @property
    def is_ready(self) -> bool:
        """Return whether the fake accepts inference requests."""
        return self._is_ready

    @property
    def is_closed(self) -> bool:
        """Return whether the fake has been closed."""
        return self._is_closed

    def load(self) -> None:
        """Transition the fake to ready state; repeated calls are harmless."""
        if self._is_closed:
            raise BackendClosedError()
        self._is_ready = True
        _LOGGER.info(
            "backend_load_completed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            model_name=self.metadata.model_name,
        )

    def close(self) -> None:
        """Close the fake; repeated calls are harmless."""
        self._is_ready = False
        self._is_closed = True
        _LOGGER.info(
            "backend_close_completed",
            pipeline_stage="runtime",
            backend_name=self.metadata.backend_name,
            model_name=self.metadata.model_name,
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Return deterministic vectors in the request's input order."""
        self._ensure_available()
        self._ensure_operation_supported(BackendOperation.EMBED)
        vectors, output_dimensions = self._embedding_vectors(request)
        self._log_operation_started(BackendOperation.EMBED)
        self._maybe_fail(BackendOperation.EMBED)
        self._config.sync_sleeper(self._config.initial_delay_seconds)
        self._config.sync_sleeper(
            self._config.per_item_delay_seconds * len(request.input_texts)
        )

        _LOGGER.debug(
            "backend_operation_completed",
            pipeline_stage="runtime",
            operation=BackendOperation.EMBED.value,
            input_count=len(request.input_texts),
        )
        return EmbeddingResult(
            vectors=vectors,
            model_metadata=replace(
                self.metadata,
                embedding_dimensions=output_dimensions,
            ),
            embedding_profile=EmbeddingProfile(
                profile_version="1.0.0",
                tokenizer_id="fake/whitespace-counter",
                tokenizer_revision="1.0.0",
                query_prefix="",
                document_prefix="",
                pooling_strategy="position-based-vector-v1",
                input_normalisation="none",
                max_input_tokens=None,
                over_limit_policy=None,
                l2_normalised=request.options.normalise,
                inference_dtype="float64",
                device_type="cpu",
            ),
            usage=EmbeddingUsage(
                input_tokens=sum(
                    self._token_count(text) for text in request.input_texts
                ),
                total_tokens=sum(
                    self._token_count(text) for text in request.input_texts
                ),
            ),
        )

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Return deterministic scores and contiguous ranks."""
        self._ensure_available()
        self._ensure_operation_supported(BackendOperation.RERANK)
        self._log_operation_started(BackendOperation.RERANK)
        self._maybe_fail(BackendOperation.RERANK)
        self._config.sync_sleeper(self._config.initial_delay_seconds)
        self._config.sync_sleeper(
            self._config.per_item_delay_seconds * len(request.candidates)
        )

        configured_scores = dict(self._config.rerank_scores)
        scored = sorted(
            (
                (
                    candidate.candidate_id,
                    configured_scores.get(candidate.candidate_id, 1.0 / (index + 1)),
                )
                for index, candidate in enumerate(request.candidates)
            ),
            key=lambda candidate: (-candidate[1], candidate[0]),
        )
        ranked = tuple(
            ScoredCandidate(
                candidate_id=candidate[0],
                score=candidate[1],
                rank=rank,
            )
            for rank, candidate in enumerate(scored[: request.top_k], start=1)
        )
        _LOGGER.debug(
            "backend_operation_completed",
            pipeline_stage="runtime",
            operation=BackendOperation.RERANK.value,
            candidate_count=len(request.candidates),
            result_count=len(ranked),
        )
        return RerankResult(scored_candidates=ranked, model_metadata=self.metadata)

    def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationEvent]:
        """Return a validated asynchronous stream of deterministic text chunks."""
        self._ensure_available()
        self._ensure_operation_supported(BackendOperation.GENERATE)
        plan = self._generation_plan(request)
        self._log_operation_started(BackendOperation.GENERATE)
        return validate_generation_stream(self._generate_events(request, plan))

    async def _generate_events(
        self,
        request: GenerationRequest,
        plan: _GenerationPlan,
    ) -> AsyncIterator[GenerationEvent]:
        """Produce raw generation events for the public validated stream."""
        self._maybe_fail(BackendOperation.GENERATE)
        await self._config.async_sleeper(self._config.initial_delay_seconds)
        emitted_chunks = 0

        for sequence_number, chunk in enumerate(plan.chunks):
            self._ensure_available()
            if (
                self._config.fail_after_generation_chunks is not None
                and emitted_chunks >= self._config.fail_after_generation_chunks
            ):
                self._raise_configured_failure(BackendOperation.GENERATE)
            await self._config.async_sleeper(self._config.per_chunk_delay_seconds)
            yield GenerationEvent(
                event_type=GenerationEventType.DELTA,
                sequence_number=sequence_number,
                text_delta=chunk,
            )
            emitted_chunks += 1

        self._ensure_available()
        _LOGGER.debug(
            "backend_stream_completed",
            pipeline_stage="runtime",
            operation=BackendOperation.GENERATE.value,
            chunk_count=emitted_chunks,
        )
        yield GenerationEvent(
            event_type=GenerationEventType.COMPLETED,
            sequence_number=emitted_chunks,
            finish_reason=plan.finish_reason,
            usage=GenerationUsage(
                input_tokens=sum(
                    self._token_count(message.content) for message in request.messages
                ),
                output_tokens=self._token_count(plan.output_text),
                total_tokens=sum(
                    self._token_count(message.content) for message in request.messages
                )
                + self._token_count(plan.output_text),
            ),
        )

    def _embedding_vectors(
        self,
        request: EmbeddingRequest,
    ) -> tuple[tuple[tuple[float, ...], ...], int]:
        """Apply requested dimension and normalisation controls."""
        configured_vectors = self._config.embedding_vectors
        available_dimensions = (
            len(configured_vectors[0])
            if configured_vectors is not None
            else self.metadata.embedding_dimensions
        )
        requested_dimensions = request.options.dimensions
        if (
            requested_dimensions is not None
            and available_dimensions is not None
            and requested_dimensions > available_dimensions
        ):
            raise BackendRequestError(
                "requested embedding dimensions exceed backend dimensions"
            )
        output_dimensions = requested_dimensions or available_dimensions or 3

        if configured_vectors is not None:
            if len(configured_vectors) != len(request.input_texts):
                raise BackendRequestError(
                    "configured embedding vector count does not match request"
                )
            vectors = tuple(vector[:output_dimensions] for vector in configured_vectors)
        else:
            vectors = tuple(
                tuple(
                    float(index + dimension) for dimension in range(output_dimensions)
                )
                for index in range(len(request.input_texts))
            )

        if request.options.normalise:
            vectors = tuple(self._normalise_vector(vector) for vector in vectors)
        return vectors, output_dimensions

    def _generation_plan(self, request: GenerationRequest) -> _GenerationPlan:
        """Apply deterministic output controls or reject sampling controls."""
        unsupported_sampling = tuple(
            field_name
            for field_name, value in (
                ("temperature", request.options.temperature),
                ("top_p", request.options.top_p),
                ("seed", request.options.seed),
            )
            if value is not None
        )
        if unsupported_sampling:
            names = ", ".join(unsupported_sampling)
            raise BackendRequestError(
                f"fake backend does not support sampling controls: {names}"
            )

        output_text = "".join(self._config.generation_chunks)
        stop_cutoffs = tuple(
            position
            for sequence in request.options.stop_sequences
            if (position := output_text.find(sequence)) >= 0
        )
        stop_cutoff = min(stop_cutoffs) if stop_cutoffs else None
        token_cutoff = self._token_limit_cutoff(
            output_text,
            request.options.maximum_output_tokens,
        )

        if token_cutoff is not None and (
            stop_cutoff is None or token_cutoff < stop_cutoff
        ):
            cutoff = token_cutoff
            finish_reason = "length"
        elif stop_cutoff is not None:
            cutoff = stop_cutoff
            finish_reason = "stop"
        else:
            cutoff = len(output_text)
            finish_reason = "stop"

        chunks = self._truncate_chunks(self._config.generation_chunks, cutoff)
        return _GenerationPlan(
            chunks=chunks,
            output_text=output_text[:cutoff],
            finish_reason=finish_reason,
        )

    @staticmethod
    def _normalise_vector(vector: tuple[float, ...]) -> tuple[float, ...]:
        """L2-normalise a vector, retaining an all-zero vector unchanged."""
        largest_magnitude = max(abs(value) for value in vector)
        if largest_magnitude == 0:
            return tuple(0.0 for _ in vector)
        scaled = tuple(float(value / largest_magnitude) for value in vector)
        scaled_norm = math.sqrt(math.fsum(value * value for value in scaled))
        return tuple(value / scaled_norm for value in scaled)

    @staticmethod
    def _token_limit_cutoff(text: str, maximum_tokens: int | None) -> int | None:
        """Return the character cutoff required by the fake whitespace tokeniser."""
        if maximum_tokens is None:
            return None
        tokens = tuple(re.finditer(r"\S+", text))
        if len(tokens) <= maximum_tokens:
            return None
        return tokens[maximum_tokens - 1].end()

    @staticmethod
    def _truncate_chunks(chunks: tuple[str, ...], cutoff: int) -> tuple[str, ...]:
        """Preserve configured chunk boundaries up to a character cutoff."""
        remaining = cutoff
        selected: list[str] = []
        for chunk in chunks:
            if remaining == 0:
                break
            selected_length = min(len(chunk), remaining)
            selected.append(chunk[:selected_length])
            remaining -= selected_length
        return tuple(selected)

    def _ensure_available(self) -> None:
        """Raise the lifecycle error appropriate to the current state."""
        if self._is_closed:
            raise BackendClosedError()
        if not self._is_ready:
            raise BackendNotReadyError()

    def _ensure_operation_supported(self, operation: BackendOperation) -> None:
        """Reject operations that the configured metadata does not advertise."""
        if operation not in self.metadata.supported_operations:
            raise BackendRequestError(
                f"{operation.value} operation is not supported by this backend"
            )

    def _maybe_fail(self, operation: BackendOperation) -> None:
        """Raise a configured failure for an operation, if requested."""
        if (
            operation is BackendOperation.GENERATE
            and self._config.fail_after_generation_chunks is not None
        ):
            return
        if self._config.fail_operation is operation:
            self._raise_configured_failure(operation)

    def _raise_configured_failure(self, operation: BackendOperation) -> None:
        """Raise the configured typed failure."""
        error_type = (
            "request"
            if self._config.failure_mode is FakeFailureMode.REQUEST
            else "resource"
            if self._config.failure_mode is FakeFailureMode.RESOURCE
            else "execution"
        )
        _LOGGER.error(
            "backend_operation_failed",
            pipeline_stage="runtime",
            operation=operation.value,
            error_type=error_type,
            retryable=error_type == "resource",
        )
        if self._config.failure_mode is FakeFailureMode.REQUEST:
            raise BackendRequestError()
        if self._config.failure_mode is FakeFailureMode.RESOURCE:
            raise BackendResourceError()
        raise BackendExecutionError()

    def _log_operation_started(self, operation: BackendOperation) -> None:
        """Log safe operation metadata without request or response payloads."""
        _LOGGER.debug(
            "backend_operation_started",
            pipeline_stage="runtime",
            operation=operation.value,
            backend_name=self.metadata.backend_name,
            model_name=self.metadata.model_name,
        )

    @staticmethod
    def _token_count(text: str) -> int:
        """Return a deterministic fake token count for test accounting."""
        return len(text.split())
