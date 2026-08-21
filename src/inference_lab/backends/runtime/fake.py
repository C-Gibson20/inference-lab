"""Deterministic fake runtime backend for contract and integration tests."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from .errors import (
    BackendClosedError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendRequestError,
    BackendResourceError,
)
from .streaming import validate_generation_stream
from .types import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
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
    """Deterministic behavior controls for :class:`FakeBackend`."""

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
        """Validate fake timing and failure controls."""
        if (
            min(
                self.initial_delay_seconds,
                self.per_item_delay_seconds,
                self.per_chunk_delay_seconds,
            )
            < 0
        ):
            raise ValueError("fake delays must not be negative")
        if any(not chunk for chunk in self.generation_chunks):
            raise ValueError("generation_chunks must not contain empty strings")
        if (
            self.fail_after_generation_chunks is not None
            and self.fail_after_generation_chunks < 0
        ):
            raise ValueError("fail_after_generation_chunks must not be negative")
        if self.failure_mode is None and self.fail_operation is not None:
            raise ValueError("failure_mode is required when fail_operation is set")


class FakeBackend:
    """Deterministic backend implementing embedding, reranking, and generation."""

    def __init__(self, config: FakeBackendConfig | None = None) -> None:
        """Create an unloaded fake backend with the supplied behavior."""
        self._config = config or FakeBackendConfig()
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

    def close(self) -> None:
        """Close the fake; repeated calls are harmless."""
        self._is_ready = False
        self._is_closed = True

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Return deterministic vectors in the request's input order."""
        self._ensure_available()
        self._maybe_fail(BackendOperation.EMBED)
        self._config.sync_sleeper(self._config.initial_delay_seconds)
        self._config.sync_sleeper(
            self._config.per_item_delay_seconds * len(request.input_texts)
        )

        vectors = self._embedding_vectors(len(request.input_texts))
        return EmbeddingResult(
            vectors=vectors,
            model_metadata=self.metadata,
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
        return RerankResult(scored_candidates=ranked, model_metadata=self.metadata)

    def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationEvent]:
        """Return a validated asynchronous stream of deterministic text chunks."""
        self._ensure_available()
        return validate_generation_stream(self._generate_events(request))

    async def _generate_events(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationEvent]:
        """Produce raw generation events for the public validated stream."""
        self._maybe_fail(BackendOperation.GENERATE)
        await self._config.async_sleeper(self._config.initial_delay_seconds)
        emitted_chunks = 0

        for sequence_number, chunk in enumerate(self._config.generation_chunks):
            self._ensure_available()
            if (
                self._config.fail_after_generation_chunks is not None
                and emitted_chunks >= self._config.fail_after_generation_chunks
            ):
                self._raise_configured_failure()
            await self._config.async_sleeper(self._config.per_chunk_delay_seconds)
            yield GenerationEvent(
                event_type=GenerationEventType.DELTA,
                sequence_number=sequence_number,
                text_delta=chunk,
            )
            emitted_chunks += 1

        self._ensure_available()
        yield GenerationEvent(
            event_type=GenerationEventType.COMPLETED,
            sequence_number=emitted_chunks,
            finish_reason="stop",
            usage=GenerationUsage(
                input_tokens=sum(
                    self._token_count(message.content) for message in request.messages
                ),
                output_tokens=self._token_count(
                    "".join(self._config.generation_chunks)
                ),
                total_tokens=sum(
                    self._token_count(message.content) for message in request.messages
                )
                + self._token_count("".join(self._config.generation_chunks)),
            ),
        )

    def _embedding_vectors(self, count: int) -> tuple[tuple[float, ...], ...]:
        """Return configured or deterministic vectors for a batch."""
        if self._config.embedding_vectors is not None:
            if len(self._config.embedding_vectors) != count:
                raise BackendRequestError(
                    "configured embedding vector count does not match request"
                )
            return self._config.embedding_vectors
        dimensions = self.metadata.embedding_dimensions or 3
        return tuple(
            tuple(float(index + dimension) for dimension in range(dimensions))
            for index in range(count)
        )

    def _ensure_available(self) -> None:
        """Raise the lifecycle error appropriate to the current state."""
        if self._is_closed:
            raise BackendClosedError()
        if not self._is_ready:
            raise BackendNotReadyError()

    def _maybe_fail(self, operation: BackendOperation) -> None:
        """Raise a configured failure for an operation, if requested."""
        if (
            operation is BackendOperation.GENERATE
            and self._config.fail_after_generation_chunks is not None
        ):
            return
        if self._config.fail_operation is operation:
            self._raise_configured_failure()

    def _raise_configured_failure(self) -> None:
        """Raise the configured typed failure."""
        if self._config.failure_mode is FakeFailureMode.REQUEST:
            raise BackendRequestError()
        if self._config.failure_mode is FakeFailureMode.RESOURCE:
            raise BackendResourceError()
        raise BackendExecutionError()

    @staticmethod
    def _token_count(text: str) -> int:
        """Return a deterministic fake token count for test accounting."""
        return len(text.split())
