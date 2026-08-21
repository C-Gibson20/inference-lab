"""Structural protocols for model runtime backends."""

from collections.abc import AsyncIterator
from typing import Protocol

from .types import (
    BackendMetadata,
    EmbeddingRequest,
    EmbeddingResult,
    GenerationEvent,
    GenerationRequest,
    RerankRequest,
    RerankResult,
)


class LifecycleBackend(Protocol):
    """Backend with explicit loading, readiness, and shutdown behavior."""

    @property
    def metadata(self) -> BackendMetadata:
        """Return backend identity and supported capabilities."""
        ...

    @property
    def is_ready(self) -> bool:
        """Return whether the backend can accept inference requests."""
        ...

    @property
    def is_closed(self) -> bool:
        """Return whether the backend has been permanently closed."""
        ...

    def load(self) -> None:
        """Load resources and transition the backend to ready state.

        Implementations should be idempotent when already ready and should
        raise a typed backend error if loading fails.
        """
        ...

    def close(self) -> None:
        """Release resources and prevent subsequent requests.

        Implementations should be idempotent. Calls after close must raise
        ``BackendClosedError`` for inference operations.
        """
        ...


class EmbeddingBackend(LifecycleBackend, Protocol):
    """Backend capable of generating embeddings for text batches."""

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate one embedding vector for each input text.

        Implementations must reject calls while not ready or after close with
        the corresponding typed backend error.
        """
        ...


class RerankerBackend(LifecycleBackend, Protocol):
    """Backend capable of scoring candidates against a query."""

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Score and rank the requested candidates.

        Implementations must reject calls while not ready or after close with
        the corresponding typed backend error.
        """
        ...


class GeneratorBackend(LifecycleBackend, Protocol):
    """Backend capable of asynchronously streaming generated text."""

    def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationEvent]:
        """Stream ordered generation events for a request.

        Consumer cancellation must stop the stream without producing a
        successful completion event. The caller owns retry and recovery policy.
        """
        ...
