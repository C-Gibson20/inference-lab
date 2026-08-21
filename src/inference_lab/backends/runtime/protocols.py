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


class EmbeddingBackend(Protocol):
    """Backend capable of generating embeddings for text batches."""

    @property
    def metadata(self) -> BackendMetadata:
        """Return the backend identity and supported capabilities."""
        ...

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate one embedding vector for each input text."""
        ...


class RerankerBackend(Protocol):
    """Backend capable of scoring candidates against a query."""

    @property
    def metadata(self) -> BackendMetadata:
        """Return the backend identity and supported capabilities."""
        ...

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Score and rank the requested candidates."""
        ...


class GeneratorBackend(Protocol):
    """Backend capable of asynchronously streaming generated text."""

    @property
    def metadata(self) -> BackendMetadata:
        """Return the backend identity and supported capabilities."""
        ...

    def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationEvent]:
        """Stream ordered generation events for a request."""
        ...
