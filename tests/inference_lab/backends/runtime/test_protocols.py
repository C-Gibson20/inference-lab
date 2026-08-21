from collections.abc import AsyncIterator

from inference_lab.backends.runtime.protocols import (
    EmbeddingBackend,
    GeneratorBackend,
    RerankerBackend,
)
from inference_lab.backends.runtime.types import (
    BackendMetadata,
    EmbeddingRequest,
    EmbeddingResult,
    GenerationEvent,
    GenerationRequest,
    RerankRequest,
    RerankResult,
)


class StructuralBackend:
    """Test implementation that does not inherit from a runtime protocol."""

    metadata: BackendMetadata

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        raise NotImplementedError

    def rerank(self, request: RerankRequest) -> RerankResult:
        raise NotImplementedError

    def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationEvent]:
        raise NotImplementedError


def test_backend_protocols_are_structural() -> None:
    backend = StructuralBackend()

    embedding_backend: EmbeddingBackend = backend
    reranker_backend: RerankerBackend = backend
    generator_backend: GeneratorBackend = backend

    assert embedding_backend is backend
    assert reranker_backend is backend
    assert generator_backend is backend
