from collections.abc import AsyncIterator

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingBackend,
    EmbeddingRequest,
    EmbeddingResult,
    FakeBackend,
    GenerationEvent,
    GenerationRequest,
    GeneratorBackend,
    LifecycleBackend,
    RerankerBackend,
    RerankRequest,
    RerankResult,
)


class StructuralBackend:
    """Test implementation that does not inherit from a runtime protocol."""

    metadata = BackendMetadata(
        backend_name="structural",
        model_name="test-model",
        model_revision="test",
        supported_operations=(
            BackendOperation.EMBED,
            BackendOperation.RERANK,
            BackendOperation.GENERATE,
        ),
        capabilities=(BackendCapability.STREAMING,),
        embedding_dimensions=3,
    )
    is_ready = False
    is_closed = False

    def load(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        raise NotImplementedError

    def rerank(self, request: RerankRequest) -> RerankResult:
        raise NotImplementedError

    def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationEvent]:
        raise NotImplementedError


def test_backend_protocols_are_structural() -> None:
    backend = StructuralBackend()

    assert isinstance(backend, LifecycleBackend)
    assert isinstance(backend, EmbeddingBackend)
    assert isinstance(backend, RerankerBackend)
    assert isinstance(backend, GeneratorBackend)


def test_fake_backend_satisfies_each_public_runtime_protocol() -> None:
    backend = FakeBackend()

    assert isinstance(backend, LifecycleBackend)
    assert isinstance(backend, EmbeddingBackend)
    assert isinstance(backend, RerankerBackend)
    assert isinstance(backend, GeneratorBackend)


def test_incomplete_backend_does_not_satisfy_operation_protocols() -> None:
    class LifecycleOnly:
        metadata = StructuralBackend.metadata
        is_ready = False
        is_closed = False

        def load(self) -> None:
            pass

        def close(self) -> None:
            pass

    backend = LifecycleOnly()

    assert isinstance(backend, LifecycleBackend)
    assert not isinstance(backend, EmbeddingBackend)
    assert not isinstance(backend, RerankerBackend)
    assert not isinstance(backend, GeneratorBackend)
