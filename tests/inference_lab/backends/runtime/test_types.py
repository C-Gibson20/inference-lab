import pytest

from inference_lab.backends.runtime.types import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingResult,
    GenerationEvent,
    GenerationEventType,
    GenerationMessage,
    GenerationUsage,
    RerankResult,
    ScoredCandidate,
)


def metadata(*, dimensions: int | None = None) -> BackendMetadata:
    return BackendMetadata(
        backend_name="fake",
        model_name="test-model",
        model_revision="test-revision",
        supported_operations=(BackendOperation.GENERATE,),
        capabilities=(BackendCapability.STREAMING,),
        embedding_dimensions=dimensions,
    )


def test_embedding_result_validates_vector_dimensions() -> None:
    result = EmbeddingResult(
        vectors=((0.1, 0.2), (0.3, 0.4)),
        model_metadata=metadata(dimensions=2),
    )

    assert len(result.vectors) == 2


def test_embedding_result_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        EmbeddingResult(
            vectors=((0.1, 0.2), (0.3,)),
            model_metadata=metadata(),
        )


def test_rerank_result_uses_contiguous_ranks() -> None:
    result = RerankResult(
        scored_candidates=(
            ScoredCandidate(candidate_id="b", score=0.9, rank=1),
            ScoredCandidate(candidate_id="a", score=0.2, rank=2),
        ),
        model_metadata=metadata(),
    )

    assert result.scored_candidates[0].candidate_id == "b"


def test_rerank_result_rejects_non_contiguous_ranks() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        RerankResult(
            scored_candidates=(ScoredCandidate(candidate_id="a", score=0.2, rank=2),),
            model_metadata=metadata(),
        )


def test_generation_event_requires_text_for_delta() -> None:
    event = GenerationEvent(
        event_type=GenerationEventType.DELTA,
        sequence_number=0,
        text_delta="hello",
    )

    assert event.text_delta == "hello"


def test_generation_event_rejects_terminal_fields_on_delta() -> None:
    with pytest.raises(ValueError, match="terminal fields"):
        GenerationEvent(
            event_type=GenerationEventType.DELTA,
            sequence_number=0,
            text_delta="hello",
            usage=GenerationUsage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
            ),
        )


def test_generation_message_is_structured() -> None:
    message = GenerationMessage(role="user", content="Question")

    assert message.role == "user"
