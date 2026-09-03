"""Tests for runtime request and result contracts."""

from collections.abc import Callable

import pytest

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingOptions,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingUsage,
    GenerationEvent,
    GenerationEventType,
    GenerationMessage,
    GenerationOptions,
    GenerationRequest,
    GenerationUsage,
    RerankCandidate,
    RerankRequest,
    RerankResult,
    ScoredCandidate,
)


def metadata(
    *,
    backend_name: str = "fake",
    model_name: str = "test-model",
    model_revision: str = "test-revision",
    dimensions: int | None = None,
    supported_operations: tuple[BackendOperation, ...] = (
        BackendOperation.EMBED,
        BackendOperation.RERANK,
        BackendOperation.GENERATE,
    ),
    capabilities: tuple[BackendCapability, ...] = (BackendCapability.STREAMING,),
) -> BackendMetadata:
    return BackendMetadata(
        backend_name=backend_name,
        model_name=model_name,
        model_revision=model_revision,
        supported_operations=supported_operations,
        capabilities=capabilities,
        embedding_dimensions=dimensions,
    )


@pytest.mark.parametrize(
    ("supported_operations", "capabilities", "message"),
    [
        ((), (), "must not be empty"),
        (
            (BackendOperation.EMBED, BackendOperation.EMBED),
            (),
            "duplicates",
        ),
        (("embed",), (), "BackendOperation"),
        (
            (BackendOperation.GENERATE,),
            (),
            "advertised together",
        ),
        (
            (BackendOperation.EMBED,),
            (BackendCapability.STREAMING,),
            "advertised together",
        ),
    ],
)
def test_backend_metadata_rejects_invalid_operation_contracts(
    supported_operations: tuple[BackendOperation, ...],
    capabilities: tuple[BackendCapability, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        metadata(
            supported_operations=supported_operations,
            capabilities=capabilities,
        )


def test_backend_metadata_rejects_embedding_dimensions_without_embed() -> None:
    with pytest.raises(ValueError, match="requires the embed operation"):
        metadata(
            dimensions=3,
            supported_operations=(BackendOperation.RERANK,),
            capabilities=(),
        )


def test_backend_metadata_rejects_duplicate_capabilities() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        metadata(
            capabilities=(
                BackendCapability.STREAMING,
                BackendCapability.STREAMING,
            )
        )


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: metadata(dimensions=True),
            id="embedding-dimensions",
        ),
        pytest.param(
            lambda: EmbeddingOptions(dimensions=True),
            id="requested-embedding-dimensions",
        ),
        pytest.param(
            lambda: EmbeddingUsage(input_tokens=True, total_tokens=1),
            id="embedding-usage",
        ),
        pytest.param(
            lambda: RerankRequest(
                query="query",
                candidates=(RerankCandidate(candidate_id="a", candidate_text="A"),),
                top_k=True,
            ),
            id="top-k",
        ),
        pytest.param(
            lambda: ScoredCandidate(candidate_id="a", score=0.5, rank=True),
            id="rank",
        ),
        pytest.param(
            lambda: GenerationOptions(maximum_output_tokens=True),
            id="maximum-output-tokens",
        ),
        pytest.param(
            lambda: GenerationOptions(seed=True),
            id="seed",
        ),
        pytest.param(
            lambda: GenerationUsage(
                input_tokens=True,
                output_tokens=0,
                total_tokens=1,
            ),
            id="generation-usage",
        ),
        pytest.param(
            lambda: GenerationEvent(
                event_type=GenerationEventType.DELTA,
                sequence_number=True,
                text_delta="delta",
            ),
            id="sequence-number",
        ),
    ],
)
def test_integer_contracts_reject_booleans(factory: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match="integer"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: GenerationOptions(temperature=float("nan")),
            id="temperature",
        ),
        pytest.param(
            lambda: GenerationOptions(top_p=float("inf")),
            id="top-p",
        ),
    ],
)
def test_generation_options_reject_non_finite_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        factory()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_embedding_result_rejects_invalid_vector_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="vector values"):
        EmbeddingResult(
            vectors=((value,),),  # type: ignore[arg-type]
            model_metadata=metadata(dimensions=1),
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_scored_candidate_rejects_invalid_score(score: object) -> None:
    with pytest.raises((TypeError, ValueError), match="score"):
        ScoredCandidate(
            candidate_id="a",
            score=score,  # type: ignore[arg-type]
            rank=1,
        )


def test_rerank_request_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValueError, match="candidate IDs must be unique"):
        RerankRequest(
            query="query",
            candidates=(
                RerankCandidate(candidate_id="a", candidate_text="first"),
                RerankCandidate(candidate_id="a", candidate_text="second"),
            ),
        )


@pytest.mark.parametrize(
    "scored_candidates",
    [
        (
            ScoredCandidate(candidate_id="a", score=0.9, rank=1),
            ScoredCandidate(candidate_id="a", score=0.8, rank=2),
        ),
        (
            ScoredCandidate(candidate_id="a", score=0.2, rank=1),
            ScoredCandidate(candidate_id="b", score=0.9, rank=2),
        ),
    ],
)
def test_rerank_result_rejects_identity_or_score_order_violations(
    scored_candidates: tuple[ScoredCandidate, ...],
) -> None:
    with pytest.raises(ValueError):
        RerankResult(
            scored_candidates=scored_candidates,
            model_metadata=metadata(),
        )


def test_results_require_metadata_for_their_operation() -> None:
    embed_only = metadata(
        dimensions=2,
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(),
    )
    rerank_only = metadata(
        supported_operations=(BackendOperation.RERANK,),
        capabilities=(),
    )

    with pytest.raises(ValueError, match="advertise embed"):
        EmbeddingResult(vectors=((0.1, 0.2),), model_metadata=rerank_only)
    with pytest.raises(ValueError, match="advertise rerank"):
        RerankResult(
            scored_candidates=(ScoredCandidate(candidate_id="a", score=0.5, rank=1),),
            model_metadata=embed_only,
        )


def test_generation_contracts_validate_runtime_container_and_member_types() -> None:
    with pytest.raises(ValueError, match="supported generation role"):
        GenerationMessage(role="invalid", content="content")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="input_texts must be a tuple"):
        EmbeddingRequest(input_texts=["text"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GenerationMessage"):
        GenerationRequest(messages=("message",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicates"):
        GenerationOptions(stop_sequences=("stop", "stop"))


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


def test_embedding_result_rejects_dimension_mismatched_with_metadata() -> None:
    with pytest.raises(ValueError, match="backend metadata"):
        EmbeddingResult(
            vectors=((0.1,),),
            model_metadata=metadata(dimensions=2),
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


def test_generation_event_rejects_unknown_event_type() -> None:
    with pytest.raises(TypeError, match="GenerationEventType"):
        GenerationEvent(
            event_type="delta",  # type: ignore[arg-type]
            sequence_number=0,
            text_delta="hello",
        )


def test_generation_message_is_structured() -> None:
    message = GenerationMessage(role="user", content="Question")

    assert message.role == "user"


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    [
        pytest.param(
            lambda: metadata(backend_name=1),  # type: ignore[arg-type]
            TypeError,
            "backend_name must be a string",
            id="metadata-name-type",
        ),
        pytest.param(
            lambda: metadata(model_revision=""),
            ValueError,
            "model_revision must not be empty",
            id="metadata-empty-revision",
        ),
        pytest.param(
            lambda: metadata(supported_operations=[]),  # type: ignore[arg-type]
            TypeError,
            "supported_operations must be a tuple",
            id="metadata-operation-container",
        ),
        pytest.param(
            lambda: metadata(capabilities=[]),  # type: ignore[arg-type]
            TypeError,
            "capabilities must be a tuple",
            id="metadata-capability-container",
        ),
        pytest.param(
            lambda: metadata(capabilities=("streaming",)),  # type: ignore[arg-type]
            TypeError,
            "BackendCapability",
            id="metadata-capability-member",
        ),
        pytest.param(
            lambda: metadata(dimensions=0),
            ValueError,
            "embedding_dimensions must be positive",
            id="metadata-embedding-dimensions",
        ),
        pytest.param(
            lambda: EmbeddingOptions(dimensions=0),
            ValueError,
            "dimensions must be positive",
            id="embedding-options-dimensions",
        ),
        pytest.param(
            lambda: EmbeddingOptions(normalise=1),  # type: ignore[arg-type]
            TypeError,
            "normalise must be a boolean",
            id="embedding-options-normalise",
        ),
        pytest.param(
            lambda: EmbeddingRequest(input_texts=()),
            ValueError,
            "input_texts must not be empty",
            id="embedding-request-empty",
        ),
        pytest.param(
            lambda: EmbeddingRequest(input_texts=(1,)),  # type: ignore[arg-type]
            TypeError,
            "input_texts must contain strings",
            id="embedding-request-member",
        ),
        pytest.param(
            lambda: EmbeddingRequest(input_texts=("",)),
            ValueError,
            "empty text",
            id="embedding-request-empty-member",
        ),
        pytest.param(
            lambda: EmbeddingRequest(
                input_texts=("text",),
                options=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "options must be EmbeddingOptions",
            id="embedding-request-options",
        ),
        pytest.param(
            lambda: EmbeddingUsage(input_tokens=-1, total_tokens=0),
            ValueError,
            "must not be negative",
            id="embedding-usage-negative",
        ),
        pytest.param(
            lambda: EmbeddingUsage(input_tokens=2, total_tokens=1),
            ValueError,
            "at least input_tokens",
            id="embedding-usage-total",
        ),
        pytest.param(
            lambda: EmbeddingResult(
                vectors=[],  # type: ignore[arg-type]
                model_metadata=metadata(),
            ),
            TypeError,
            "vectors must be a tuple",
            id="embedding-result-container",
        ),
        pytest.param(
            lambda: EmbeddingResult(vectors=(), model_metadata=metadata()),
            ValueError,
            "vectors must not be empty",
            id="embedding-result-empty",
        ),
        pytest.param(
            lambda: EmbeddingResult(
                vectors=([0.1],),  # type: ignore[arg-type]
                model_metadata=metadata(),
            ),
            TypeError,
            "vectors must contain tuples",
            id="embedding-result-vector-container",
        ),
        pytest.param(
            lambda: EmbeddingResult(
                vectors=((0.1,),),
                model_metadata=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "model_metadata must be BackendMetadata",
            id="embedding-result-metadata",
        ),
        pytest.param(
            lambda: EmbeddingResult(
                vectors=((0.1,),),
                model_metadata=metadata(),
                usage=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "usage must be EmbeddingUsage",
            id="embedding-result-usage",
        ),
        pytest.param(
            lambda: EmbeddingResult(vectors=((),), model_metadata=metadata()),
            ValueError,
            "empty vectors",
            id="embedding-result-empty-vector",
        ),
        pytest.param(
            lambda: RerankRequest(query="query", candidates=[]),  # type: ignore[arg-type]
            TypeError,
            "candidates must be a tuple",
            id="rerank-request-container",
        ),
        pytest.param(
            lambda: RerankRequest(query="query", candidates=()),
            ValueError,
            "candidates must not be empty",
            id="rerank-request-empty",
        ),
        pytest.param(
            lambda: RerankRequest(
                query="query",
                candidates=("candidate",),  # type: ignore[arg-type]
            ),
            TypeError,
            "RerankCandidate",
            id="rerank-request-member",
        ),
        pytest.param(
            lambda: RerankRequest(
                query="query",
                candidates=(RerankCandidate(candidate_id="a", candidate_text="A"),),
                top_k=0,
            ),
            ValueError,
            "top_k must be positive",
            id="rerank-request-top-k",
        ),
        pytest.param(
            lambda: ScoredCandidate(candidate_id="a", score=0.5, rank=0),
            ValueError,
            "rank must be positive",
            id="scored-candidate-rank",
        ),
        pytest.param(
            lambda: RerankResult(
                scored_candidates=[],  # type: ignore[arg-type]
                model_metadata=metadata(),
            ),
            TypeError,
            "scored_candidates must be a tuple",
            id="rerank-result-container",
        ),
        pytest.param(
            lambda: RerankResult(scored_candidates=(), model_metadata=metadata()),
            ValueError,
            "scored_candidates must not be empty",
            id="rerank-result-empty",
        ),
        pytest.param(
            lambda: RerankResult(
                scored_candidates=("candidate",),  # type: ignore[arg-type]
                model_metadata=metadata(),
            ),
            TypeError,
            "ScoredCandidate",
            id="rerank-result-member",
        ),
        pytest.param(
            lambda: RerankResult(
                scored_candidates=(
                    ScoredCandidate(candidate_id="a", score=0.5, rank=1),
                ),
                model_metadata=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "model_metadata must be BackendMetadata",
            id="rerank-result-metadata",
        ),
        pytest.param(
            lambda: RerankResult(
                scored_candidates=(
                    ScoredCandidate(candidate_id="a", score=0.9, rank=1),
                    ScoredCandidate(candidate_id="b", score=0.8, rank=1),
                ),
                model_metadata=metadata(),
            ),
            ValueError,
            "ranks must be unique",
            id="rerank-result-duplicate-rank",
        ),
        pytest.param(
            lambda: GenerationMessage(
                role=1,  # type: ignore[arg-type]
                content="content",
            ),
            TypeError,
            "role must be a string",
            id="generation-message-role-type",
        ),
        pytest.param(
            lambda: GenerationMessage(role="user", content=""),
            ValueError,
            "content must not be empty",
            id="generation-message-content",
        ),
        pytest.param(
            lambda: GenerationOptions(maximum_output_tokens=0),
            ValueError,
            "maximum_output_tokens must be positive",
            id="generation-options-output-tokens",
        ),
        pytest.param(
            lambda: GenerationOptions(temperature=-0.1),
            ValueError,
            "temperature must not be negative",
            id="generation-options-temperature",
        ),
        pytest.param(
            lambda: GenerationOptions(top_p=0),
            ValueError,
            "top_p must be greater than zero",
            id="generation-options-top-p",
        ),
        pytest.param(
            lambda: GenerationOptions(stop_sequences=["stop"]),  # type: ignore[arg-type]
            TypeError,
            "stop_sequences must be a tuple",
            id="generation-options-stop-container",
        ),
        pytest.param(
            lambda: GenerationOptions(stop_sequences=(1,)),  # type: ignore[arg-type]
            TypeError,
            "stop_sequences must contain strings",
            id="generation-options-stop-member",
        ),
        pytest.param(
            lambda: GenerationOptions(stop_sequences=("",)),
            ValueError,
            "empty strings",
            id="generation-options-empty-stop",
        ),
        pytest.param(
            lambda: GenerationRequest(messages=[]),  # type: ignore[arg-type]
            TypeError,
            "messages must be a tuple",
            id="generation-request-container",
        ),
        pytest.param(
            lambda: GenerationRequest(messages=()),
            ValueError,
            "messages must not be empty",
            id="generation-request-empty",
        ),
        pytest.param(
            lambda: GenerationRequest(
                messages=(GenerationMessage(role="user", content="content"),),
                options=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "options must be GenerationOptions",
            id="generation-request-options",
        ),
        pytest.param(
            lambda: GenerationRequest(
                messages=(GenerationMessage(role="user", content="content"),),
                request_id="",
            ),
            ValueError,
            "request_id must not be empty",
            id="generation-request-id",
        ),
        pytest.param(
            lambda: GenerationUsage(
                input_tokens=-1,
                output_tokens=0,
                total_tokens=-1,
            ),
            ValueError,
            "token counts must not be negative",
            id="generation-usage-negative",
        ),
        pytest.param(
            lambda: GenerationUsage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=3,
            ),
            ValueError,
            "total_tokens must equal",
            id="generation-usage-total",
        ),
        pytest.param(
            lambda: GenerationEvent(
                event_type=GenerationEventType.DELTA,
                sequence_number=-1,
                text_delta="delta",
            ),
            ValueError,
            "sequence_number must not be negative",
            id="generation-event-sequence",
        ),
        pytest.param(
            lambda: GenerationEvent(
                event_type=GenerationEventType.COMPLETED,
                sequence_number=0,
                text_delta="unexpected",
                finish_reason="stop",
            ),
            ValueError,
            "cannot contain text_delta",
            id="generation-event-completed-delta",
        ),
        pytest.param(
            lambda: GenerationEvent(
                event_type=GenerationEventType.COMPLETED,
                sequence_number=0,
            ),
            TypeError,
            "finish_reason must be a string",
            id="generation-event-finish-reason",
        ),
        pytest.param(
            lambda: GenerationEvent(
                event_type=GenerationEventType.COMPLETED,
                sequence_number=0,
                finish_reason="stop",
                usage=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "usage must be GenerationUsage",
            id="generation-event-usage",
        ),
    ],
)
def test_runtime_contracts_reject_invalid_boundaries(
    factory: Callable[[], object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        factory()
