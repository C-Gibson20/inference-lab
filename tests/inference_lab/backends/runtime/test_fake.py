import asyncio
import json
from io import StringIO

import pytest

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendClosedError,
    BackendExecutionError,
    BackendMetadata,
    BackendNotReadyError,
    BackendOperation,
    BackendRequestError,
    BackendResourceError,
    EmbeddingOptions,
    EmbeddingRequest,
    FakeBackend,
    FakeBackendConfig,
    FakeFailureMode,
    GenerationEventType,
    GenerationMessage,
    GenerationOptions,
    GenerationRequest,
    RerankCandidate,
    RerankRequest,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig


def metadata_for(*operations: BackendOperation) -> BackendMetadata:
    """Build internally consistent metadata for selected fake operations."""
    return BackendMetadata(
        backend_name="fake",
        model_name="fake-model",
        model_revision="test",
        supported_operations=operations,
        capabilities=(
            (BackendCapability.STREAMING,)
            if BackendOperation.GENERATE in operations
            else ()
        ),
        embedding_dimensions=3 if BackendOperation.EMBED in operations else None,
    )


def test_fake_requires_load_before_inference() -> None:
    backend = FakeBackend()

    with pytest.raises(BackendNotReadyError):
        backend.embed(EmbeddingRequest(input_texts=("one",)))


def test_fake_lifecycle_is_idempotent_and_close_is_permanent() -> None:
    backend = FakeBackend()

    assert backend.is_ready is False
    assert backend.is_closed is False

    backend.load()
    backend.load()
    assert backend.is_ready is True
    assert backend.is_closed is False

    backend.close()
    backend.close()
    assert backend.is_ready is False
    assert backend.is_closed is True

    with pytest.raises(BackendClosedError):
        backend.embed(EmbeddingRequest(input_texts=("one",)))
    with pytest.raises(BackendClosedError):
        backend.load()


@pytest.mark.parametrize("operation", tuple(BackendOperation))
def test_fake_rejects_operations_not_advertised_by_metadata(
    operation: BackendOperation,
) -> None:
    supported_operations = tuple(
        candidate for candidate in BackendOperation if candidate is not operation
    )
    backend = FakeBackend(
        FakeBackendConfig(
            metadata=metadata_for(*supported_operations),
        )
    )
    backend.load()

    with pytest.raises(BackendRequestError, match=operation.value):
        if operation is BackendOperation.EMBED:
            backend.embed(EmbeddingRequest(input_texts=("one",)))
        elif operation is BackendOperation.RERANK:
            backend.rerank(
                RerankRequest(
                    query="query",
                    candidates=(RerankCandidate(candidate_id="a", candidate_text="A"),),
                )
            )
        else:
            backend.generate(
                GenerationRequest(
                    messages=(GenerationMessage(role="user", content="Hi"),)
                )
            )


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("initial_delay_seconds", -0.1, ValueError, "must not be negative"),
        ("per_item_delay_seconds", -0.1, ValueError, "must not be negative"),
        ("per_chunk_delay_seconds", -0.1, ValueError, "must not be negative"),
        ("initial_delay_seconds", float("nan"), ValueError, "must be finite"),
        ("initial_delay_seconds", float("inf"), ValueError, "must be finite"),
        ("initial_delay_seconds", True, TypeError, "must be a number"),
        ("initial_delay_seconds", "0.1", TypeError, "must be a number"),
    ],
)
def test_fake_config_rejects_invalid_delays(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        FakeBackendConfig(**{field_name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("chunks", "error_type", "message"),
    [
        (["chunk"], TypeError, "must be a tuple"),
        ((1,), TypeError, "must contain strings"),
        (("",), ValueError, "empty strings"),
    ],
)
def test_fake_config_rejects_invalid_generation_chunks(
    chunks: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        FakeBackendConfig(generation_chunks=chunks)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("vectors", "error_type", "message"),
    [
        ([], TypeError, "must be a tuple"),
        ((), ValueError, "must not be empty"),
        (([1.0, 2.0, 3.0],), TypeError, "must contain tuples"),
        (((),), ValueError, "empty vectors"),
        (((1.0,), (1.0, 2.0)), ValueError, "consistent dimension"),
        (((float("nan"),),), ValueError, "must be finite"),
        (((True,),), TypeError, "must be a number"),
        (((1.0, 2.0),), ValueError, "match backend metadata"),
    ],
)
def test_fake_config_rejects_invalid_embedding_vectors(
    vectors: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        FakeBackendConfig(embedding_vectors=vectors)  # type: ignore[arg-type]


def test_fake_config_requires_embed_operation_for_configured_vectors() -> None:
    with pytest.raises(ValueError, match="requires the embed operation"):
        FakeBackendConfig(
            metadata=metadata_for(BackendOperation.RERANK),
            embedding_vectors=((1.0, 2.0, 3.0),),
        )


@pytest.mark.parametrize(
    ("scores", "error_type", "message"),
    [
        ([("a", 0.1)], TypeError, "must be a tuple"),
        ((["a", 0.1],), TypeError, "must contain tuples"),
        ((("a",),), ValueError, "an ID and score"),
        (((1, 0.1),), TypeError, "candidate IDs must be strings"),
        ((("", 0.1),), ValueError, "candidate IDs must not be empty"),
        ((("a", float("inf")),), ValueError, "must be finite"),
        ((("a", True),), TypeError, "must be a number"),
        ((("a", 0.1), ("a", 0.2)), ValueError, "must be unique"),
    ],
)
def test_fake_config_rejects_invalid_rerank_scores(
    scores: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        FakeBackendConfig(rerank_scores=scores)  # type: ignore[arg-type]


def test_fake_config_requires_rerank_operation_for_configured_scores() -> None:
    with pytest.raises(ValueError, match="requires the rerank operation"):
        FakeBackendConfig(
            metadata=metadata_for(BackendOperation.EMBED),
            rerank_scores=(("a", 0.1),),
        )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"metadata": object()},
            TypeError,
            "metadata must be BackendMetadata",
        ),
        (
            {"failure_mode": FakeFailureMode.EXECUTION},
            ValueError,
            "configured together",
        ),
        (
            {"fail_operation": BackendOperation.EMBED},
            ValueError,
            "configured together",
        ),
        (
            {
                "failure_mode": "execution",
                "fail_operation": BackendOperation.EMBED,
            },
            TypeError,
            "failure_mode must be FakeFailureMode",
        ),
        (
            {
                "failure_mode": FakeFailureMode.EXECUTION,
                "fail_operation": "embed",
            },
            TypeError,
            "fail_operation must be BackendOperation",
        ),
        (
            {"fail_after_generation_chunks": True},
            TypeError,
            "must be an integer",
        ),
        (
            {"fail_after_generation_chunks": 1.5},
            TypeError,
            "must be an integer",
        ),
        (
            {"fail_after_generation_chunks": -1},
            ValueError,
            "must not be negative",
        ),
        (
            {"fail_after_generation_chunks": 0},
            ValueError,
            "requires a generate failure operation",
        ),
        (
            {
                "failure_mode": FakeFailureMode.EXECUTION,
                "fail_operation": BackendOperation.EMBED,
                "fail_after_generation_chunks": 0,
            },
            ValueError,
            "requires a generate failure operation",
        ),
        (
            {
                "failure_mode": FakeFailureMode.EXECUTION,
                "fail_operation": BackendOperation.GENERATE,
                "fail_after_generation_chunks": 2,
            },
            ValueError,
            "emitted chunk boundary",
        ),
        ({"sync_sleeper": object()}, TypeError, "sync_sleeper must be callable"),
        ({"async_sleeper": object()}, TypeError, "async_sleeper must be callable"),
    ],
)
def test_fake_config_rejects_incoherent_controls(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        FakeBackendConfig(**kwargs)  # type: ignore[arg-type]


def test_fake_config_failure_operation_must_be_advertised() -> None:
    with pytest.raises(ValueError, match="advertised by backend metadata"):
        FakeBackendConfig(
            metadata=metadata_for(BackendOperation.RERANK),
            failure_mode=FakeFailureMode.EXECUTION,
            fail_operation=BackendOperation.EMBED,
        )


def test_fake_backend_rejects_invalid_config_object() -> None:
    with pytest.raises(TypeError, match="FakeBackendConfig or None"):
        FakeBackend(config=0)  # type: ignore[arg-type]


def test_fake_embedding_is_deterministic_and_ordered() -> None:
    backend = FakeBackend()
    backend.load()

    result = backend.embed(EmbeddingRequest(input_texts=("one", "two")))

    assert result.vectors == ((0.0, 1.0, 2.0), (1.0, 2.0, 3.0))
    assert result.usage is not None
    assert result.usage.input_tokens == 2


def test_fake_returns_valid_configured_embedding_vectors() -> None:
    backend = FakeBackend(FakeBackendConfig(embedding_vectors=((0.1, 0.2, 0.3),)))
    backend.load()

    result = backend.embed(
        EmbeddingRequest(
            input_texts=("one",),
            options=EmbeddingOptions(dimensions=2),
        )
    )

    assert result.vectors == ((0.1, 0.2),)
    assert result.model_metadata.embedding_dimensions == 2


def test_fake_rejects_configured_vector_count_mismatched_with_request() -> None:
    backend = FakeBackend(FakeBackendConfig(embedding_vectors=((0.1, 0.2, 0.3),)))
    backend.load()

    with pytest.raises(BackendRequestError, match="vector count"):
        backend.embed(EmbeddingRequest(input_texts=("one", "two")))


def test_fake_applies_embedding_dimension_and_normalization_options() -> None:
    backend = FakeBackend()
    backend.load()

    result = backend.embed(
        EmbeddingRequest(
            input_texts=("one", "two"),
            options=EmbeddingOptions(dimensions=2, normalize=True),
        )
    )

    assert result.vectors[0] == pytest.approx((0.0, 1.0))
    assert result.vectors[1] == pytest.approx((1 / (5**0.5), 2 / (5**0.5)))
    assert result.model_metadata.embedding_dimensions == 2


def test_fake_normalization_preserves_zero_vector() -> None:
    backend = FakeBackend(FakeBackendConfig(embedding_vectors=((0.0, 0.0, 0.0),)))
    backend.load()

    result = backend.embed(
        EmbeddingRequest(
            input_texts=("one",),
            options=EmbeddingOptions(normalize=True),
        )
    )

    assert result.vectors == ((0.0, 0.0, 0.0),)


def test_fake_rejects_unsupported_embedding_dimensions_before_delay() -> None:
    delays: list[float] = []
    backend = FakeBackend(FakeBackendConfig(sync_sleeper=delays.append))
    backend.load()

    with pytest.raises(BackendRequestError, match="exceed backend dimensions"):
        backend.embed(
            EmbeddingRequest(
                input_texts=("one",),
                options=EmbeddingOptions(dimensions=4),
            )
        )

    assert delays == []


def test_fake_reranker_applies_configured_scores_and_top_k() -> None:
    backend = FakeBackend(
        FakeBackendConfig(
            rerank_scores=(("a", 0.1), ("b", 0.9)),
        )
    )
    backend.load()

    result = backend.rerank(
        RerankRequest(
            query="query",
            candidates=(
                RerankCandidate(candidate_id="a", candidate_text="A"),
                RerankCandidate(candidate_id="b", candidate_text="B"),
            ),
            top_k=1,
        )
    )

    assert result.scored_candidates[0].candidate_id == "b"
    assert result.scored_candidates[0].rank == 1


def test_fake_generator_streams_chunks_and_completion() -> None:
    backend = FakeBackend(FakeBackendConfig(generation_chunks=("hello", " world")))
    backend.load()

    async def collect() -> list:
        return [
            event
            async for event in backend.generate(
                GenerationRequest(
                    messages=(GenerationMessage(role="user", content="Hi"),)
                )
            )
        ]

    result = asyncio.run(collect())

    assert [event.event_type for event in result] == [
        GenerationEventType.DELTA,
        GenerationEventType.DELTA,
        GenerationEventType.COMPLETED,
    ]
    assert "".join(event.text_delta or "" for event in result) == "hello world"
    assert result[-1].usage is not None


def test_fake_generator_applies_output_token_limit() -> None:
    backend = FakeBackend(
        FakeBackendConfig(generation_chunks=("one ", "two ", "three"))
    )
    backend.load()

    async def collect() -> list:
        return [
            event
            async for event in backend.generate(
                GenerationRequest(
                    messages=(GenerationMessage(role="user", content="Hi"),),
                    options=GenerationOptions(maximum_output_tokens=2),
                )
            )
        ]

    result = asyncio.run(collect())

    assert "".join(event.text_delta or "" for event in result) == "one two"
    assert result[-1].finish_reason == "length"
    assert result[-1].usage is not None
    assert result[-1].usage.output_tokens == 2


def test_fake_generator_does_not_report_length_when_output_fits_limit() -> None:
    backend = FakeBackend(FakeBackendConfig(generation_chunks=("one", " two")))
    backend.load()

    async def collect() -> list:
        return [
            event
            async for event in backend.generate(
                GenerationRequest(
                    messages=(GenerationMessage(role="user", content="Hi"),),
                    options=GenerationOptions(maximum_output_tokens=3),
                )
            )
        ]

    result = asyncio.run(collect())

    assert "".join(event.text_delta or "" for event in result) == "one two"
    assert result[-1].finish_reason == "stop"


def test_fake_generator_applies_stop_sequence_across_chunk_boundaries() -> None:
    backend = FakeBackend(FakeBackendConfig(generation_chunks=("alpha EN", "D omega")))
    backend.load()

    async def collect() -> list:
        return [
            event
            async for event in backend.generate(
                GenerationRequest(
                    messages=(GenerationMessage(role="user", content="Hi"),),
                    options=GenerationOptions(stop_sequences=("END",)),
                )
            )
        ]

    result = asyncio.run(collect())

    assert "".join(event.text_delta or "" for event in result) == "alpha "
    assert result[-1].finish_reason == "stop"
    assert result[-1].sequence_number == 1


@pytest.mark.parametrize(
    "options",
    [
        GenerationOptions(temperature=0.0),
        GenerationOptions(top_p=1.0),
        GenerationOptions(seed=0),
    ],
)
def test_fake_generator_explicitly_rejects_sampling_controls(
    options: GenerationOptions,
) -> None:
    backend = FakeBackend()
    backend.load()

    with pytest.raises(BackendRequestError, match="sampling controls"):
        backend.generate(
            GenerationRequest(
                messages=(GenerationMessage(role="user", content="Hi"),),
                options=options,
            )
        )


def test_fake_generator_preserves_partial_output_before_failure() -> None:
    backend = FakeBackend(
        FakeBackendConfig(
            generation_chunks=("partial", "unseen"),
            failure_mode=FakeFailureMode.EXECUTION,
            fail_operation=BackendOperation.GENERATE,
            fail_after_generation_chunks=1,
        )
    )
    backend.load()

    async def consume() -> list[str | None]:
        output = []
        with pytest.raises(BackendExecutionError):
            async for event in backend.generate(
                GenerationRequest(
                    messages=(GenerationMessage(role="user", content="Hi"),)
                )
            ):
                output.append(event.text_delta)
        return output

    assert asyncio.run(consume()) == ["partial"]


def test_fake_can_inject_resource_failure() -> None:
    backend = FakeBackend(
        FakeBackendConfig(
            failure_mode=FakeFailureMode.RESOURCE,
            fail_operation=BackendOperation.EMBED,
        )
    )
    backend.load()

    with pytest.raises(BackendResourceError):
        backend.embed(EmbeddingRequest(input_texts=("one",)))


@pytest.mark.parametrize(
    ("failure_mode", "error_type", "retryable"),
    [
        (FakeFailureMode.REQUEST, BackendRequestError, False),
        (FakeFailureMode.EXECUTION, BackendExecutionError, False),
        (FakeFailureMode.RESOURCE, BackendResourceError, True),
    ],
)
def test_fake_logs_failed_operations_at_error_level(
    failure_mode: FakeFailureMode,
    error_type: type[Exception],
    retryable: bool,
) -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    secret = "PRIVATE REQUEST CONTENT"
    backend = FakeBackend(
        FakeBackendConfig(
            failure_mode=failure_mode,
            fail_operation=BackendOperation.EMBED,
        )
    )
    backend.load()

    with pytest.raises(error_type):
        backend.embed(EmbeddingRequest(input_texts=(secret,)))

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    failure_record = next(
        record for record in records if record["event"] == "backend_operation_failed"
    )
    assert failure_record["level"] == "error"
    assert failure_record["error_type"] == failure_mode.value
    assert failure_record["retryable"] is retryable
    assert secret not in stream.getvalue()


def test_fake_exposes_configured_delays_to_injected_sleepers() -> None:
    sync_delays: list[float] = []
    async_delays: list[float] = []

    async def record_async_delay(delay: float) -> None:
        async_delays.append(delay)

    backend = FakeBackend(
        FakeBackendConfig(
            initial_delay_seconds=0.1,
            per_item_delay_seconds=0.2,
            per_chunk_delay_seconds=0.3,
            sync_sleeper=sync_delays.append,
            async_sleeper=record_async_delay,
        )
    )
    backend.load()
    backend.embed(EmbeddingRequest(input_texts=("one", "two")))

    async def consume() -> None:
        async for _ in backend.generate(
            GenerationRequest(messages=(GenerationMessage(role="user", content="Hi"),))
        ):
            pass

    asyncio.run(consume())

    assert sync_delays == [0.1, 0.4]
    assert async_delays == [0.1, 0.3, 0.3]


def test_fake_logs_safe_metadata_without_payloads() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    secret = "PRIVATE DOCUMENT CONTENT"
    backend = FakeBackend(FakeBackendConfig(generation_chunks=(secret,)))
    backend.load()
    backend.embed(EmbeddingRequest(input_texts=(secret,)))

    async def consume() -> None:
        async for _ in backend.generate(
            GenerationRequest(
                messages=(GenerationMessage(role="user", content=secret),)
            )
        ):
            pass

    asyncio.run(consume())
    records = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert secret not in stream.getvalue()
    assert {record["event"] for record in records} >= {
        "backend_load_completed",
        "backend_operation_started",
        "backend_operation_completed",
        "backend_stream_completed",
    }
