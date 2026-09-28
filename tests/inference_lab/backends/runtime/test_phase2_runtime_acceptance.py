"""RUN-01 phase acceptance for the public runtime boundary and deterministic fake."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from io import StringIO

import pytest

from inference_lab.backends.runtime import (
    BackendClosedError,
    BackendExecutionError,
    BackendMetadata,
    BackendNotReadyError,
    BackendOperation,
    BackendProtocolError,
    BackendRequestError,
    BackendResourceError,
    EmbeddingBackend,
    EmbeddingOptions,
    EmbeddingRequest,
    EmbeddingResult,
    FakeBackend,
    FakeBackendConfig,
    FakeFailureMode,
    GenerationEvent,
    GenerationEventType,
    GenerationMessage,
    GenerationOptions,
    GenerationRequest,
    GeneratorBackend,
    LifecycleBackend,
    RerankCandidate,
    RerankerBackend,
    RerankRequest,
    ScoredCandidate,
    validate_generation_stream,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig


def _collect(stream: AsyncIterator[GenerationEvent]) -> list[GenerationEvent]:
    async def collect() -> list[GenerationEvent]:
        return [item async for item in stream]

    return asyncio.run(collect())


def test_fake_backend_is_a_public_lifecycle_and_operation_boundary() -> None:
    backend = FakeBackend()
    assert isinstance(backend, LifecycleBackend)
    assert isinstance(backend, EmbeddingBackend)
    assert isinstance(backend, RerankerBackend)
    assert isinstance(backend, GeneratorBackend)
    with pytest.raises(BackendNotReadyError):
        backend.embed(EmbeddingRequest(input_texts=("query",)))
    backend.load()
    assert backend.is_ready
    backend.close()
    with pytest.raises(BackendClosedError):
        backend.load()

    only_embed = BackendMetadata(
        backend_name="fake",
        model_name="fixture",
        model_revision="1",
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(),
        embedding_dimensions=3,
    )
    restricted = FakeBackend(FakeBackendConfig(metadata=only_embed))
    restricted.load()
    with pytest.raises(BackendRequestError, match="rerank"):
        restricted.rerank(
            RerankRequest(
                query="q",
                candidates=(RerankCandidate(candidate_id="a", candidate_text="A"),),
            )
        )

    delays: list[float] = []
    delayed = FakeBackend(
        FakeBackendConfig(
            initial_delay_seconds=0.1,
            per_item_delay_seconds=0.2,
            sync_sleeper=delays.append,
        )
    )
    delayed.load()
    delayed.embed(EmbeddingRequest(input_texts=("one", "two")))
    assert delays == [0.1, 0.4]


def test_fake_composes_embedding_reranking_and_bounded_generation() -> None:
    backend = FakeBackend(
        FakeBackendConfig(
            rerank_scores=(("a", 0.1), ("b", 0.9)),
            generation_chunks=("alpha EN", "D ignored"),
        )
    )
    backend.load()
    embedded = backend.embed(EmbeddingRequest(input_texts=("one", "two")))
    assert embedded.vectors == ((0.0, 1.0, 2.0), (1.0, 2.0, 3.0))
    reranked = backend.rerank(
        RerankRequest(
            query="q",
            candidates=(
                RerankCandidate(candidate_id="a", candidate_text="A"),
                RerankCandidate(candidate_id="b", candidate_text="B"),
            ),
            top_k=1,
        )
    )
    assert [(item.candidate_id, item.rank) for item in reranked.scored_candidates] == [
        ("b", 1)
    ]
    events = _collect(
        backend.generate(
            GenerationRequest(
                messages=(GenerationMessage(role="user", content="question"),),
                options=GenerationOptions(stop_sequences=("END",)),
            )
        )
    )
    assert "".join(item.text_delta or "" for item in events) == "alpha "
    assert events[-1].event_type is GenerationEventType.COMPLETED
    assert events[-1].sequence_number == 1


def test_stream_boundary_requires_contiguous_deltas_and_completion() -> None:
    async def malformed() -> AsyncIterator[GenerationEvent]:
        yield GenerationEvent(
            event_type=GenerationEventType.DELTA,
            sequence_number=1,
            text_delta="partial",
        )

    with pytest.raises(BackendProtocolError, match="contiguous"):
        _collect(validate_generation_stream(malformed()))

    async def incomplete() -> AsyncIterator[GenerationEvent]:
        yield GenerationEvent(
            event_type=GenerationEventType.DELTA,
            sequence_number=0,
            text_delta="partial",
        )

    with pytest.raises(BackendProtocolError, match="without completion"):
        _collect(validate_generation_stream(incomplete()))


def test_fake_failure_modes_keep_typed_errors_and_hide_payloads() -> None:
    secret = "PRIVATE DOCUMENT PAYLOAD"
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    error_types = {
        FakeFailureMode.REQUEST: BackendRequestError,
        FakeFailureMode.EXECUTION: BackendExecutionError,
        FakeFailureMode.RESOURCE: BackendResourceError,
    }
    for mode, error_type in error_types.items():
        backend = FakeBackend(
            FakeBackendConfig(
                failure_mode=mode,
                fail_operation=BackendOperation.EMBED,
            )
        )
        backend.load()
        with pytest.raises(error_type):
            backend.embed(EmbeddingRequest(input_texts=(secret,)))
    assert secret not in stream.getvalue()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    failures = [
        item for item in records if item["event"] == "backend_operation_failed"
    ]
    assert len(failures) == 3
    assert all(item["level"] == "error" for item in failures)


def test_partial_generation_survives_execution_failure_and_cancellation() -> None:
    failing = FakeBackend(
        FakeBackendConfig(
            generation_chunks=("visible", "hidden"),
            failure_mode=FakeFailureMode.EXECUTION,
            fail_operation=BackendOperation.GENERATE,
            fail_after_generation_chunks=1,
        )
    )
    failing.load()
    request = GenerationRequest(
        messages=(GenerationMessage(role="user", content="q"),)
    )

    async def read_partial() -> list[str | None]:
        output: list[str | None] = []
        with pytest.raises(BackendExecutionError):
            async for item in failing.generate(request):
                output.append(item.text_delta)
        return output

    assert asyncio.run(read_partial()) == ["visible"]

    async def cancelled() -> AsyncIterator[GenerationEvent]:
        yield GenerationEvent(
            event_type=GenerationEventType.DELTA,
            sequence_number=0,
            text_delta="visible",
        )
        raise asyncio.CancelledError

    async def read_cancelled() -> list[str | None]:
        output: list[str | None] = []
        with pytest.raises(asyncio.CancelledError):
            async for item in validate_generation_stream(cancelled()):
                output.append(item.text_delta)
        return output

    assert asyncio.run(read_cancelled()) == ["visible"]


def test_contract_values_reject_coercion_nonfinite_and_mismatched_shapes() -> None:
    with pytest.raises(TypeError):
        EmbeddingOptions(dimensions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GenerationOptions(temperature=float("nan"))
    metadata = BackendMetadata(
        backend_name="fake",
        model_name="fixture",
        model_revision="1",
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(),
        embedding_dimensions=2,
    )
    with pytest.raises(ValueError):
        EmbeddingResult(vectors=((1.0,),), model_metadata=metadata)
    with pytest.raises(ValueError):
        ScoredCandidate(candidate_id="a", score=float("inf"), rank=1)
    with pytest.raises(ValueError):
        FakeBackendConfig(initial_delay_seconds=-0.1)
