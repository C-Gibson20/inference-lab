import asyncio
import json
from io import StringIO

import pytest

from inference_lab.backends.runtime.errors import (
    BackendExecutionError,
    BackendNotReadyError,
    BackendResourceError,
)
from inference_lab.backends.runtime.fake import (
    FakeBackend,
    FakeBackendConfig,
    FakeFailureMode,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig
from inference_lab.backends.runtime.types import (
    BackendOperation,
    EmbeddingRequest,
    GenerationEventType,
    GenerationMessage,
    GenerationRequest,
    RerankCandidate,
    RerankRequest,
)


def test_fake_requires_load_before_inference() -> None:
    backend = FakeBackend()

    with pytest.raises(BackendNotReadyError):
        backend.embed(EmbeddingRequest(input_texts=("one",)))


def test_fake_embedding_is_deterministic_and_ordered() -> None:
    backend = FakeBackend()
    backend.load()

    result = backend.embed(EmbeddingRequest(input_texts=("one", "two")))

    assert result.vectors == ((0.0, 1.0, 2.0), (1.0, 2.0, 3.0))
    assert result.usage is not None
    assert result.usage.input_tokens == 2


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
