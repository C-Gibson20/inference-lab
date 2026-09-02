import asyncio
import json
from collections.abc import AsyncGenerator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendClosedError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendOperation,
    BackendRequestError,
    GenerationEvent,
    GenerationMessage,
    GenerationOptions,
    GenerationRequest,
    GeneratorBackend,
    LifecycleBackend,
    LocalCausalBackend,
)
from inference_lab.backends.runtime.local import (
    CacheObservationOutcome,
    LocalMechanicsHarness,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig

FIXTURE_DIR = Path("tests/fixtures/backends/runtime/local_mechanics")


def test_local_backend_exposes_generation_only_metadata() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)

    assert isinstance(backend, LifecycleBackend)
    assert isinstance(backend, GeneratorBackend)
    assert backend.metadata.supported_operations == (BackendOperation.GENERATE,)
    assert backend.metadata.capabilities == (
        BackendCapability.STREAMING,
        BackendCapability.CANCELLATION,
        BackendCapability.TOKEN_USAGE,
    )
    assert backend.metadata.embedding_dimensions is None
    assert "prompt-v1" in backend.metadata.model_revision


@pytest.mark.parametrize(
    "artefact_dir",
    ["https://example.invalid/model", "s3://bucket/model", "//server/model"],
)
def test_local_backend_rejects_non_local_paths(artefact_dir: str) -> None:
    with pytest.raises(BackendRequestError):
        LocalCausalBackend(artefact_dir)


def test_local_backend_rejects_invalid_path_types() -> None:
    with pytest.raises(TypeError, match="artefact_dir"):
        LocalCausalBackend(42)  # type: ignore[arg-type]


def test_local_backend_is_unloaded_until_explicit_load() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)

    assert not backend.is_ready
    assert not backend.is_closed
    with pytest.raises(BackendNotReadyError):
        backend.generate(
            GenerationRequest(messages=(GenerationMessage("user", "hello"),))
        )


def test_local_backend_load_is_idempotent_and_releases_resources_on_close() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)

    backend.load()
    assert backend.effective_context_limit_tokens == 64
    model = backend._model
    tokeniser = backend._tokeniser
    backend.load()

    assert backend.is_ready
    assert backend._model is model
    assert backend._tokeniser is tokeniser

    backend.close()
    backend.close()

    assert not backend.is_ready
    assert backend.is_closed
    assert backend._model is None
    assert backend._tokeniser is None


def test_local_backend_cannot_load_or_infer_after_close() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.close()

    with pytest.raises(BackendClosedError):
        backend.load()
    with pytest.raises(BackendClosedError):
        backend.generate(
            GenerationRequest(messages=(GenerationMessage("user", "hello"),))
        )


def test_local_backend_fails_closed_when_artefacts_are_unavailable(
    tmp_path: Path,
) -> None:
    backend = LocalCausalBackend(tmp_path / "missing")

    with pytest.raises(BackendExecutionError) as error:
        backend.load()

    assert str(error.value) == "The backend failed while executing the request."
    assert error.value.__cause__ is None
    assert not backend.is_ready
    assert backend._model is None
    assert backend._tokeniser is None


def test_local_backend_renders_and_counts_the_complete_prompt() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(messages=(GenerationMessage("user", "hello world"),))

    prepared = backend._prepare_generation(request)

    assert prepared.rendered_prompt == "[BOS][USER] hello world[ASSISTANT]"
    assert prepared.input_ids == (1, 5, 40, 44, 6)
    assert prepared.input_tokens == 5
    assert prepared.available_output_tokens == 59


def test_local_backend_preserves_role_boundaries_and_unicode_deterministically() -> (
    None
):
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(
        messages=(
            GenerationMessage("system", "a"),
            GenerationMessage("user", "  hello\tworld  "),
            GenerationMessage("assistant", "café"),
        )
    )

    first = backend._prepare_generation(request)
    second = backend._prepare_generation(request)

    assert first == second
    assert first.rendered_prompt == (
        "[BOS][SYSTEM] a[USER]   hello\tworld  [ASSISTANT] café[ASSISTANT]"
    )
    assert first.input_tokens == len(first.input_ids)


@pytest.mark.parametrize("option", ["temperature", "top_p", "seed"])
def test_local_backend_rejects_sampling_options(option: str) -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    kwargs: dict[str, object] = {option: 1}
    request = GenerationRequest(
        messages=(GenerationMessage("user", "hello"),),
        options=GenerationOptions(**kwargs),  # type: ignore[arg-type]
    )

    with pytest.raises(BackendRequestError):
        backend.generate(request)


def test_local_backend_caps_output_by_context_and_request_limit() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR, context_limit_tokens=8)
    backend.load()
    request = GenerationRequest(
        messages=(GenerationMessage("user", "hello world"),),
        options=GenerationOptions(maximum_output_tokens=2),
    )

    prepared = backend._prepare_generation(request)

    assert prepared.input_tokens == 5
    assert prepared.available_output_tokens == 2


def test_local_backend_rejects_prompt_with_no_context_for_output() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR, context_limit_tokens=5)
    backend.load()
    request = GenerationRequest(messages=(GenerationMessage("user", "hello world"),))

    with pytest.raises(BackendRequestError):
        backend._prepare_generation(request)


def test_local_backend_streams_greedy_deltas_and_exact_usage() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(
        messages=(GenerationMessage("user", "hello"),),
        options=GenerationOptions(maximum_output_tokens=3),
    )

    async def consume() -> list[GenerationEvent]:
        return [event async for event in backend.generate(request)]

    events = asyncio.run(consume())

    assert [event.sequence_number for event in events] == [0, 1, 2]
    assert all(event.text_delta for event in events[:2])
    completion = events[-1]
    assert completion.finish_reason == "length"
    assert completion.usage is not None
    assert completion.usage.input_tokens == 4
    assert completion.usage.output_tokens == 3
    assert completion.usage.total_tokens == 7


def test_local_backend_generation_is_repeatable() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(
        messages=(GenerationMessage("user", "hello"),),
        options=GenerationOptions(maximum_output_tokens=4),
    )

    async def consume() -> tuple[str, ...]:
        chunks: list[str] = []
        async for event in backend.generate(request):
            if event.text_delta is not None:
                chunks.append(event.text_delta)
        return tuple(chunks)

    assert asyncio.run(consume()) == asyncio.run(consume())


def test_local_backend_buffers_and_excludes_a_stop_sequence() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(
        messages=(GenerationMessage("user", "hello"),),
        options=GenerationOptions(stop_sequences=(" ",)),
    )

    async def consume() -> list[GenerationEvent]:
        return [event async for event in backend.generate(request)]

    events = asyncio.run(consume())

    assert " " not in "".join(event.text_delta or "" for event in events)
    assert events[-1].finish_reason == "stop"
    assert events[-1].usage is not None
    assert events[-1].usage.output_tokens == 3


def test_local_backend_translates_model_failures_without_raw_details() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    marker = "private-model-failure-marker"

    class FailingModel:
        config = SimpleNamespace(eos_token_id=2)

        def __call__(self, **_kwargs: object) -> None:
            raise RuntimeError(marker)

    setattr(backend, "_model", FailingModel())
    request = GenerationRequest(messages=(GenerationMessage("user", "hello"),))

    async def consume() -> None:
        async for _event in backend.generate(request):
            pass

    with pytest.raises(BackendExecutionError) as error:
        asyncio.run(consume())

    assert marker not in str(error.value)
    assert error.value.__cause__ is None


def test_local_backend_cancellation_stops_without_completion() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(
        messages=(GenerationMessage("user", "hello"),),
        options=GenerationOptions(maximum_output_tokens=8),
    )

    async def consume() -> None:
        stream = cast(AsyncGenerator[GenerationEvent, None], backend.generate(request))
        first_event = await anext(stream)
        assert first_event.text_delta is not None

        async def read_next_event() -> GenerationEvent:
            return await anext(stream)

        pending_event = asyncio.create_task(read_next_event())
        pending_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_event

        await stream.aclose()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

        follow_up_events = [event async for event in backend.generate(request)]
        assert follow_up_events[-1].usage is not None

    asyncio.run(consume())


def test_local_backend_logs_are_structured_and_payload_free() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(
        messages=(
            GenerationMessage(
                "user", "private-prompt-marker /Users/private/prompt.txt"
            ),
        ),
        options=GenerationOptions(maximum_output_tokens=2),
    )

    async def consume() -> None:
        async for _event in backend.generate(request):
            pass

    asyncio.run(consume())
    LocalMechanicsHarness(backend).observe_prefixes(request, request)
    backend.close()

    records = [
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if '"module": "inference_lab.backends.runtime.local"' in line
    ]
    assert [record["event"] for record in records] == [
        "backend_load_completed",
        "backend_generation_started",
        "backend_cache_observation_completed",
        "backend_close_completed",
    ]
    forbidden_values = (
        "private-prompt-marker",
        "/Users/private/prompt.txt",
        "private-model-output-marker",
    )
    for record in records:
        assert {
            "timestamp",
            "level",
            "event",
            "module",
            "pipeline_stage",
            "document_id",
            "corpus_version",
        } <= record.keys()
        rendered_record = json.dumps(record)
        assert all(value not in rendered_record for value in forbidden_values)


def test_local_backend_load_failure_log_excludes_path_and_exception_details(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    private_path = tmp_path / "private-model-path-marker"
    backend = LocalCausalBackend(private_path)

    with pytest.raises(BackendExecutionError):
        backend.load()

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "backend_load_failed"
    assert record["failure_category"] == "load"
    assert str(private_path) not in json.dumps(record)
    assert "artefact directory is unavailable" not in json.dumps(record)
    assert "traceback" not in record


def test_mechanics_harness_proves_exact_repeated_prefix_equivalence() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(messages=(GenerationMessage("user", "hello"),))

    observation = LocalMechanicsHarness(backend).observe_prefixes(request, request)

    assert observation.rendered_input_tokens == 4
    assert observation.common_prefix_tokens == 4
    assert observation.tokens_evaluated_without_cache == 4
    assert observation.tokens_evaluated_after_cache == 0
    assert observation.cache_outcomes == (
        CacheObservationOutcome.CREATED,
        CacheObservationOutcome.REUSED,
        CacheObservationOutcome.CLEARED,
    )
    assert observation.logits_equivalent is True
    assert observation.selection_equivalent is True


def test_mechanics_harness_proves_partial_prefix_equivalence() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    first = GenerationRequest(messages=(GenerationMessage("user", "hello"),))
    second = GenerationRequest(messages=(GenerationMessage("user", "hello world"),))

    observation = LocalMechanicsHarness(backend).observe_prefixes(first, second)

    assert observation.common_prefix_tokens == 3
    assert observation.tokens_evaluated_without_cache == 5
    assert observation.tokens_evaluated_after_cache == 2
    assert observation.logits_equivalent is True
    assert observation.selection_equivalent is True


@pytest.mark.parametrize(
    ("first_token_ids", "second_token_ids", "expected_common"),
    [
        ((1, 2, 3), (1, 2, 4), 2),
        ((1, 2), (3, 4), 0),
    ],
)
def test_mechanics_harness_records_one_token_difference_and_no_prefix(
    first_token_ids: tuple[int, ...],
    second_token_ids: tuple[int, ...],
    expected_common: int,
) -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()

    observation = LocalMechanicsHarness(backend).observe_token_prefixes(
        first_token_ids, second_token_ids
    )

    assert observation.common_prefix_tokens == expected_common
    assert observation.tokens_evaluated_without_cache == len(second_token_ids)
    if expected_common:
        assert observation.cache_outcomes == (
            CacheObservationOutcome.CREATED,
            CacheObservationOutcome.REUSED,
            CacheObservationOutcome.CLEARED,
        )
        assert observation.tokens_evaluated_after_cache == (
            len(second_token_ids) - expected_common
        )
        assert observation.logits_equivalent is True
        assert observation.selection_equivalent is True
    else:
        assert observation.cache_outcomes == (
            CacheObservationOutcome.REJECTED,
            CacheObservationOutcome.CLEARED,
        )
        assert observation.tokens_evaluated_after_cache == 0
        assert observation.logits_equivalent is None
        assert observation.selection_equivalent is None


def test_mechanics_harness_rejects_incompatible_cache_identity() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(messages=(GenerationMessage("user", "hello"),))

    observation = LocalMechanicsHarness(backend).observe_prefixes(
        request,
        request,
        cache_revision="incompatible-revision",
    )

    assert observation.cache_outcomes == (
        CacheObservationOutcome.REJECTED,
        CacheObservationOutcome.CLEARED,
    )
    assert observation.tokens_evaluated_after_cache == 0
    assert observation.logits_equivalent is None
    assert observation.selection_equivalent is None


def test_mechanics_observation_is_immutable_and_payload_free() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(messages=(GenerationMessage("user", "hello"),))
    observation = LocalMechanicsHarness(backend).observe_prefixes(request, request)

    with pytest.raises(AttributeError):
        observation.common_prefix_tokens = 0  # type: ignore[misc]
    assert "hello" not in repr(observation)
    assert "1, 5, 40" not in repr(observation)


def test_mechanics_harness_clears_cache_after_failure() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    marker = "private-cache-failure-marker"

    class FailingModel:
        config = SimpleNamespace(eos_token_id=2)

        def __call__(self, **_kwargs: object) -> None:
            raise RuntimeError(marker)

    setattr(backend, "_model", FailingModel())

    with pytest.raises(BackendExecutionError) as error:
        LocalMechanicsHarness(backend).observe_token_prefixes(
            (1, 5, 40),
            (1, 5, 40, 44),
        )

    assert backend._active_cache_state is None
    assert marker not in str(error.value)
    assert error.value.__cause__ is None
