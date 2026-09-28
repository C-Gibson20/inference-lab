"""RUN-02 local-model acceptance for capabilities, limits, cache, and privacy."""

import asyncio
import json
from io import StringIO
from pathlib import Path

import pytest

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendNotReadyError,
    BackendOperation,
    BackendRequestError,
    GenerationMessage,
    GenerationOptions,
    GenerationRequest,
    GeneratorBackend,
    LifecycleBackend,
    LocalCausalBackend,
)
from inference_lab.backends.runtime._local_mechanics import (
    CacheObservationOutcome,
    LocalMechanicsHarness,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig

FIXTURE_DIR = Path("tests/fixtures/backends/runtime/local_mechanics")


def test_local_backend_advertises_only_supported_capabilities_and_rejects_sampling() -> None:
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

    backend.load()
    request = GenerationRequest(
        messages=(GenerationMessage("user", "hello"),),
        options=GenerationOptions(temperature=1.0),
    )
    with pytest.raises(BackendRequestError):
        backend.generate(request)
    backend.close()


def test_local_backend_lifecycle_loads_once_and_releases_resources() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    assert not backend.is_ready
    with pytest.raises(BackendNotReadyError):
        backend.generate(
            GenerationRequest(messages=(GenerationMessage("user", "hello"),))
        )

    backend.load()
    model = backend._model
    tokeniser = backend._tokeniser
    backend.load()
    assert backend.is_ready
    assert backend._model is model
    assert backend._tokeniser is tokeniser

    backend.close()
    backend.close()
    assert backend.is_closed
    assert backend._model is None
    assert backend._tokeniser is None


def test_local_backend_enforces_context_and_output_limits() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR, context_limit_tokens=8)
    backend.load()
    within_limit = GenerationRequest(
        messages=(GenerationMessage("user", "hello world"),),
        options=GenerationOptions(maximum_output_tokens=2),
    )
    prepared = backend._prepare_generation(within_limit)
    assert prepared.input_tokens == 5
    assert prepared.available_output_tokens == 2

    too_large = LocalCausalBackend(FIXTURE_DIR, context_limit_tokens=5)
    too_large.load()
    request = GenerationRequest(
        messages=(GenerationMessage("user", "hello world"),)
    )
    with pytest.raises(BackendRequestError):
        too_large._prepare_generation(request)
    backend.close()
    too_large.close()


def test_local_mechanics_reuses_exact_repeated_prefix_without_recomputation() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(messages=(GenerationMessage("user", "hello"),))

    observation = LocalMechanicsHarness(backend).observe_prefixes(request, request)

    assert observation.rendered_input_tokens == 4
    assert observation.common_prefix_tokens == 4
    assert observation.tokens_evaluated_without_cache == 4
    assert observation.tokens_evaluated_to_create_cache == 4
    assert observation.tokens_evaluated_after_cache == 0
    assert observation.cache_outcomes == (
        CacheObservationOutcome.CREATED,
        CacheObservationOutcome.REUSED,
        CacheObservationOutcome.CLEARED,
    )
    assert observation.logits_equivalent is True
    assert observation.selection_equivalent is True
    backend.close()


def test_local_mechanics_rejects_cache_from_incompatible_identity() -> None:
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
    backend.close()


def test_local_backend_logs_remain_structured_and_payload_free() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(
        messages=(
            GenerationMessage("user", "private-prompt-marker /Users/private/prompt.txt"),
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
        record
        for line in stream.getvalue().splitlines()
        if (record := json.loads(line))["module"].startswith(
            "inference_lab.backends.runtime."
        )
    ]
    payload = json.dumps(records)
    assert "private-prompt-marker" not in payload
    assert "/Users/private/prompt.txt" not in payload
    assert [record["event"] for record in records] == [
        "backend_load_completed",
        "backend_generation_started",
        "backend_cache_observation_completed",
        "backend_close_completed",
    ]
    assert all(
        {
            "timestamp",
            "level",
            "event",
            "module",
            "pipeline_stage",
            "document_id",
            "corpus_id",
            "corpus_version",
        }
        <= record.keys()
        for record in records
    )
