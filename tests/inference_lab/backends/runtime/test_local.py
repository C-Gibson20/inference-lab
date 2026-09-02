import asyncio
from pathlib import Path

import pytest

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendClosedError,
    BackendExecutionError,
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


def test_generation_remains_an_explicit_boundary_until_streaming_sub_pr() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(messages=(GenerationMessage("user", "hello"),))

    async def consume() -> None:
        with pytest.raises(BackendRequestError):
            async for _event in backend.generate(request):
                pass

    asyncio.run(consume())
