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
    assert "local_mechanics" not in backend.metadata.model_revision


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


def test_generation_remains_an_explicit_boundary_until_streaming_sub_pr() -> None:
    backend = LocalCausalBackend(FIXTURE_DIR)
    backend.load()
    request = GenerationRequest(messages=(GenerationMessage("user", "hello"),))

    async def consume() -> None:
        with pytest.raises(BackendRequestError):
            async for _event in backend.generate(request):
                pass

    asyncio.run(consume())
