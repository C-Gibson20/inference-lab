"""Public acceptance tests for the fake and local embedding backends."""

import json
import math
import shutil
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import cast

import pytest

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendClosedError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendOperation,
    EmbeddingBackend,
    EmbeddingOptions,
    EmbeddingRequest,
    FakeBackend,
    LifecycleBackend,
    QwenEmbeddingBackend,
    qwen_embedding,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig

_FIXTURE_DIR = (
    Path(__file__).parents[4]
    / "tests"
    / "fixtures"
    / "backends"
    / "runtime"
    / "local_embedding"
)


@pytest.fixture
def local_embedding_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[QwenEmbeddingBackend]:
    """Load the pinned, tiny Qwen-architecture fixture through Transformers."""
    manifest_object = json.loads(
        (_FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = cast(dict[str, object], manifest_object)
    fixture_files = cast(dict[str, str], manifest["files"])
    fixture_revision = cast(str, manifest["revision"])
    vocab_size = cast(int, manifest["vocab_size"])
    monkeypatch.setattr(qwen_embedding, "_MODEL_NAME", "fixture/qwen3-mini")
    monkeypatch.setattr(qwen_embedding, "_MODEL_REVISION", fixture_revision)
    monkeypatch.setattr(qwen_embedding, "_EMBEDDING_DIMENSIONS", 32)
    monkeypatch.setattr(qwen_embedding, "_MIN_EMBEDDING_DIMENSIONS", 8)
    monkeypatch.setattr(qwen_embedding, "_TOKENIZER_VOCAB_SIZE", vocab_size)
    monkeypatch.setattr(qwen_embedding, "_MAX_INPUT_TOKENS", 64)
    monkeypatch.setattr(
        qwen_embedding,
        "_ARTEFACT_DIGESTS",
        tuple(sorted(fixture_files.items())),
    )
    backend = QwenEmbeddingBackend(_FIXTURE_DIR)
    try:
        yield backend
    finally:
        backend.close()


def test_fake_and_local_backends_obey_lifecycle_and_embedding_contracts(
    local_embedding_backend: QwenEmbeddingBackend,
) -> None:
    backends: tuple[EmbeddingBackend, ...] = (
        FakeBackend(),
        local_embedding_backend,
    )
    for backend in backends:
        assert isinstance(backend, LifecycleBackend)
        assert isinstance(backend, EmbeddingBackend)
        assert BackendOperation.EMBED in backend.metadata.supported_operations
        with pytest.raises(BackendNotReadyError):
            backend.embed(EmbeddingRequest(input_texts=("SARM1 inhibition",)))

        backend.load()
        result = backend.embed(EmbeddingRequest(input_texts=("SARM1 inhibition",)))
        assert len(result.vectors) == 1
        assert len(result.vectors[0]) == result.model_metadata.embedding_dimensions
        assert all(math.isfinite(value) for value in result.vectors[0])
        assert result.embedding_profile.tokenizer_id

        backend.close()
        backend.close()
        with pytest.raises(BackendClosedError):
            backend.embed(EmbeddingRequest(input_texts=("SARM1 inhibition",)))


def test_local_fixture_is_repeatable_and_internal_batches_preserve_order(
    local_embedding_backend: QwenEmbeddingBackend,
) -> None:
    local_embedding_backend.load()
    request = EmbeddingRequest(
        input_texts=(
            "SARM1 inhibition reduces neuroinflammation",
            "alpha beta gamma",
            "epsilon zeta eta",
        )
    )
    first = local_embedding_backend.embed(request)
    repeated = local_embedding_backend.embed(request)
    individual = tuple(
        local_embedding_backend.embed(EmbeddingRequest(input_texts=(text,))).vectors[0]
        for text in request.input_texts
    )

    assert first.vectors == repeated.vectors
    for batched, separate in zip(first.vectors, individual, strict=True):
        assert batched == pytest.approx(separate, rel=1e-5, abs=1e-6)
    assert len(first.vectors) == len(request.input_texts)
    assert first.model_metadata.embedding_dimensions == 32
    assert all(math.isfinite(value) for vector in first.vectors for value in vector)


def test_local_fixture_applies_options_and_returns_path_free_identity(
    local_embedding_backend: QwenEmbeddingBackend,
) -> None:
    local_embedding_backend.load()
    result = local_embedding_backend.embed(
        EmbeddingRequest(
            input_texts=("SARM1 inhibition",),
            options=EmbeddingOptions(dimensions=16, normalise=True),
            purpose="query",
        )
    )

    assert len(result.vectors[0]) == 16
    assert result.model_metadata.embedding_dimensions == 16
    assert math.sqrt(sum(value**2 for value in result.vectors[0])) == pytest.approx(1.0)
    assert result.embedding_profile.tokenizer_id == "fixture/qwen3-mini"
    assert result.embedding_profile.query_prefix.startswith("Instruct:")
    assert result.embedding_profile.document_prefix == ""
    assert result.embedding_profile.l2_normalised is True
    assert result.embedding_profile.device_type == "cpu"
    assert "/" not in result.embedding_profile.tokenizer_revision


def test_local_fixture_truncation_logs_counts_without_text(
    local_embedding_backend: QwenEmbeddingBackend,
) -> None:
    local_embedding_backend.load()
    stream = StringIO()
    configure_logging(LoggingConfig(level="WARNING", format="json"), stream=stream)
    secret = "SENSITIVE_FIXTURE_TEXT " + "alpha " * 80
    result = local_embedding_backend.embed(
        EmbeddingRequest(input_texts=(secret,), purpose="query")
    )
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    warning = next(
        record for record in records if record["event"] == "embedding_input_truncated"
    )

    assert warning["affected_input_count"] == 1
    assert warning["original_token_count"] > warning["token_limit"]
    assert warning["retained_token_count"] == warning["token_limit"] == 64
    assert result.usage is not None
    assert result.usage.input_tokens == 64
    assert secret not in stream.getvalue()


def test_local_fixture_rejects_corrupt_model_files_with_generic_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_object = json.loads(
        (_FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = cast(dict[str, object], manifest_object)
    fixture_files = cast(dict[str, str], manifest["files"])
    monkeypatch.setattr(qwen_embedding, "_MODEL_NAME", "fixture/qwen3-mini")
    monkeypatch.setattr(
        qwen_embedding, "_MODEL_REVISION", cast(str, manifest["revision"])
    )
    monkeypatch.setattr(qwen_embedding, "_EMBEDDING_DIMENSIONS", 32)
    monkeypatch.setattr(qwen_embedding, "_MIN_EMBEDDING_DIMENSIONS", 8)
    monkeypatch.setattr(
        qwen_embedding, "_TOKENIZER_VOCAB_SIZE", cast(int, manifest["vocab_size"])
    )
    monkeypatch.setattr(
        qwen_embedding,
        "_ARTEFACT_DIGESTS",
        tuple(sorted(fixture_files.items())),
    )
    corrupt_dir = tmp_path / "corrupt-fixture"
    shutil.copytree(_FIXTURE_DIR, corrupt_dir)
    model_path = corrupt_dir / "model.safetensors"
    model_path.write_bytes(model_path.read_bytes() + b"changed")

    backend = QwenEmbeddingBackend(corrupt_dir)
    with pytest.raises(BackendExecutionError) as caught:
        backend.load()
    assert str(corrupt_dir) not in str(caught.value)
    assert "changed" not in str(caught.value)


def test_local_backend_advertises_batching_and_token_usage(
    local_embedding_backend: QwenEmbeddingBackend,
) -> None:
    local_embedding_backend.load()
    assert local_embedding_backend.metadata.capabilities == (
        BackendCapability.BATCHING,
        BackendCapability.TOKEN_USAGE,
    )
    result = local_embedding_backend.embed(
        EmbeddingRequest(input_texts=("alpha beta", "gamma delta"))
    )
    assert result.usage is not None
    assert result.usage.input_tokens > 0
