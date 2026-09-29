"""Acceptance coverage for embedding memory and measurement reporting."""

import json
from pathlib import Path
from typing import cast

import pytest

from inference_lab.backends.runtime import (
    MemoryAccountingAdapter,
    MemoryComponentRole,
    QwenEmbeddingBackend,
    qwen_embedding,
)
from scripts.inference_lab.backends.benchmark_local_embedding import run_benchmark

_FIXTURE_DIR = (
    Path(__file__).parents[4]
    / "tests"
    / "fixtures"
    / "backends"
    / "runtime"
    / "local_embedding"
)
_INPUTS = (
    Path(__file__).parents[4] / "benchmarks" / "run-04" / "throughput-inputs.jsonl"
)


def test_benchmark_reports_pinned_memory_and_safe_repeatable_metadata(
    monkeypatch: pytest.MonkeyPatch,
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
    assert isinstance(QwenEmbeddingBackend(_FIXTURE_DIR), MemoryAccountingAdapter)

    report = run_benchmark(
        _FIXTURE_DIR,
        inputs_path=_INPUTS,
        warmup_iterations=0,
        measured_iterations=2,
    )

    model_memory = report["memory"]
    assert isinstance(model_memory, list)
    assert len(model_memory) == 1
    observation = model_memory[0]
    assert isinstance(observation, dict)
    assert observation["component_role"] == MemoryComponentRole.EMBEDDER.value
    assert observation["category"] == "model_weights"
    assert observation["status"] == "measured"
    assert isinstance(observation["value_bytes"], int)
    assert observation["value_bytes"] > 0

    encoded = json.dumps(report, sort_keys=True)
    assert str(_FIXTURE_DIR) not in encoded
    for line in _INPUTS.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["text"] not in encoded
    inference = report["inference"]
    assert isinstance(inference, dict)
    assert inference["embeddings_per_second"] > 0
    assert inference["input_tokens_per_second"] > 0
    load = report["load"]
    assert isinstance(load, dict)
    assert load["cold_load_duration_ms"] > 0
    inputs = report["inputs"]
    assert isinstance(inputs, dict)
    assert inputs["input_count_per_batch"] == 8
    assert inputs["measured_iterations"] == 2
    assert inputs["backend_internal_batch_size"] == 2
    model = report["model"]
    assert isinstance(model, dict)
    assert model["artifact_identity_verified"] is True
    assert model["model_revision"] == manifest["revision"]
    profile = report["embedding_profile"]
    assert isinstance(profile, dict)
    assert profile["tokenizer_revision"] == manifest["revision"]
    environment = report["environment"]
    assert isinstance(environment, dict)
    assert environment["inference_dtype"] == "float32"
    assert environment["device_type"] == "cpu"
    assert report["measurement_method"]
    acceptance = report["acceptance"]
    assert isinstance(acceptance, dict)
    assert acceptance["vector_count_per_request"] == 8
    assert acceptance["output_dimensions"] == 32
    assert acceptance["finite_vectors"] is True
    assert acceptance["repeatability_iterations"] == 2
    assert acceptance["reversed_batch_order"] == "passed"


def test_benchmark_rejects_invalid_iteration_counts() -> None:
    with pytest.raises(ValueError):
        run_benchmark(_FIXTURE_DIR, warmup_iterations=-1)
    with pytest.raises(ValueError):
        run_benchmark(_FIXTURE_DIR, measured_iterations=0)
