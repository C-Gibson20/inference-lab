"""Measure load and inference throughput for the local Qwen embedding backend."""

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import transformers

from inference_lab._clock import utc_now
from inference_lab.backends.runtime import (
    EmbeddingRequest,
    MemoryMeasurement,
    QwenEmbeddingBackend,
    qwen_embedding,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig

_DEFAULT_INPUTS = (
    Path(__file__).parents[3] / "benchmarks" / "run-04" / "throughput-inputs.jsonl"
)


def _load_inputs(path: Path) -> tuple[tuple[str, ...], str]:
    """Read fixed JSONL text inputs and return texts plus their file digest."""
    contents = path.read_bytes()
    texts: list[str] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(contents.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"input record {line_number} must be an object")
        input_id = value.get("input_id")
        text = value.get("text")
        if not isinstance(input_id, str) or not input_id.strip():
            raise ValueError(f"input record {line_number} has an invalid input_id")
        if input_id in identifiers:
            raise ValueError(f"input record {line_number} repeats an input_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"input record {line_number} has invalid text")
        identifiers.add(input_id)
        texts.append(text)
    if not texts:
        raise ValueError("input file must contain at least one text")
    return tuple(texts), hashlib.sha256(contents).hexdigest()


def _measurement_record(measurement: MemoryMeasurement) -> dict[str, object]:
    """Convert one RUN-03 observation to bounded JSON metadata."""
    return {
        "measurement_id": measurement.measurement_id,
        "component_id": measurement.component_id,
        "component_role": measurement.component_role.value,
        "category": measurement.category,
        "memory_space": measurement.memory_space.value,
        "device_id": measurement.device_id,
        "status": measurement.status,
        "value_bytes": measurement.value_bytes,
        "adapter_name": measurement.adapter.adapter_name,
        "adapter_version": measurement.adapter.adapter_version,
        "calibration_revision": measurement.adapter.calibration_revision,
        "observed_at": measurement.observed_at.isoformat(),
    }


def run_benchmark(
    model_path: Path,
    *,
    inputs_path: Path = _DEFAULT_INPUTS,
    warmup_iterations: int = 2,
    measured_iterations: int = 5,
) -> dict[str, object]:
    """Run a cold load, warmups, and repeated fixed-input inference batches."""
    if (
        not isinstance(warmup_iterations, int)
        or isinstance(warmup_iterations, bool)
        or warmup_iterations < 0
    ):
        raise ValueError("warmup_iterations must be a non-negative integer")
    if (
        not isinstance(measured_iterations, int)
        or isinstance(measured_iterations, bool)
        or measured_iterations <= 0
    ):
        raise ValueError("measured_iterations must be a positive integer")
    texts, inputs_digest = _load_inputs(inputs_path)
    backend = QwenEmbeddingBackend(model_path)
    try:
        load_started = time.perf_counter_ns()
        backend.load()
        load_duration_ms = (time.perf_counter_ns() - load_started) / 1_000_000

        memory = tuple(
            _measurement_record(measurement) for measurement in backend.measure_memory()
        )
        request = EmbeddingRequest(input_texts=texts, purpose="document")
        for _ in range(warmup_iterations):
            backend.embed(request)

        durations_ns: list[int] = []
        total_input_tokens = 0
        output_dimensions: int | None = None
        embedding_profile: dict[str, object] | None = None
        reference_vectors: torch.Tensor | None = None
        for _ in range(measured_iterations):
            started = time.perf_counter_ns()
            result = backend.embed(request)
            durations_ns.append(time.perf_counter_ns() - started)
            if len(result.vectors) != len(texts):
                raise RuntimeError("backend returned an invalid vector count")
            if result.usage is None:
                raise RuntimeError("backend did not report input token usage")
            if result.model_metadata.model_revision != backend.metadata.model_revision:
                raise RuntimeError("backend returned mismatched model metadata")
            vectors = torch.tensor(result.vectors, dtype=torch.float32)
            if not torch.isfinite(vectors).all():
                raise RuntimeError("backend returned non-finite vectors")
            if reference_vectors is None:
                reference_vectors = vectors
            elif not torch.allclose(reference_vectors, vectors, rtol=1e-5, atol=1e-6):
                raise RuntimeError("backend vectors were not repeatable")
            total_input_tokens += result.usage.input_tokens
            output_dimensions = result.model_metadata.embedding_dimensions
            profile = result.embedding_profile
            embedding_profile = {
                "profile_version": profile.profile_version,
                "tokenizer_id": profile.tokenizer_id,
                "tokenizer_revision": profile.tokenizer_revision,
                "query_prefix": profile.query_prefix,
                "document_prefix": profile.document_prefix,
                "pooling_strategy": profile.pooling_strategy,
                "input_normalisation": profile.input_normalisation,
                "max_input_tokens": profile.max_input_tokens,
                "over_limit_policy": profile.over_limit_policy,
                "l2_normalised": profile.l2_normalised,
                "inference_dtype": profile.inference_dtype,
                "device_type": profile.device_type,
            }

        if reference_vectors is None or output_dimensions is None:
            raise RuntimeError("backend returned no embedding vectors")
        if reference_vectors.shape != (len(texts), output_dimensions):
            raise RuntimeError("backend returned invalid vector dimensions")
        reversed_result = backend.embed(
            EmbeddingRequest(input_texts=tuple(reversed(texts)))
        )
        reversed_vectors = torch.tensor(reversed_result.vectors, dtype=torch.float32)
        if not torch.allclose(
            reference_vectors.flip(0), reversed_vectors, rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError("backend did not preserve request order")

        elapsed_seconds = sum(durations_ns) / 1_000_000_000
        total_embeddings = len(texts) * measured_iterations
        return {
            "report_schema_version": "1.0.0",
            "measured_at": utc_now().isoformat(),
            "model": {
                "backend_name": backend.metadata.backend_name,
                "model_name": backend.metadata.model_name,
                "model_revision": backend.metadata.model_revision,
                "artifact_identity_verified": True,
                "output_dimensions": output_dimensions,
            },
            "acceptance": {
                "vector_count_per_request": len(texts),
                "output_dimensions": output_dimensions,
                "finite_vectors": True,
                "repeatability_iterations": measured_iterations,
                "repeatability_tolerance": "rtol=1e-5, atol=1e-6",
                "reversed_batch_order": "passed",
            },
            "embedding_profile": embedding_profile,
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.system(),
                "platform_release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "device_type": "cpu",
                "inference_dtype": "float32",
            },
            "inputs": {
                "sha256": inputs_digest,
                "input_count_per_batch": len(texts),
                "warmup_iterations": warmup_iterations,
                "measured_iterations": measured_iterations,
                "request_batch_size": len(texts),
                "backend_internal_batch_size": qwen_embedding._INTERNAL_BATCH_SIZE,
                "input_tokens_total": total_input_tokens,
            },
            "load": {"cold_load_duration_ms": load_duration_ms},
            "measurement_method": (
                "time.perf_counter_ns; one cold local load, configured warmups, "
                "then sequential repeated batches"
            ),
            "inference": {
                "elapsed_ms_total": elapsed_seconds * 1000,
                "median_batch_duration_ms": statistics.median(durations_ns) / 1_000_000,
                "embeddings_per_second": total_embeddings / elapsed_seconds,
                "input_tokens_per_second": total_input_tokens / elapsed_seconds,
            },
            "memory": list(memory),
            "memory_limitations": [
                "Only inspectable model tensor storage is measured.",
                "Allocator reservation, process RSS, activations, and framework overhead are not measured.",
            ],
        }
    finally:
        backend.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse benchmark options and write a path-free JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=_DEFAULT_INPUTS)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measured-iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    configure_logging(LoggingConfig(level="WARNING", format="json"), stream=sys.stderr)
    try:
        report = run_benchmark(
            arguments.model_path,
            inputs_path=arguments.inputs,
            warmup_iterations=arguments.warmup_iterations,
            measured_iterations=arguments.measured_iterations,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "embedding_benchmark_failed",
                    "error_type": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 1

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote embedding benchmark report ({arguments.output.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
