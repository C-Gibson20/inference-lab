"""Measure pinned local BGE reranker candidates against a fixed fixture."""

import argparse
import gc
import hashlib
import importlib
import json
import math
import platform
import statistics
import sys
import time
import types
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from inference_lab._clock import utc_now
from inference_lab.backends.runtime import MemoryComponentRole, MemoryMeasurement
from inference_lab.backends.runtime.memory_pytorch import (
    PyTorchMemoryCalibrationAdapter,
)

_FIXTURE = (
    Path(__file__).parents[3] / "benchmarks" / "run-05" / "candidate-fixture.jsonl"
)
_MANIFEST = (
    Path(__file__).parents[3] / "benchmarks" / "run-05" / "artifact-manifest.json"
)
_MAX_ENCODER_TOKENS = 512
_MAX_DECODER_BODY_TOKENS = 1024
_QUERY_MAX_TOKENS = 256
_INTERNAL_BATCH_SIZE = 4
_MEASURED_ITERATIONS = 3
_WARMUP_ITERATIONS = 1


@dataclass(frozen=True, slots=True)
class Candidate:
    """One fixed query/candidate pair in the development fixture."""

    candidate_id: str
    text: str
    relevant: bool


@dataclass(frozen=True, slots=True)
class Case:
    """One query and its human-labelled candidate passages."""

    case_id: str
    query: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Pinned model identity and local loading policy."""

    name: str
    repository: str
    revision: str
    repository_license: str
    upstream_license: str
    architecture: str
    dtype: torch.dtype


MODEL_SPECS = (
    ModelSpec(
        name="bge-reranker-v2-m3",
        repository="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        repository_license="Apache-2.0",
        upstream_license="MIT (BAAI/bge-m3)",
        architecture="encoder_sequence_classification",
        dtype=torch.float32,
    ),
    ModelSpec(
        name="bge-reranker-v2-gemma",
        repository="BAAI/bge-reranker-v2-gemma",
        revision="1787044f8b6fb740a9de4557c3a12377f84d9e17",
        repository_license="Apache-2.0",
        upstream_license="Gemma Terms of Use (google/gemma-2b)",
        architecture="decoder_yes_logit",
        dtype=torch.bfloat16,
    ),
    ModelSpec(
        name="bge-reranker-v2-minicpm-layerwise",
        repository="BAAI/bge-reranker-v2-minicpm-layerwise",
        revision="47b5332b296c4d8cb6ee2c60502cc62a0d708881",
        repository_license="Apache-2.0",
        upstream_license="General Model License (OpenBMB MiniCPM)",
        architecture="decoder_layerwise_score",
        dtype=torch.bfloat16,
    ),
)


class _Tokenizer(Protocol):
    """Transformers tokeniser methods used by the candidate pilot."""

    bos_token_id: int | None
    padding_side: str

    def __call__(
        self,
        text: str | list[str],
        text_pair: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, object]: ...

    def prepare_for_model(
        self,
        ids: list[int],
        pair_ids: list[int],
        **kwargs: object,
    ) -> dict[str, object]: ...

    def pad(
        self,
        encoded_inputs: list[dict[str, list[int]]],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]: ...


def load_fixture(path: Path) -> tuple[Case, ...]:
    """Load strict JSONL query and relevance fixtures without logging text."""
    cases: list[Case] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {
            "case_id",
            "query",
            "candidates",
        }:
            raise ValueError(f"fixture record {line_number} has invalid fields")
        case_id = value["case_id"]
        query = value["query"]
        raw_candidates = value["candidates"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise ValueError(f"fixture record {line_number} has an invalid case_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"fixture record {line_number} has an invalid query")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise ValueError(
                f"fixture record {line_number} needs at least two candidates"
            )
        candidates: list[Candidate] = []
        candidate_ids: set[str] = set()
        for candidate in raw_candidates:
            if not isinstance(candidate, dict) or set(candidate) != {
                "candidate_id",
                "text",
                "relevant",
            }:
                raise ValueError(
                    f"fixture record {line_number} has an invalid candidate"
                )
            candidate_id = candidate["candidate_id"]
            text = candidate["text"]
            relevant = candidate["relevant"]
            if (
                not isinstance(candidate_id, str)
                or not candidate_id.strip()
                or candidate_id in candidate_ids
            ):
                raise ValueError(
                    f"fixture record {line_number} has an invalid candidate_id"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"fixture record {line_number} has invalid candidate text"
                )
            if not isinstance(relevant, bool):
                raise ValueError(
                    f"fixture record {line_number} has an invalid relevance label"
                )
            candidate_ids.add(candidate_id)
            candidates.append(Candidate(candidate_id, text, relevant))
        if not any(candidate.relevant for candidate in candidates):
            raise ValueError(f"fixture record {line_number} has no relevant candidate")
        case_ids.add(case_id)
        cases.append(Case(case_id, query, tuple(candidates)))
    if not cases:
        raise ValueError("fixture contains no cases")
    return tuple(cases)


def ranking_metrics(
    cases: Sequence[Case], scores: dict[str, float]
) -> dict[str, float]:
    """Compute deterministic binary ranking metrics for labelled cases."""
    reciprocal_ranks: list[float] = []
    recall_at_1: list[float] = []
    recall_at_5: list[float] = []
    ndcg_at_5: list[float] = []
    for case in cases:
        ranked = sorted(
            case.candidates,
            key=lambda candidate: (
                -scores[candidate.candidate_id],
                candidate.candidate_id,
            ),
        )
        relevant_ids = {
            candidate.candidate_id
            for candidate in case.candidates
            if candidate.relevant
        }
        relevant_ranks = [
            index for index, candidate in enumerate(ranked, 1) if candidate.relevant
        ]
        reciprocal_ranks.append(1 / relevant_ranks[0])
        recall_at_1.append(float(any(candidate.relevant for candidate in ranked[:1])))
        recall_at_5.append(float(any(candidate.relevant for candidate in ranked[:5])))
        dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks if rank <= 5)
        ideal_count = min(len(relevant_ids), 5)
        ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg_at_5.append(dcg / ideal_dcg)
    return {
        "recall_at_1": statistics.mean(recall_at_1),
        "recall_at_5": statistics.mean(recall_at_5),
        "mrr": statistics.mean(reciprocal_ranks),
        "ndcg_at_5": statistics.mean(ndcg_at_5),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entry(manifest: dict[str, Any], spec: ModelSpec) -> dict[str, Any]:
    entry = manifest.get(spec.name)
    if not isinstance(entry, dict) or entry.get("revision") != spec.revision:
        raise ValueError("artifact manifest does not match the pinned candidate")
    files = entry.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("artifact manifest has no file inventory")
    return entry


def _verify_artifacts(model_dir: Path, entry: dict[str, Any]) -> None:
    """Verify every locally acquired model file against the checked-in manifest."""
    expected = entry["files"]
    actual_files = {
        path.relative_to(model_dir).as_posix(): path
        for path in model_dir.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    if set(actual_files) != set(expected):
        raise ValueError("local model file inventory differs from the pinned manifest")
    for name, path in sorted(actual_files.items()):
        digest = _sha256(path)
        if digest != expected[name]:
            raise ValueError("local model file digest differs from the pinned manifest")


def _decode_model(
    spec: ModelSpec, model_dir: Path
) -> tuple[_Tokenizer, torch.nn.Module]:
    """Load the selected architecture exclusively from its verified directory."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    if spec.architecture == "encoder_sequence_classification":
        model = cast(
            torch.nn.Module,
            AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=spec.dtype,
            ),
        )
    else:
        if spec.name == "bge-reranker-v2-minicpm-layerwise":
            model = _load_minicpm_from_verified_local_code(model_dir, spec.dtype)
        else:
            model = cast(
                torch.nn.Module,
                AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=spec.dtype,
                ),
            )
    model.to("cpu")
    model.eval()
    return cast(_Tokenizer, tokenizer), model


def _load_minicpm_from_verified_local_code(
    model_dir: Path, dtype: torch.dtype
) -> torch.nn.Module:
    """Import MiniCPM's pinned code from its verified directory, without Hub lookup."""
    package_name = "run05_verified_minicpm"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    config_module = importlib.import_module(
        f"{package_name}.configuration_minicpm_reranker"
    )
    model_module = importlib.import_module(f"{package_name}.modeling_minicpm_reranker")
    config_type = config_module.LayerWiseMiniCPMConfig
    model_type = model_module.LayerWiseMiniCPMForCausalLM
    config = config_type.from_pretrained(model_dir, local_files_only=True)
    return cast(
        torch.nn.Module,
        model_type.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            torch_dtype=dtype,
        ),
    )


def _decoder_inputs(
    pairs: Sequence[tuple[str, str]], tokenizer: _Tokenizer, *, body_limit: int
) -> dict[str, torch.Tensor]:
    """Prepare BGE decoder prompts while preserving each complete query."""
    prompt = (
        "Given a query A and a passage B, determine whether the passage contains "
        "an answer to the query by providing a prediction of either 'Yes' or 'No'."
    )
    separator_ids = cast(
        list[int], tokenizer("\n", add_special_tokens=False)["input_ids"]
    )
    prompt_ids = cast(
        list[int], tokenizer(prompt, add_special_tokens=False)["input_ids"]
    )
    items: list[dict[str, list[int]]] = []
    for query, passage in pairs:
        query_ids = cast(
            list[int],
            tokenizer(f"A: {query}", add_special_tokens=False, truncation=False)[
                "input_ids"
            ],
        )
        passage_ids = cast(
            list[int],
            tokenizer(
                f"B: {passage}",
                add_special_tokens=False,
                truncation=True,
                max_length=body_limit,
            )["input_ids"],
        )
        if len(query_ids) > _QUERY_MAX_TOKENS:
            raise ValueError("fixture query exceeds the preserved query token limit")
        if tokenizer.bos_token_id is None:
            raise ValueError("the pinned decoder tokeniser has no BOS token")
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id, *query_ids],
            [*separator_ids, *passage_ids],
            truncation="only_second",
            max_length=body_limit,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        input_ids = [
            *cast(list[int], item["input_ids"]),
            *separator_ids,
            *prompt_ids,
        ]
        items.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
    tokenizer.padding_side = "left"
    return tokenizer.pad(items, padding=True, pad_to_multiple_of=8, return_tensors="pt")


def _score_batch(
    spec: ModelSpec,
    tokenizer: _Tokenizer,
    model: torch.nn.Module,
    pairs: Sequence[tuple[str, str]],
) -> tuple[float, ...]:
    if spec.architecture == "encoder_sequence_classification":
        inputs = tokenizer(
            [query for query, _ in pairs],
            [passage for _, passage in pairs],
            padding=True,
            truncation="only_second",
            max_length=_MAX_ENCODER_TOKENS,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**inputs, return_dict=True).logits.reshape(-1).float()
    else:
        inputs = _decoder_inputs(pairs, tokenizer, body_limit=_MAX_DECODER_BODY_TOKENS)
        with torch.inference_mode():
            if spec.architecture == "decoder_yes_logit":
                yes_ids = cast(
                    list[int], tokenizer("Yes", add_special_tokens=False)["input_ids"]
                )
                if len(yes_ids) != 1:
                    raise ValueError(
                        "the pinned Gemma tokeniser does not encode Yes as one token"
                    )
                logits = (
                    model(**inputs, return_dict=True).logits[:, -1, yes_ids[0]].float()
                )
            else:
                outputs = model(**inputs, return_dict=True, cutoff_layers=[28])
                logits = outputs.logits[0][:, -1].reshape(-1).float()
    values = tuple(float(value) for value in logits.tolist())
    if len(values) != len(pairs) or any(not math.isfinite(value) for value in values):
        raise ValueError("model returned an invalid score batch")
    return values


def _measurement_record(value: MemoryMeasurement) -> dict[str, Any]:
    """Project one RUN-03 measurement to path-free report fields."""
    return {
        "measurement_id": value.measurement_id,
        "component_role": value.component_role.value,
        "category": value.category,
        "memory_space": value.memory_space.value,
        "device_id": value.device_id,
        "status": value.status,
        "value_bytes": value.value_bytes,
        "adapter_name": value.adapter.adapter_name,
        "adapter_version": value.adapter.adapter_version,
        "calibration_revision": value.adapter.calibration_revision,
        "observed_at": value.observed_at.isoformat(),
    }


def _run_one(
    spec: ModelSpec,
    model_dir: Path,
    manifest_entry: dict[str, Any],
    cases: tuple[Case, ...],
) -> dict[str, Any]:
    model: torch.nn.Module | None = None
    tokenizer: _Tokenizer | None = None
    try:
        _verify_artifacts(model_dir, manifest_entry)
        started = time.perf_counter_ns()
        tokenizer, model = _decode_model(spec, model_dir)
        load_ms = (time.perf_counter_ns() - started) / 1_000_000
        memory = tuple(
            _measurement_record(item)
            for item in PyTorchMemoryCalibrationAdapter(
                model,
                component_id=f"reranker-{spec.name}",
                component_role=MemoryComponentRole.RERANKER,
            ).measure_memory()
        )
        flat = tuple(
            (case.case_id, candidate, case.query)
            for case in cases
            for candidate in case.candidates
        )

        def score_all() -> dict[str, float]:
            result: dict[str, float] = {}
            for start in range(0, len(flat), _INTERNAL_BATCH_SIZE):
                batch = flat[start : start + _INTERNAL_BATCH_SIZE]
                scores = _score_batch(
                    spec,
                    tokenizer,
                    model,
                    tuple((query, candidate.text) for _, candidate, query in batch),
                )
                result.update(
                    (candidate.candidate_id, score)
                    for (_, candidate, _), score in zip(batch, scores, strict=True)
                )
            return result

        for _ in range(_WARMUP_ITERATIONS):
            score_all()
        durations_ns: list[int] = []
        scores: dict[str, float] = {}
        measured_candidates = 0
        for _ in range(_MEASURED_ITERATIONS):
            batch_started = time.perf_counter_ns()
            scores = score_all()
            durations_ns.append(time.perf_counter_ns() - batch_started)
            measured_candidates += len(flat)
        metrics = ranking_metrics(cases, scores)
        elapsed_seconds = sum(durations_ns) / 1_000_000_000
        return {
            "status": "measured",
            "identity": {
                "repository": spec.repository,
                "revision": spec.revision,
                "repository_license": spec.repository_license,
                "upstream_license": spec.upstream_license,
                "architecture": spec.architecture,
                "inference_dtype": str(spec.dtype).removeprefix("torch."),
                "quantization": "none",
            },
            "preprocessing": {
                "profile_version": "1.0.0",
                "encoder_max_pair_tokens": _MAX_ENCODER_TOKENS,
                "decoder_body_limit": _MAX_DECODER_BODY_TOKENS,
                "query_truncation": "reject-over-256-tokens",
                "candidate_truncation": "tail-truncate-after-complete-query",
                "internal_batch_size": _INTERNAL_BATCH_SIZE,
            },
            "quality": metrics,
            "performance": {
                "load_duration_ms": load_ms,
                "warmup_iterations": _WARMUP_ITERATIONS,
                "measured_iterations": _MEASURED_ITERATIONS,
                "candidate_pairs_per_pass": len(flat),
                "candidate_pairs_measured": measured_candidates,
                "median_pass_duration_ms": statistics.median(durations_ns) / 1_000_000,
                "candidates_per_second": measured_candidates / elapsed_seconds,
            },
            "memory": list(memory),
            "memory_limitations": [
                "RUN-03 measurement counts unique inspectable tensor storage only.",
                "Process RSS and allocator reservation are not attributed per candidate.",
                "Allocator reservation, activation peaks, power, and thermal behaviour are not measured.",
            ],
        }
    except Exception as error:
        return {
            "status": "failed",
            "identity": {
                "repository": spec.repository,
                "revision": spec.revision,
                "repository_license": spec.repository_license,
                "upstream_license": spec.upstream_license,
                "architecture": spec.architecture,
                "inference_dtype": str(spec.dtype).removeprefix("torch."),
                "quantization": "none",
            },
            "failure_category": type(error).__name__,
        }
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()


def run_pilot(
    model_directories: dict[str, Path],
    *,
    fixture_path: Path = _FIXTURE,
    manifest_path: Path = _MANIFEST,
) -> dict[str, Any]:
    """Run the fixed local candidate comparison and return a private-data-free report."""
    cases = load_fixture(fixture_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        model_dir = model_directories.get(spec.name)
        if model_dir is None:
            continue
        entry = _manifest_entry(manifest, spec)
        candidates[spec.name] = _run_one(spec, model_dir, entry, cases)
    if not candidates:
        raise ValueError("at least one explicit local model directory is required")
    return {
        "report_schema_version": "1.0.0",
        "measured_at": utc_now().isoformat(),
        "purpose": "exploratory development pilot; not held-out evaluation",
        "fixture": {
            "sha256": _sha256(fixture_path),
            "case_count": len(cases),
            "candidate_pairs_per_pass": sum(len(case.candidates) for case in cases),
            "source": "synthetic fixed fixture; no private corpus passages",
        },
        "artifact_manifest_sha256": _sha256(manifest_path),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "device_type": "cpu",
        },
        "method": {
            "clock": "time.perf_counter_ns",
            "ranking_tie_break": "ascending candidate_id",
            "metrics": ["recall_at_1", "recall_at_5", "mrr", "ndcg_at_5"],
            "candidate_order": "fixture order before score ranking",
            "quantization_policy": "none; weights use the recorded inference dtype",
        },
        "candidates": candidates,
        "interpretation_limitations": [
            "Synthetic cases are a mechanics and directional-quality smoke fixture, not representative corpus evidence.",
            "Do not use this report alone to fix a production model revision.",
            "Upstream model terms require operator review for the intended use.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run local-only candidate measurements and write a path-free report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-path", type=Path)
    parser.add_argument("--gemma-path", type=Path)
    parser.add_argument("--minicpm-path", type=Path)
    parser.add_argument("--fixture", type=Path, default=_FIXTURE)
    parser.add_argument("--manifest", type=Path, default=_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    directories = {
        spec.name: path
        for spec, path in zip(
            MODEL_SPECS,
            (arguments.m3_path, arguments.gemma_path, arguments.minicpm_path),
            strict=True,
        )
        if path is not None
    }
    try:
        report = run_pilot(
            directories,
            fixture_path=arguments.fixture,
            manifest_path=arguments.manifest,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "reranker_candidate_pilot_failed",
                    "error_type": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    measured = sum(
        value["status"] == "measured" for value in report["candidates"].values()
    )
    print(
        json.dumps(
            {
                "event": "reranker_candidate_pilot_completed",
                "candidate_count": len(report["candidates"]),
                "measured_count": measured,
                "output_name": arguments.output.name,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
