"""Versioned evaluation cases, datasets, and held-out split contracts."""

from inference_lab.evaluation.cases import EvaluationCase, QuestionType
from inference_lab.evaluation.coverage import (
    FINAL_BENCHMARK_POLICY,
    CoveragePolicy,
    CoverageReport,
    validate_benchmark_coverage,
    validate_final_benchmark,
)
from inference_lab.evaluation.datasets import DatasetManifest
from inference_lab.evaluation.loader import LoadedDataset, load_dataset
from inference_lab.evaluation.privacy import validate_local_path
from inference_lab.evaluation.schemas import (
    AbstentionExpectation,
    AnswerClaim,
    DateExpectation,
    EvaluationPolicy,
    EvidenceReference,
    ExpectedAnswer,
    NumericExpectation,
)
from inference_lab.evaluation.serialisation import (
    atomic_write,
    evaluation_json_schema,
    serialise_cases,
    serialise_manifest,
    write_dataset,
    write_json_schema,
)
from inference_lab.evaluation.splitting import (
    LeakageGroup,
    SplitPolicy,
    build_leakage_groups,
    generate_splits,
    validate_no_leakage,
)
from inference_lab.evaluation.utils import HELD_OUT_SPLIT, SplitName

__all__ = [
    "FINAL_BENCHMARK_POLICY",
    "HELD_OUT_SPLIT",
    "AbstentionExpectation",
    "AnswerClaim",
    "CoveragePolicy",
    "CoverageReport",
    "DatasetManifest",
    "DateExpectation",
    "EvaluationCase",
    "EvaluationPolicy",
    "EvidenceReference",
    "ExpectedAnswer",
    "LeakageGroup",
    "LoadedDataset",
    "NumericExpectation",
    "QuestionType",
    "SplitName",
    "SplitPolicy",
    "atomic_write",
    "build_leakage_groups",
    "evaluation_json_schema",
    "generate_splits",
    "load_dataset",
    "serialise_cases",
    "serialise_manifest",
    "validate_benchmark_coverage",
    "validate_final_benchmark",
    "validate_local_path",
    "validate_no_leakage",
    "write_dataset",
    "write_json_schema",
]
