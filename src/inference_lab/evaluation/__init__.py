"""Question sets, held-out cases, and quality/system metric evaluation."""

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.utils import HELD_OUT_SPLIT, SplitName
from inference_lab.evaluation.datasets import DatasetManifest
from inference_lab.evaluation.loader import LoadedDataset, load_dataset
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
from inference_lab.evaluation.schemas import (
    AbstentionExpectation,
    AnswerClaim,
    EvidenceReference,
    ExpectedAnswer,
    NumericExpectation,
)

__all__ = [
    "AbstentionExpectation",
    "AnswerClaim",
    "DatasetManifest",
    "EvaluationCase",
    "EvidenceReference",
    "ExpectedAnswer",
    "HELD_OUT_SPLIT",
    "LoadedDataset",
    "LeakageGroup",
    "load_dataset",
    "NumericExpectation",
    "atomic_write",
    "evaluation_json_schema",
    "serialise_cases",
    "serialise_manifest",
    "SplitName",
    "SplitPolicy",
    "build_leakage_groups",
    "generate_splits",
    "validate_no_leakage",
    "write_dataset",
    "write_json_schema",
]
