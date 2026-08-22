"""Question sets, held-out cases, and quality/system metric evaluation."""

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.coverage import (
    CoverageReport,
    validate_final_benchmark,
    validate_benchmark_coverage,
)
from inference_lab.evaluation.answer_matching import (
    AnswerObservation,
    MatchResult,
    claim_matches,
    comparison_matches,
    date_matches,
    match_expected_answer,
    numeric_matches,
)
from inference_lab.evaluation.utils import HELD_OUT_SPLIT, SplitName
from inference_lab.evaluation.datasets import DatasetManifest
from inference_lab.evaluation.loader import LoadedDataset, load_dataset
from inference_lab.evaluation.runs import (
    EvaluationPurpose,
    EvaluationRun,
    EvaluationRunRequest,
    compute_run_id,
    configuration_digest,
    create_evaluation_run,
    validate_run_inputs,
    validate_run_purpose,
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
from inference_lab.evaluation.schemas import (
    AbstentionExpectation,
    AnswerClaim,
    DateExpectation,
    EvaluationPolicy,
    EvidenceReference,
    ExpectedAnswer,
    NumericExpectation,
)

__all__ = [
    "AbstentionExpectation",
    "AnswerClaim",
    "AnswerObservation",
    "DateExpectation",
    "EvaluationPolicy",
    "EvaluationPurpose",
    "EvaluationRun",
    "EvaluationRunRequest",
    "CoverageReport",
    "DatasetManifest",
    "EvaluationCase",
    "EvidenceReference",
    "ExpectedAnswer",
    "HELD_OUT_SPLIT",
    "LoadedDataset",
    "LeakageGroup",
    "load_dataset",
    "NumericExpectation",
    "MatchResult",
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
    "claim_matches",
    "compute_run_id",
    "comparison_matches",
    "configuration_digest",
    "create_evaluation_run",
    "date_matches",
    "match_expected_answer",
    "numeric_matches",
    "validate_run_inputs",
    "validate_run_purpose",
    "validate_benchmark_coverage",
    "validate_final_benchmark",
]
