"""Versioned evaluation cases, datasets, and corpus-audit contracts."""

from inference_lab.evaluation.audit_capabilities import (
    CURRENT_PARSER_CAPABILITY_PROFILE,
    PARSER_CAPABILITY_PROFILES,
    ParserCapabilityProfile,
    parser_capability_profile,
    validate_parsed_document_capabilities,
)
from inference_lab.evaluation.audit_contracts import (
    AUDIT_CONTRACT_VERSION,
    AcquisitionStatus,
    AuditAcquisitionObservation,
    AuditCoverage,
    AuditDistribution,
    AuditMissingReason,
    AuditParseObservation,
    AuditRate,
    AuditRelationshipInventory,
    AuditSourceFailure,
    AuditSourceInventory,
    AuditStructureKind,
    CorpusAuditPolicy,
    CorpusAuditRequest,
    CorpusAuditScope,
)
from inference_lab.evaluation.audit_errors import (
    AuditContractError,
    AuditErrorCategory,
)
from inference_lab.evaluation.cases import EvaluationCase, QuestionType
from inference_lab.evaluation.contracts import HELD_OUT_SPLIT, SplitName
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

__all__ = [
    "AUDIT_CONTRACT_VERSION",
    "CURRENT_PARSER_CAPABILITY_PROFILE",
    "FINAL_BENCHMARK_POLICY",
    "HELD_OUT_SPLIT",
    "PARSER_CAPABILITY_PROFILES",
    "AbstentionExpectation",
    "AcquisitionStatus",
    "AnswerClaim",
    "AuditAcquisitionObservation",
    "AuditContractError",
    "AuditCoverage",
    "AuditDistribution",
    "AuditErrorCategory",
    "AuditMissingReason",
    "AuditParseObservation",
    "AuditRate",
    "AuditRelationshipInventory",
    "AuditSourceFailure",
    "AuditSourceInventory",
    "AuditStructureKind",
    "CorpusAuditPolicy",
    "CorpusAuditRequest",
    "CorpusAuditScope",
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
    "ParserCapabilityProfile",
    "QuestionType",
    "SplitName",
    "SplitPolicy",
    "atomic_write",
    "build_leakage_groups",
    "evaluation_json_schema",
    "generate_splits",
    "load_dataset",
    "parser_capability_profile",
    "serialise_cases",
    "serialise_manifest",
    "validate_benchmark_coverage",
    "validate_final_benchmark",
    "validate_local_path",
    "validate_no_leakage",
    "validate_parsed_document_capabilities",
    "write_dataset",
    "write_json_schema",
]
