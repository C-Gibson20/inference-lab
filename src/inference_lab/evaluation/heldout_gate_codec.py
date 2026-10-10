"""Strict decoding of the approved held-out retrieval gate policy artefact."""

from dataclasses import dataclass
from typing import Final

from inference_lab._json import (
    JsonObject,
    JsonValue,
    load_json_object,
    require_exact_json_fields,
    require_json_integer,
    require_json_object,
    require_json_string,
)
from inference_lab._validation import is_canonical_semantic_version
from inference_lab.evaluation._validation import is_sha256_checksum
from inference_lab.evaluation.heldout_gate import (
    HELD_OUT_GATE_POLICY_VERSION,
    HELD_OUT_UNCERTAINTY_METHOD,
    HeldOutChannel,
    HeldOutGatePolicy,
)
from inference_lab.evaluation.retrieval_contracts import (
    MetricFraction,
    RetrievalEvaluationContractError,
)

HELD_OUT_GATE_PRIMARY_METRIC: Final[str] = "document_recall"
HELD_OUT_GATE_PRIMARY_CUTOFF: Final[int] = 10
HELD_OUT_GATE_PRIMARY_DENOMINATOR: Final[str] = (
    "all relevant document targets, including failed and empty outcomes"
)
HELD_OUT_GATE_SPLIT: Final[str] = "held_out"

_ROOT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "dataset",
        "corpus",
        "split",
        "primary_gate",
        "maximum_failure_rate",
        "maximum_degraded_rate",
        "category_coverage",
        "uncertainty",
        "rationale",
    }
)
_DATASET_FIELDS: Final = frozenset(
    {"id", "version", "schema_version", "case_file_checksum"}
)
_CORPUS_FIELDS: Final = frozenset({"id", "version"})
_PRIMARY_GATE_FIELDS: Final = frozenset(
    {"channel", "metric", "cutoff", "minimum", "denominator"}
)
_COVERAGE_FIELDS: Final = frozenset({"minimum_cases", "minimum_leakage_groups"})
_UNCERTAINTY_FIELDS: Final = frozenset(
    {"method", "confidence_level_percent", "replicates"}
)
_FRACTION_FIELDS: Final = frozenset({"numerator", "denominator"})


@dataclass(frozen=True, slots=True)
class HeldOutGateBinding:
    """The exact frozen inputs an approved gate policy was approved against."""

    dataset_id: str
    dataset_version: str
    dataset_schema_version: str
    case_file_checksum: str
    corpus_id: str
    corpus_version: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Reject a binding that cannot identify one frozen benchmark."""
        for name in (
            "dataset_id",
            "dataset_version",
            "dataset_schema_version",
            "case_file_checksum",
            "corpus_id",
            "corpus_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string.")
            if not value:
                raise RetrievalEvaluationContractError(f"{name} must not be empty.")
        for name in ("dataset_version", "dataset_schema_version"):
            if not is_canonical_semantic_version(getattr(self, name)):
                raise RetrievalEvaluationContractError(
                    f"{name} must be a canonical semantic version."
                )
        if not is_sha256_checksum(self.case_file_checksum):
            raise RetrievalEvaluationContractError(
                "case_file_checksum must be a prefixed SHA-256 checksum."
            )


@dataclass(frozen=True, slots=True)
class ApprovedHeldOutGate:
    """An approved gate policy together with the inputs it is bound to."""

    binding: HeldOutGateBinding
    policy: HeldOutGatePolicy

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Reject a pairing that is not the approved policy and its binding."""
        if not isinstance(self.binding, HeldOutGateBinding):
            raise TypeError("binding must be a HeldOutGateBinding.")
        if not isinstance(self.policy, HeldOutGatePolicy):
            raise TypeError("policy must be a HeldOutGatePolicy.")


def _require_fraction(value: JsonValue, field_name: str) -> MetricFraction:
    """Decode one exact numerator and denominator pair."""
    payload = require_json_object(value)
    require_exact_json_fields(payload, _FRACTION_FIELDS)
    numerator = require_json_integer(payload["numerator"])
    denominator = require_json_integer(payload["denominator"])
    try:
        return MetricFraction(numerator, denominator)
    except ValueError as error:
        raise RetrievalEvaluationContractError(
            f"{field_name} must be a valid metric fraction."
        ) from error


def _require_binding(payload: JsonObject) -> HeldOutGateBinding:
    """Decode the dataset and corpus identities the policy was approved for."""
    dataset = require_json_object(payload["dataset"])
    require_exact_json_fields(dataset, _DATASET_FIELDS)
    corpus = require_json_object(payload["corpus"])
    require_exact_json_fields(corpus, _CORPUS_FIELDS)
    return HeldOutGateBinding(
        dataset_id=require_json_string(dataset["id"]),
        dataset_version=require_json_string(dataset["version"]),
        dataset_schema_version=require_json_string(dataset["schema_version"]),
        case_file_checksum=require_json_string(dataset["case_file_checksum"]),
        corpus_id=require_json_string(corpus["id"]),
        corpus_version=require_json_string(corpus["version"]),
    )


def _require_primary_minimum(payload: JsonObject) -> MetricFraction:
    """Decode the primary threshold, rejecting any other metric semantics."""
    primary = require_json_object(payload["primary_gate"])
    require_exact_json_fields(primary, _PRIMARY_GATE_FIELDS)
    if (
        require_json_string(primary["channel"]) != HeldOutChannel.FUSED_PRE_RERANK
        or require_json_string(primary["metric"]) != HELD_OUT_GATE_PRIMARY_METRIC
        or require_json_integer(primary["cutoff"]) != HELD_OUT_GATE_PRIMARY_CUTOFF
        or require_json_string(primary["denominator"])
        != HELD_OUT_GATE_PRIMARY_DENOMINATOR
    ):
        raise RetrievalEvaluationContractError(
            "primary_gate must state the supported metric semantics."
        )
    return _require_fraction(primary["minimum"], "primary_gate.minimum")


def deserialise_approved_held_out_gate(data: bytes) -> ApprovedHeldOutGate:
    """Decode the approved gate artefact without trusting any of its fields."""
    payload = load_json_object(data)
    require_exact_json_fields(payload, _ROOT_FIELDS)
    if require_json_string(payload["schema_version"]) != HELD_OUT_GATE_POLICY_VERSION:
        raise RetrievalEvaluationContractError(
            "schema_version names an unsupported gate policy version."
        )
    if require_json_string(payload["split"]) != HELD_OUT_GATE_SPLIT:
        raise RetrievalEvaluationContractError("split must be the held-out split.")
    coverage = require_json_object(payload["category_coverage"])
    require_exact_json_fields(coverage, _COVERAGE_FIELDS)
    uncertainty = require_json_object(payload["uncertainty"])
    require_exact_json_fields(uncertainty, _UNCERTAINTY_FIELDS)
    if require_json_string(uncertainty["method"]) != HELD_OUT_UNCERTAINTY_METHOD:
        raise RetrievalEvaluationContractError(
            "uncertainty.method names an unsupported method."
        )
    return ApprovedHeldOutGate(
        binding=_require_binding(payload),
        policy=HeldOutGatePolicy(
            minimum_document_recall_at_10=_require_primary_minimum(payload),
            maximum_failure_rate=_require_fraction(
                payload["maximum_failure_rate"], "maximum_failure_rate"
            ),
            maximum_degraded_rate=_require_fraction(
                payload["maximum_degraded_rate"], "maximum_degraded_rate"
            ),
            minimum_category_cases=require_json_integer(coverage["minimum_cases"]),
            minimum_category_leakage_groups=require_json_integer(
                coverage["minimum_leakage_groups"]
            ),
            confidence_level_percent=require_json_integer(
                uncertainty["confidence_level_percent"]
            ),
            bootstrap_replicates=require_json_integer(uncertainty["replicates"]),
            rationale=require_json_string(payload["rationale"]),
        ),
    )
