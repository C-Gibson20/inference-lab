"""Predeclared gate policy and coverage checks for held-out retrieval runs."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab._validation import is_strict_integer
from inference_lab.evaluation.cases import QUESTION_TYPES, QuestionType
from inference_lab.evaluation.retrieval_contracts import (
    MetricFraction,
    RetrievalEvaluationContractError,
    RetrievalEvaluationRunIdentity,
    RetrievalSystemIdentity,
)
from inference_lab.retrieval import FusionPolicy
from inference_lab.retrieval.reranking import RerankingIdentity

HELD_OUT_RUN_CONTRACT_VERSION: Final[str] = "1.0.0"
HELD_OUT_GATE_POLICY_VERSION: Final[str] = "1.0.0"
HELD_OUT_UNCERTAINTY_METHOD: Final[str] = "leakage-cluster-bootstrap"


class HeldOutGateReason(StrEnum):
    """Closed set of reasons a held-out gate records with its verdict."""

    MISSING_QUESTION_CATEGORIES = "missing_question_categories"
    UNDERPOWERED_QUESTION_CATEGORIES = "underpowered_question_categories"
    INCOMPLETE_CATEGORY_MEASUREMENT = "incomplete_category_measurement"
    MISSING_ANSWERABILITY_STATUS = "missing_answerability_status"
    INCOMPLETE_PRIMARY_METRIC = "incomplete_primary_metric"
    FAILURE_RATE_EXCEEDS_POLICY = "failure_rate_exceeds_policy"
    DEGRADED_RATE_EXCEEDS_POLICY = "degraded_rate_exceeds_policy"
    PRIMARY_METRIC_BELOW_MINIMUM = "primary_metric_below_minimum"
    PRIMARY_METRIC_UNCERTAIN_AT_THRESHOLD = "primary_metric_uncertain_at_threshold"
    INCOMPLETE_FINAL_EVIDENCE_MEASUREMENT = "incomplete_final_evidence_measurement"


class HeldOutGateOutcome(StrEnum):
    """Conservative outcome of a frozen held-out retrieval gate."""

    PASS = "pass"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class HeldOutChannel(StrEnum):
    """Retrieval paths compared by the hybrid retrieval baseline."""

    LEXICAL_ONLY = "lexical_only"
    DENSE_ONLY = "dense_only"
    METADATA = "metadata"
    FUSED_PRE_RERANK = "fused_pre_rerank"
    FUSED_RERANKED = "fused_reranked"


@dataclass(frozen=True, slots=True)
class HeldOutChannelIdentity:
    """Path-free identity for one frozen channel comparison path."""

    channel: HeldOutChannel
    system_identity: RetrievalSystemIdentity
    fusion_policy: FusionPolicy | None = None
    reranking_identity: RerankingIdentity | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Reject incomplete or unsupported channel identities."""
        if not isinstance(self.channel, HeldOutChannel):
            raise TypeError("channel must be a HeldOutChannel.")
        if not isinstance(self.system_identity, RetrievalSystemIdentity):
            raise TypeError("system_identity must be a RetrievalSystemIdentity.")
        fused = self.channel in {
            HeldOutChannel.FUSED_PRE_RERANK,
            HeldOutChannel.FUSED_RERANKED,
        }
        if fused != isinstance(self.fusion_policy, FusionPolicy):
            raise RetrievalEvaluationContractError(
                "fusion paths must identify their FusionPolicy."
            )
        if self.fusion_policy is not None and not isinstance(
            self.fusion_policy, FusionPolicy
        ):
            raise TypeError("fusion_policy must be a FusionPolicy or None.")
        reranked = self.channel is HeldOutChannel.FUSED_RERANKED
        if reranked != isinstance(self.reranking_identity, RerankingIdentity):
            raise RetrievalEvaluationContractError(
                "the reranked path must identify its reranking configuration."
            )
        if self.reranking_identity is not None and not isinstance(
            self.reranking_identity, RerankingIdentity
        ):
            raise TypeError("reranking_identity must be a RerankingIdentity or None.")

    def as_dict(self) -> dict[str, object]:
        """Return the channel label and its full path-free identity."""
        return {
            "channel": self.channel.value,
            "system_identity": self.system_identity.as_dict(),
            "fusion_policy": (
                {
                    "algorithm": self.fusion_policy.algorithm,
                    "version": self.fusion_policy.version,
                    "rank_constant": self.fusion_policy.rank_constant,
                    "lexical_weight": self.fusion_policy.lexical_weight,
                    "dense_weight": self.fusion_policy.dense_weight,
                    "identity": self.fusion_policy.identity,
                }
                if self.fusion_policy is not None
                else None
            ),
            "reranking_identity": (
                {
                    "backend_name": self.reranking_identity.backend_name,
                    "model_name": self.reranking_identity.model_name,
                    "model_revision": self.reranking_identity.model_revision,
                    "preprocessing_profile_version": (
                        self.reranking_identity.preprocessing_profile_version
                    ),
                    "evidence_count": self.reranking_identity.evidence_count,
                    "candidate_limit": self.reranking_identity.candidate_limit,
                    "deduplication_version": (
                        self.reranking_identity.deduplication_version
                    ),
                    "policy_version": self.reranking_identity.policy_version,
                    "digest": self.reranking_identity.digest,
                }
                if self.reranking_identity is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class HeldOutRunContract:
    """Frozen single-corpus run identity and gate policy for the held-out baseline."""

    evaluation_identity: RetrievalEvaluationRunIdentity
    gate_policy: "HeldOutGatePolicy"
    channel_identities: tuple[HeldOutChannelIdentity, ...]
    candidate_recall_cutoff: int = 10
    final_evidence_count: int = 8
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the run is held out and pins each compared retrieval path."""
        if not isinstance(self.evaluation_identity, RetrievalEvaluationRunIdentity):
            raise TypeError(
                "evaluation_identity must be a RetrievalEvaluationRunIdentity."
            )
        if self.evaluation_identity.selected_split != "held_out":
            raise RetrievalEvaluationContractError(
                "the held-out run identity must select the held_out split."
            )
        if not isinstance(self.gate_policy, HeldOutGatePolicy):
            raise TypeError("gate_policy must be a HeldOutGatePolicy.")
        if not isinstance(self.channel_identities, tuple):
            raise TypeError("channel_identities must be a tuple.")
        if any(
            not isinstance(identity, HeldOutChannelIdentity)
            for identity in self.channel_identities
        ):
            raise TypeError(
                "channel_identities must contain HeldOutChannelIdentity values."
            )
        channels = tuple(identity.channel for identity in self.channel_identities)
        required = tuple(HeldOutChannel)
        if len(channels) != len(set(channels)) or set(channels) != set(required):
            raise RetrievalEvaluationContractError(
                "run contract must identify every compared retrieval path once."
            )
        identity_by_channel = {item.channel: item for item in self.channel_identities}
        pre_rerank_fusion = identity_by_channel[
            HeldOutChannel.FUSED_PRE_RERANK
        ].fusion_policy
        reranked_identity = identity_by_channel[HeldOutChannel.FUSED_RERANKED]
        if reranked_identity.fusion_policy != pre_rerank_fusion:
            raise RetrievalEvaluationContractError(
                "pre-rerank and reranked paths must share one fusion policy."
            )
        if not is_strict_integer(self.candidate_recall_cutoff):
            raise TypeError("candidate_recall_cutoff must be an integer.")
        if self.candidate_recall_cutoff != 10:
            raise RetrievalEvaluationContractError(
                "candidate_recall_cutoff is fixed at 10."
            )
        if not is_strict_integer(self.final_evidence_count):
            raise TypeError("final_evidence_count must be an integer.")
        if not 5 <= self.final_evidence_count <= 10:
            raise RetrievalEvaluationContractError(
                "final_evidence_count must be between 5 and 10."
            )
        assert reranked_identity.reranking_identity is not None
        if (
            reranked_identity.reranking_identity.evidence_count
            != self.final_evidence_count
        ):
            raise RetrievalEvaluationContractError(
                "the reranked evidence count must match final_evidence_count."
            )
        calculated = self._calculate_fingerprint()
        if self.fingerprint is None:
            object.__setattr__(self, "fingerprint", calculated)
        else:
            if not isinstance(self.fingerprint, str):
                raise TypeError("fingerprint must be a string or None.")
            if self.fingerprint != calculated:
                raise RetrievalEvaluationContractError(
                    "fingerprint does not match the frozen run contract."
                )

    def _semantic_payload(self) -> JsonObject:
        """Return every run input that affects held-out interpretation."""
        return cast(
            JsonObject,
            {
                "contract_version": HELD_OUT_RUN_CONTRACT_VERSION,
                "evaluation_identity": self.evaluation_identity.as_dict(),
                "gate_policy": self.gate_policy.as_dict(),
                "channel_identities": [
                    identity.as_dict()
                    for identity in sorted(
                        self.channel_identities, key=lambda item: item.channel.value
                    )
                ],
                "candidate_recall_cutoff": self.candidate_recall_cutoff,
                "final_evidence_count": self.final_evidence_count,
            },
        )

    def _calculate_fingerprint(self) -> str:
        """Hash the canonical complete run contract."""
        digest = hashlib.sha256(
            canonical_json_bytes(self._semantic_payload())
        ).hexdigest()
        return f"sha256:{digest}"

    def as_dict(self) -> dict[str, object]:
        """Return the complete frozen run identity."""
        payload = dict(self._semantic_payload())
        payload["fingerprint"] = self.fingerprint
        return cast(dict[str, object], payload)


@dataclass(frozen=True, slots=True)
class HeldOutGatePolicy:
    """Immutable thresholds fixed before opening held-out results.

    Deliberately has no defaults: a run must state its threshold and rationale.
    """

    minimum_document_recall: MetricFraction
    document_recall_cutoff: int
    maximum_failure_rate: MetricFraction
    maximum_degraded_rate: MetricFraction
    minimum_category_cases: int
    minimum_category_leakage_groups: int
    confidence_level_percent: int
    bootstrap_replicates: int
    rationale: str
    version: str = HELD_OUT_GATE_POLICY_VERSION
    uncertainty_method: str = HELD_OUT_UNCERTAINTY_METHOD

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Reject incomplete, unsupported, or mutable policy inputs."""
        for name in (
            "minimum_document_recall",
            "maximum_failure_rate",
            "maximum_degraded_rate",
        ):
            if not isinstance(getattr(self, name), MetricFraction):
                raise TypeError(f"{name} must be a MetricFraction.")
        for name in (
            "document_recall_cutoff",
            "minimum_category_cases",
            "minimum_category_leakage_groups",
            "bootstrap_replicates",
        ):
            value = getattr(self, name)
            if not is_strict_integer(value):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise RetrievalEvaluationContractError(f"{name} must be positive.")
        if not is_strict_integer(self.confidence_level_percent):
            raise TypeError("confidence_level_percent must be an integer.")
        if not 50 <= self.confidence_level_percent < 100:
            raise RetrievalEvaluationContractError(
                "confidence_level_percent must be from 50 through 99."
            )
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string.")
        if not self.rationale.strip():
            raise RetrievalEvaluationContractError("rationale must not be blank.")
        if not isinstance(self.version, str):
            raise TypeError("version must be a string.")
        if self.version != HELD_OUT_GATE_POLICY_VERSION:
            raise RetrievalEvaluationContractError(
                "unsupported held-out gate policy version."
            )
        if not isinstance(self.uncertainty_method, str):
            raise TypeError("uncertainty_method must be a string.")
        if self.uncertainty_method != HELD_OUT_UNCERTAINTY_METHOD:
            raise RetrievalEvaluationContractError(
                "unsupported held-out uncertainty method."
            )

    def as_dict(self) -> dict[str, object]:
        """Return the complete policy identity for inclusion in a run record."""
        return {
            "version": self.version,
            "document_recall_cutoff": self.document_recall_cutoff,
            "minimum_document_recall": (self.minimum_document_recall.as_dict()),
            "maximum_failure_rate": self.maximum_failure_rate.as_dict(),
            "maximum_degraded_rate": self.maximum_degraded_rate.as_dict(),
            "minimum_category_cases": self.minimum_category_cases,
            "minimum_category_leakage_groups": self.minimum_category_leakage_groups,
            "confidence_level_percent": self.confidence_level_percent,
            "uncertainty_method": self.uncertainty_method,
            "bootstrap_replicates": self.bootstrap_replicates,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class HeldOutCategoryCoverage:
    """Aggregate coverage and uncertainty for one question category."""

    question_type: QuestionType
    case_count: int
    answerable_case_count: int
    expected_abstention_case_count: int
    leakage_group_count: int
    failed_case_count: int
    degraded_case_count: int
    document_recall_cutoff: int
    document_recall: MetricFraction | None
    recall_interval_lower: MetricFraction | None
    recall_interval_upper: MetricFraction | None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate population reconciliation and optional interval bounds."""
        if self.question_type not in QUESTION_TYPES:
            raise RetrievalEvaluationContractError("unsupported question category.")
        for name in (
            "case_count",
            "answerable_case_count",
            "expected_abstention_case_count",
            "leakage_group_count",
            "failed_case_count",
            "degraded_case_count",
        ):
            value = getattr(self, name)
            if not is_strict_integer(value):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise RetrievalEvaluationContractError(f"{name} must not be negative.")
        if self.case_count == 0:
            raise RetrievalEvaluationContractError("category coverage cannot be empty.")
        if self.answerable_case_count + self.expected_abstention_case_count != (
            self.case_count
        ):
            raise RetrievalEvaluationContractError(
                "answerability counts must reconcile to the category population."
            )
        if self.leakage_group_count == 0 or self.leakage_group_count > self.case_count:
            raise RetrievalEvaluationContractError(
                "leakage_group_count must be within the category population."
            )
        if self.failed_case_count > self.case_count:
            raise RetrievalEvaluationContractError(
                "failed_case_count must not exceed the category population."
            )
        if self.degraded_case_count > self.case_count:
            raise RetrievalEvaluationContractError(
                "degraded_case_count must not exceed the category population."
            )
        if not is_strict_integer(self.document_recall_cutoff):
            raise TypeError("document_recall_cutoff must be an integer.")
        if self.document_recall_cutoff <= 0:
            raise RetrievalEvaluationContractError(
                "document_recall_cutoff must be positive."
            )
        for name in (
            "document_recall",
            "recall_interval_lower",
            "recall_interval_upper",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, MetricFraction):
                raise TypeError(f"{name} must be a MetricFraction or None.")
        lower = self.recall_interval_lower
        upper = self.recall_interval_upper
        recall = self.document_recall
        if (lower is None) != (upper is None):
            raise RetrievalEvaluationContractError(
                "recall interval bounds must be supplied together."
            )
        if lower is not None and upper is not None:
            if recall is None:
                raise RetrievalEvaluationContractError(
                    "recall interval requires an observed recall value."
                )
            if _fraction_less(upper, lower):
                raise RetrievalEvaluationContractError(
                    "recall interval lower bound must not exceed its upper bound."
                )

    def as_dict(self) -> dict[str, object]:
        """Return only category aggregates and bounded uncertainty values."""
        return {
            "question_type": self.question_type,
            "case_count": self.case_count,
            "answerable_case_count": self.answerable_case_count,
            "expected_abstention_case_count": self.expected_abstention_case_count,
            "leakage_group_count": self.leakage_group_count,
            "failed_case_count": self.failed_case_count,
            "degraded_case_count": self.degraded_case_count,
            "document_recall_cutoff": self.document_recall_cutoff,
            "document_recall": (
                self.document_recall.as_dict()
                if self.document_recall is not None
                else None
            ),
            "recall_interval_lower": (
                self.recall_interval_lower.as_dict()
                if self.recall_interval_lower is not None
                else None
            ),
            "recall_interval_upper": (
                self.recall_interval_upper.as_dict()
                if self.recall_interval_upper is not None
                else None
            ),
        }


def _fraction_less(left: MetricFraction, right: MetricFraction) -> bool:
    """Compare exact fractions without float conversion."""
    return left.numerator * right.denominator < right.numerator * left.denominator


@dataclass(frozen=True, slots=True)
class HeldOutGateObservations:
    """The measured held-out outcome that a gate policy is applied to."""

    categories: tuple[HeldOutCategoryCoverage, ...]
    overall_document_recall: MetricFraction | None
    overall_recall_interval_lower: MetricFraction | None
    overall_recall_interval_upper: MetricFraction | None
    case_count: int
    failed_case_count: int
    degraded_case_count: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Reject observations that cannot describe one complete held-out run."""
        self._validate_categories()
        self._validate_counts()
        self._validate_metrics()

    def _validate_categories(self) -> None:
        """Require unique categories whose populations fit the overall run."""
        if not isinstance(self.categories, tuple) or any(
            not isinstance(category, HeldOutCategoryCoverage)
            for category in self.categories
        ):
            raise TypeError("categories must be a tuple of HeldOutCategoryCoverage.")
        if len({category.question_type for category in self.categories}) != len(
            self.categories
        ):
            raise RetrievalEvaluationContractError(
                "question categories must be unique."
            )
        if not self.categories:
            return
        totals = (
            sum(category.case_count for category in self.categories),
            sum(category.failed_case_count for category in self.categories),
            sum(category.degraded_case_count for category in self.categories),
        )
        overall = (self.case_count, self.failed_case_count, self.degraded_case_count)
        if any(total > limit for total, limit in zip(totals, overall, strict=True)):
            raise RetrievalEvaluationContractError(
                "category populations must not exceed overall counts."
            )
        if len(self.categories) == len(QUESTION_TYPES) and totals != overall:
            raise RetrievalEvaluationContractError(
                "complete category totals must reconcile to overall counts."
            )

    def _validate_counts(self) -> None:
        """Require non-negative counts that reconcile with the case total."""
        for name in ("case_count", "failed_case_count", "degraded_case_count"):
            value = getattr(self, name)
            if not is_strict_integer(value):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise RetrievalEvaluationContractError(f"{name} must not be negative.")
        if (
            self.case_count == 0
            or self.failed_case_count > self.case_count
            or self.degraded_case_count > self.case_count
        ):
            raise RetrievalEvaluationContractError("overall counts do not reconcile.")

    def _validate_metrics(self) -> None:
        """Require a consistent primary metric and uncertainty interval."""
        for name in (
            "overall_document_recall",
            "overall_recall_interval_lower",
            "overall_recall_interval_upper",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, MetricFraction):
                raise TypeError(f"{name} must be a MetricFraction or None.")
        lower = self.overall_recall_interval_lower
        upper = self.overall_recall_interval_upper
        if (lower is None) != (upper is None):
            raise RetrievalEvaluationContractError(
                "overall recall interval bounds must be supplied together."
            )
        if lower is not None and upper is not None and _fraction_less(upper, lower):
            raise RetrievalEvaluationContractError(
                "overall recall interval lower bound must not exceed its upper bound."
            )

    @property
    def primary_metric_incomplete(self) -> bool:
        """Return whether the primary metric or its interval is missing."""
        return (
            self.overall_document_recall is None
            or self.overall_recall_interval_lower is None
            or self.overall_recall_interval_upper is None
        )


def evaluate_held_out_gate(
    *,
    policy: HeldOutGatePolicy,
    observations: HeldOutGateObservations,
) -> tuple[HeldOutGateOutcome, tuple[HeldOutGateReason, ...]]:
    """Apply a fixed gate policy while preserving uncertainty and missing coverage."""
    if not isinstance(policy, HeldOutGatePolicy):
        raise TypeError("policy must be a HeldOutGatePolicy.")
    if not isinstance(observations, HeldOutGateObservations):
        raise TypeError("observations must be a HeldOutGateObservations.")
    reasons = [
        *_coverage_reasons(policy, observations),
        *_rate_reasons(policy, observations),
    ]
    return _gate_verdict(policy, observations, reasons), tuple(reasons)


_INCONCLUSIVE_REASONS: Final = (
    HeldOutGateReason.MISSING_QUESTION_CATEGORIES,
    HeldOutGateReason.UNDERPOWERED_QUESTION_CATEGORIES,
    HeldOutGateReason.INCOMPLETE_CATEGORY_MEASUREMENT,
    HeldOutGateReason.INCOMPLETE_PRIMARY_METRIC,
    HeldOutGateReason.MISSING_ANSWERABILITY_STATUS,
)


def _coverage_reasons(
    policy: HeldOutGatePolicy,
    observations: HeldOutGateObservations,
) -> list[HeldOutGateReason]:
    """Report every way this run is too thin to conclude anything."""
    categories = observations.categories
    by_type = {coverage.question_type: coverage for coverage in categories}
    missing_categories = tuple(
        question_type
        for question_type in QUESTION_TYPES
        if question_type not in by_type
    )
    reasons: list[HeldOutGateReason] = []
    if missing_categories:
        reasons.append(HeldOutGateReason.MISSING_QUESTION_CATEGORIES)
    if any(
        coverage.case_count < policy.minimum_category_cases
        or coverage.leakage_group_count < policy.minimum_category_leakage_groups
        for coverage in categories
    ):
        reasons.append(HeldOutGateReason.UNDERPOWERED_QUESTION_CATEGORIES)
    if any(
        coverage.document_recall is None
        or coverage.recall_interval_lower is None
        or coverage.recall_interval_upper is None
        for coverage in categories
    ):
        reasons.append(HeldOutGateReason.INCOMPLETE_CATEGORY_MEASUREMENT)
    if categories and not missing_categories:
        answerable_count = sum(
            coverage.answerable_case_count for coverage in categories
        )
        expected_abstention_count = sum(
            coverage.expected_abstention_case_count for coverage in categories
        )
        if answerable_count == 0 or expected_abstention_count == 0:
            reasons.append(HeldOutGateReason.MISSING_ANSWERABILITY_STATUS)
    if observations.primary_metric_incomplete:
        reasons.append(HeldOutGateReason.INCOMPLETE_PRIMARY_METRIC)
    return reasons


def _rate_reasons(
    policy: HeldOutGatePolicy,
    observations: HeldOutGateObservations,
) -> list[HeldOutGateReason]:
    """Report failure and degradation rates that breach the approved policy."""
    reasons: list[HeldOutGateReason] = []
    if _fraction_less(
        policy.maximum_failure_rate,
        MetricFraction(observations.failed_case_count, observations.case_count),
    ):
        reasons.append(HeldOutGateReason.FAILURE_RATE_EXCEEDS_POLICY)
    if _fraction_less(
        policy.maximum_degraded_rate,
        MetricFraction(observations.degraded_case_count, observations.case_count),
    ):
        reasons.append(HeldOutGateReason.DEGRADED_RATE_EXCEEDS_POLICY)
    return reasons


def _gate_verdict(
    policy: HeldOutGatePolicy,
    observations: HeldOutGateObservations,
    reasons: list[HeldOutGateReason],
) -> HeldOutGateOutcome:
    """Decide the verdict, appending any primary-metric reason it establishes."""
    if (
        HeldOutGateReason.FAILURE_RATE_EXCEEDS_POLICY in reasons
        or HeldOutGateReason.DEGRADED_RATE_EXCEEDS_POLICY in reasons
    ):
        return HeldOutGateOutcome.REJECT
    if any(reason in reasons for reason in _INCONCLUSIVE_REASONS):
        return HeldOutGateOutcome.INCONCLUSIVE
    upper = observations.overall_recall_interval_upper
    lower = observations.overall_recall_interval_lower
    minimum = policy.minimum_document_recall
    if upper is not None and _fraction_less(upper, minimum):
        reasons.append(HeldOutGateReason.PRIMARY_METRIC_BELOW_MINIMUM)
        return HeldOutGateOutcome.REJECT
    if lower is not None and _fraction_less(lower, minimum):
        reasons.append(HeldOutGateReason.PRIMARY_METRIC_UNCERTAIN_AT_THRESHOLD)
        return HeldOutGateOutcome.INCONCLUSIVE
    return HeldOutGateOutcome.PASS
