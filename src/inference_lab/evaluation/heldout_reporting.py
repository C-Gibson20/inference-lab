"""Aggregate, compare, serialise, and observe frozen held-out retrieval runs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Final, cast

import structlog

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab._validation import is_strict_integer
from inference_lab.evaluation._validation import is_sha256_checksum
from inference_lab.evaluation.cases import QUESTION_TYPES, QuestionType
from inference_lab.evaluation.heldout_gate import (
    HeldOutCategoryCoverage,
    HeldOutChannel,
    HeldOutGateOutcome,
    HeldOutGatePolicy,
    HeldOutRunContract,
    evaluate_held_out_gate,
)
from inference_lab.evaluation.retrieval_contracts import (
    AggregateMetric,
    MetricFraction,
    RecallMetric,
    RetrievalCaseResult,
    RetrievalEvaluationContractError,
)
from inference_lab.retrieval import FailureCategory, ResultStatus

_LOGGER = structlog.get_logger(__name__)
_REPORT_SCHEMA_VERSION: Final[str] = "1.0.0"
_CANDIDATE_CUTOFF: Final[int] = 10
_CHANNEL_COMPARISON_PAIRS: Final[tuple[tuple[HeldOutChannel, HeldOutChannel], ...]] = (
    (HeldOutChannel.FUSED_PRE_RERANK, HeldOutChannel.LEXICAL_ONLY),
    (HeldOutChannel.FUSED_PRE_RERANK, HeldOutChannel.DENSE_ONLY),
    (HeldOutChannel.FUSED_PRE_RERANK, HeldOutChannel.METADATA),
    (HeldOutChannel.FUSED_RERANKED, HeldOutChannel.FUSED_PRE_RERANK),
)


@dataclass(frozen=True, slots=True)
class HeldOutChannelCaseResult:
    """One sanitised channel result, retaining RET-06 degradation state."""

    channel: HeldOutChannel
    result: RetrievalCaseResult
    degraded: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.channel, HeldOutChannel):
            raise TypeError("channel must be a HeldOutChannel.")
        if not isinstance(self.result, RetrievalCaseResult):
            raise TypeError("result must be a RetrievalCaseResult.")
        if type(self.degraded) is not bool:
            raise TypeError("degraded must be a boolean.")
        if self.result.split != "held_out":
            raise RetrievalEvaluationContractError(
                "held-out channel results must use the held_out split."
            )
        if self.channel is HeldOutChannel.FUSED_RERANKED:
            if len(self.result.hits) > 10:
                raise RetrievalEvaluationContractError(
                    "reranked evidence must respect the RET-06 evidence limit."
                )
        elif self.result.requested_limit < _CANDIDATE_CUTOFF:
            raise RetrievalEvaluationContractError(
                "candidate channel results must request at least ten items."
            )


@dataclass(frozen=True, slots=True)
class HeldOutCaseObservation:
    """Payload-free per-case inputs used only while aggregating a run."""

    case_id: str
    question_type: QuestionType
    expected_abstention: bool
    leakage_group_id: str
    channel_results: tuple[HeldOutChannelCaseResult, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the instance's attributes."""
        if not isinstance(self.case_id, str) or not self.case_id:
            raise TypeError("case_id must be a non-empty string.")
        if self.question_type not in QUESTION_TYPES:
            raise RetrievalEvaluationContractError("unsupported question category.")
        if type(self.expected_abstention) is not bool:
            raise TypeError("expected_abstention must be a boolean.")
        if not isinstance(self.leakage_group_id, str) or not is_sha256_checksum(
            self.leakage_group_id
        ):
            raise RetrievalEvaluationContractError(
                "leakage_group_id must be an opaque SHA-256 identity."
            )
        if not isinstance(self.channel_results, tuple):
            raise TypeError("channel_results must be a tuple.")
        if any(
            not isinstance(item, HeldOutChannelCaseResult)
            for item in self.channel_results
        ):
            raise TypeError(
                "channel_results must contain HeldOutChannelCaseResult values."
            )
        channels = tuple(item.channel for item in self.channel_results)
        if len(channels) != len(set(channels)) or set(channels) != set(HeldOutChannel):
            raise RetrievalEvaluationContractError(
                "every held-out case must retain all five channel outcomes."
            )
        if any(item.result.case_id != self.case_id for item in self.channel_results):
            raise RetrievalEvaluationContractError(
                "channel result case IDs must match their observation."
            )

    def result_for(self, channel: HeldOutChannel) -> HeldOutChannelCaseResult:
        """Return the result for one required channel."""
        if not isinstance(channel, HeldOutChannel):
            raise TypeError("channel must be a HeldOutChannel.")
        return next(item for item in self.channel_results if item.channel is channel)


@dataclass(frozen=True, slots=True)
class HeldOutChannelSummary:
    """Aggregate candidate or reranked evidence measurements for one path."""

    channel: HeldOutChannel
    stage: str
    recall_cutoff: int
    actual_item_count_min: int
    actual_item_count_max: int
    actual_item_count_total: int
    document_recall: AggregateMetric
    passage_recall: AggregateMetric
    recall_interval_lower: MetricFraction | None
    recall_interval_upper: MetricFraction | None
    failed_case_count: int
    degraded_case_count: int
    no_result_case_count: int
    failure_category_counts: tuple[tuple[str, int], ...]
    document_rank_distribution: tuple[tuple[str, int], ...]
    duplicate_diagnostic_totals: tuple[tuple[str, int], ...]
    categories: tuple[HeldOutCategoryCoverage, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the instance's attributes."""
        if not isinstance(self.channel, HeldOutChannel):
            raise TypeError("channel must be a HeldOutChannel.")
        if self.stage not in {"candidate", "final_evidence"}:
            raise RetrievalEvaluationContractError("unsupported retrieval stage.")
        if not is_strict_integer(self.recall_cutoff) or self.recall_cutoff <= 0:
            raise RetrievalEvaluationContractError("recall_cutoff must be positive.")
        population = (
            self.document_recall.applicable_case_count
            + self.document_recall.non_applicable_case_count
        )
        for name, value in (
            ("actual_item_count_min", self.actual_item_count_min),
            ("actual_item_count_max", self.actual_item_count_max),
            ("actual_item_count_total", self.actual_item_count_total),
            ("failed_case_count", self.failed_case_count),
            ("degraded_case_count", self.degraded_case_count),
            ("no_result_case_count", self.no_result_case_count),
        ):
            if not is_strict_integer(value) or value < 0:
                raise RetrievalEvaluationContractError(f"{name} must be non-negative.")
        if self.actual_item_count_min > self.actual_item_count_max:
            raise RetrievalEvaluationContractError(
                "actual item minimum must not exceed its maximum."
            )
        if any(
            value > population
            for value in (
                self.failed_case_count,
                self.degraded_case_count,
                self.no_result_case_count,
            )
        ):
            raise RetrievalEvaluationContractError(
                "case diagnostics must not exceed the channel population."
            )
        if self.passage_recall.cutoff != self.recall_cutoff:
            raise RetrievalEvaluationContractError(
                "passage recall cutoff must match the channel cutoff."
            )
        if not isinstance(self.categories, tuple) or any(
            not isinstance(item, HeldOutCategoryCoverage) for item in self.categories
        ):
            raise TypeError("categories must be a tuple of category coverage values.")
        if sum(item.case_count for item in self.categories) != population:
            raise RetrievalEvaluationContractError(
                "category populations must reconcile to the channel population."
            )
        if not isinstance(self.failure_category_counts, tuple):
            raise TypeError("failure_category_counts must be a tuple.")
        if any(
            not isinstance(name, str)
            or name not in {category.value for category in FailureCategory}
            or not is_strict_integer(count)
            or count <= 0
            for name, count in self.failure_category_counts
        ):
            raise RetrievalEvaluationContractError(
                "failure_category_counts must contain bounded failure counts."
            )
        expected_rank_buckets = (
            "rank_1",
            "rank_2_to_5",
            "rank_6_to_10",
            "rank_over_10",
            "unretrieved",
        )
        if tuple(name for name, _ in self.document_rank_distribution) != (
            expected_rank_buckets
        ):
            raise RetrievalEvaluationContractError(
                "document rank distribution must use canonical buckets."
            )
        if any(
            not is_strict_integer(count) or count < 0
            for _, count in self.document_rank_distribution
        ):
            raise RetrievalEvaluationContractError(
                "document rank counts must be non-negative integers."
            )
        if sum(count for _, count in self.document_rank_distribution) != (
            self.document_recall.relevant_target_count
        ):
            raise RetrievalEvaluationContractError(
                "document rank distribution must reconcile to relevant targets."
            )
        if not isinstance(self.duplicate_diagnostic_totals, tuple) or any(
            not isinstance(name, str) or not is_strict_integer(count) or count < 0
            for name, count in self.duplicate_diagnostic_totals
        ):
            raise RetrievalEvaluationContractError(
                "duplicate diagnostic totals must be bounded non-negative counts."
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the channel summary."""
        return {
            "channel": self.channel.value,
            "stage": self.stage,
            "recall_cutoff": self.recall_cutoff,
            "actual_item_count_min": self.actual_item_count_min,
            "actual_item_count_max": self.actual_item_count_max,
            "actual_item_count_total": self.actual_item_count_total,
            "document_recall": self.document_recall.as_dict(),
            "passage_recall": self.passage_recall.as_dict(),
            "recall_interval_lower": _fraction_dict(self.recall_interval_lower),
            "recall_interval_upper": _fraction_dict(self.recall_interval_upper),
            "failed_case_count": self.failed_case_count,
            "degraded_case_count": self.degraded_case_count,
            "no_result_case_count": self.no_result_case_count,
            "failure_category_counts": {
                name: count for name, count in self.failure_category_counts
            },
            "document_rank_distribution": {
                name: count for name, count in self.document_rank_distribution
            },
            "duplicate_diagnostic_totals": {
                name: count for name, count in self.duplicate_diagnostic_totals
            },
            "categories": [item.as_dict() for item in self.categories],
        }


@dataclass(frozen=True, slots=True)
class HeldOutPairedComparison:
    """Paired query-level comparison for two candidate retrieval paths."""

    candidate: HeldOutChannel
    reference: HeldOutChannel
    cutoff: int
    paired_case_count: int
    candidate_wins: int
    ties: int
    reference_wins: int
    non_applicable_case_count: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the instance's attributes."""
        if not isinstance(self.candidate, HeldOutChannel):
            raise TypeError("candidate must be a HeldOutChannel.")
        if not isinstance(self.reference, HeldOutChannel):
            raise TypeError("reference must be a HeldOutChannel.")
        for name in (
            "cutoff",
            "paired_case_count",
            "candidate_wins",
            "ties",
            "reference_wins",
            "non_applicable_case_count",
        ):
            value = getattr(self, name)
            if not is_strict_integer(value):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise RetrievalEvaluationContractError(f"{name} must not be negative.")
        if self.candidate is self.reference:
            raise RetrievalEvaluationContractError("comparison paths must differ.")
        if self.cutoff <= 0:
            raise RetrievalEvaluationContractError("cutoff must be positive.")
        if self.candidate_wins + self.ties + self.reference_wins != (
            self.paired_case_count
        ):
            raise RetrievalEvaluationContractError(
                "paired outcomes must reconcile to paired_case_count."
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the paired comparison."""
        return {
            "candidate": self.candidate.value,
            "reference": self.reference.value,
            "cutoff": self.cutoff,
            "paired_case_count": self.paired_case_count,
            "candidate_wins": self.candidate_wins,
            "ties": self.ties,
            "reference_wins": self.reference_wins,
            "non_applicable_case_count": self.non_applicable_case_count,
        }


@dataclass(frozen=True, slots=True)
class HeldOutBaselineReport:
    """Aggregate-only, reproducible result of an EVAL-04 held-out run."""

    contract: HeldOutRunContract
    case_count: int
    channel_summaries: tuple[HeldOutChannelSummary, ...]
    paired_comparisons: tuple[HeldOutPairedComparison, ...]
    outcome: HeldOutGateOutcome
    outcome_reasons: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    schema_version: str = _REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the instance's attributes."""
        if not isinstance(self.contract, HeldOutRunContract):
            raise TypeError("contract must be a HeldOutRunContract.")
        if not is_strict_integer(self.case_count) or self.case_count <= 0:
            raise RetrievalEvaluationContractError("case_count must be positive.")
        if not isinstance(self.channel_summaries, tuple) or any(
            not isinstance(item, HeldOutChannelSummary)
            for item in self.channel_summaries
        ):
            raise TypeError("channel_summaries must be a tuple of summaries.")
        if tuple(item.channel for item in self.channel_summaries) != tuple(
            sorted(HeldOutChannel, key=lambda channel: channel.value)
        ):
            raise RetrievalEvaluationContractError(
                "channel summaries must contain every channel in canonical order."
            )
        if not isinstance(self.paired_comparisons, tuple) or any(
            not isinstance(item, HeldOutPairedComparison)
            for item in self.paired_comparisons
        ):
            raise TypeError("paired_comparisons must contain paired comparisons.")
        if not isinstance(self.outcome, HeldOutGateOutcome):
            raise TypeError("outcome must be a HeldOutGateOutcome.")
        for name, values in (
            ("outcome_reasons", self.outcome_reasons),
            ("limitations", self.limitations),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise TypeError(f"{name} must be a tuple of non-empty strings.")
        if (
            not isinstance(self.schema_version, str)
            or self.schema_version != _REPORT_SCHEMA_VERSION
        ):
            raise RetrievalEvaluationContractError("unsupported report schema version.")

    @property
    def digest(self) -> str:
        """Return the deterministic digest of the complete aggregate report."""
        digest = hashlib.sha256(
            canonical_json_bytes(cast(JsonObject, self.as_dict()))
        ).hexdigest()
        return f"sha256:{digest}"

    def as_dict(self) -> dict[str, object]:
        """Return the aggregate-first report without private case payloads."""
        return {
            "schema_version": self.schema_version,
            "run_contract": self.contract.as_dict(),
            "case_count": self.case_count,
            "channels": [item.as_dict() for item in self.channel_summaries],
            "paired_comparisons": [item.as_dict() for item in self.paired_comparisons],
            "gate": {
                "outcome": self.outcome.value,
                "reasons": list(self.outcome_reasons),
                "policy": self.contract.gate_policy.as_dict(),
            },
            "limitations": list(self.limitations),
        }


def _fraction_dict(value: MetricFraction | None) -> dict[str, int] | None:
    return value.as_dict() if value is not None else None


def _fraction_sum(metrics: tuple[RecallMetric | None, ...]) -> tuple[int, int]:
    selected = tuple(metric.value for metric in metrics if metric is not None)
    return (
        sum(value.numerator for value in selected),
        sum(value.denominator for value in selected),
    )


def _aggregate_recall(
    metrics: tuple[RecallMetric | None, ...],
    results: tuple[RetrievalCaseResult, ...],
    cutoff: int,
) -> AggregateMetric:
    """Aggregate exact counts while retaining every case outcome."""
    selected = tuple(metric.value for metric in metrics if metric is not None)
    matched, relevant = _fraction_sum(metrics)
    macro = None
    if selected:
        macro_value = sum(
            (Fraction(value.numerator, value.denominator) for value in selected),
            Fraction(0),
        ) / len(selected)
        macro = MetricFraction(macro_value.numerator, macro_value.denominator)
    micro = MetricFraction(matched, relevant) if relevant else None
    completed = sum(item.status is ResultStatus.COMPLETED for item in results)
    partial = sum(item.status is ResultStatus.PARTIAL for item in results)
    failed = sum(item.status is ResultStatus.FAILED for item in results)
    empty = sum(item.is_empty for item in results)
    return AggregateMetric(
        cutoff=cutoff,
        macro=macro,
        micro=micro,
        applicable_case_count=len(selected),
        non_applicable_case_count=len(results) - len(selected),
        completed_case_count=completed,
        partial_case_count=partial,
        failed_case_count=failed,
        empty_case_count=empty,
        matched_target_count=matched,
        relevant_target_count=relevant,
    )


def _metric_at(
    result: RetrievalCaseResult,
    *,
    passage_cutoff: int,
) -> tuple[RecallMetric | None, RecallMetric | None]:
    """Return document Recall@10 and the requested passage recall metric."""
    passage_metric = next(
        (
            item
            for item in result.passage_recall
            if item and item.cutoff == passage_cutoff
        ),
        None,
    )
    return result.document_recall, passage_metric


def _bootstrap_interval(
    observations: tuple[tuple[str, MetricFraction | None], ...],
    *,
    policy: HeldOutGatePolicy,
    seed: str,
) -> tuple[MetricFraction | None, MetricFraction | None]:
    """Return a deterministic percentile interval resampling leakage groups."""
    by_group: dict[str, list[MetricFraction]] = defaultdict(list)
    for group_id, metric in observations:
        if metric is not None:
            by_group[group_id].append(metric)
    groups = tuple(sorted(by_group))
    if not groups:
        return None, None
    group_counts = {
        group_id: (
            sum(item.numerator for item in by_group[group_id]),
            sum(item.denominator for item in by_group[group_id]),
        )
        for group_id in groups
    }
    samples: list[Fraction] = []
    group_count = len(groups)
    for replicate in range(policy.bootstrap_replicates):
        matched = 0
        relevant = 0
        for draw in range(group_count):
            material = f"{seed}:{replicate}:{draw}".encode("ascii")
            position = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            group_id = groups[position % group_count]
            group_matched, group_relevant = group_counts[group_id]
            matched += group_matched
            relevant += group_relevant
        if relevant:
            samples.append(Fraction(matched, relevant))
    if not samples:
        return None, None
    samples.sort()
    tail_percent = 100 - policy.confidence_level_percent
    lower_rank = max(1, (len(samples) * tail_percent + 199) // 200)
    upper_rank = max(1, (len(samples) * (200 - tail_percent) + 199) // 200)
    lower = samples[lower_rank - 1]
    upper = samples[upper_rank - 1]
    return (
        MetricFraction(lower.numerator, lower.denominator),
        MetricFraction(upper.numerator, upper.denominator),
    )


def _category_coverage(
    observations: tuple[HeldOutCaseObservation, ...],
    *,
    channel: HeldOutChannel,
    passage_cutoff: int,
    policy: HeldOutGatePolicy,
    seed: str,
) -> HeldOutCategoryCoverage:
    """Aggregate one category and its group-aware document recall interval."""
    results = tuple(item.result_for(channel) for item in observations)
    metrics = tuple(
        _metric_at(item.result, passage_cutoff=passage_cutoff)[0] for item in results
    )
    matched, relevant = _fraction_sum(metrics)
    recall = MetricFraction(matched, relevant) if relevant else None
    lower, upper = _bootstrap_interval(
        tuple(
            (item.leakage_group_id, metric.value if metric is not None else None)
            for item, metric in zip(observations, metrics, strict=True)
        ),
        policy=policy,
        seed=f"{seed}:{channel.value}:{observations[0].question_type}",
    )
    groups = {item.leakage_group_id for item in observations}
    return HeldOutCategoryCoverage(
        question_type=observations[0].question_type,
        case_count=len(observations),
        answerable_case_count=sum(
            not item.expected_abstention for item in observations
        ),
        expected_abstention_case_count=sum(
            item.expected_abstention for item in observations
        ),
        leakage_group_count=len(groups),
        failed_case_count=sum(
            item.result.status is ResultStatus.FAILED for item in results
        ),
        degraded_case_count=sum(item.degraded for item in results),
        document_recall_cutoff=passage_cutoff,
        document_recall=recall,
        recall_interval_lower=lower,
        recall_interval_upper=upper,
    )


def _summarise_channel(
    observations: tuple[HeldOutCaseObservation, ...],
    *,
    channel: HeldOutChannel,
    contract: HeldOutRunContract,
) -> HeldOutChannelSummary:
    """Build overall and category metrics for one frozen retrieval path."""
    selected_count = (
        contract.final_evidence_count
        if channel is HeldOutChannel.FUSED_RERANKED
        else contract.candidate_recall_cutoff
    )
    stage = (
        "final_evidence" if channel is HeldOutChannel.FUSED_RERANKED else "candidate"
    )
    passage_cutoff = selected_count
    entries = tuple(item.result_for(channel) for item in observations)
    results = tuple(item.result for item in entries)
    document_metrics = tuple(
        _metric_at(result, passage_cutoff=passage_cutoff)[0] for result in results
    )
    passage_metrics = tuple(
        _metric_at(result, passage_cutoff=passage_cutoff)[1] for result in results
    )
    document = _aggregate_recall(document_metrics, results, selected_count)
    passage = _aggregate_recall(passage_metrics, results, passage_cutoff)
    matched, relevant = _fraction_sum(document_metrics)
    point = MetricFraction(matched, relevant) if relevant else None
    lower, upper = _bootstrap_interval(
        tuple(
            (
                case.leakage_group_id,
                metric.value if metric is not None else None,
            )
            for case, metric in zip(observations, document_metrics, strict=True)
        ),
        policy=contract.gate_policy,
        seed=f"{contract.fingerprint}:{channel.value}:overall",
    )
    categories: list[HeldOutCategoryCoverage] = []
    for question_type in QUESTION_TYPES:
        category_cases = tuple(
            item for item in observations if item.question_type == question_type
        )
        if category_cases:
            categories.append(
                _category_coverage(
                    category_cases,
                    channel=channel,
                    passage_cutoff=passage_cutoff,
                    policy=contract.gate_policy,
                    seed=str(contract.fingerprint),
                )
            )
    # Retain the measured point in the interval contract even for an empty
    # result population; the gate sees null when the gold denominator is zero.
    if point is not None and (lower is None or upper is None):
        raise RetrievalEvaluationContractError(
            "bootstrap interval is unavailable for an applicable recall metric."
        )
    failure_counts: dict[str, int] = defaultdict(int)
    rank_counts = {
        "rank_1": 0,
        "rank_2_to_5": 0,
        "rank_6_to_10": 0,
        "rank_over_10": 0,
        "unretrieved": 0,
    }
    duplicate_totals = {
        field_name: 0
        for field_name in (
            "raw_hit_count",
            "distinct_record_count",
            "distinct_document_count",
            "distinct_duplicate_group_count",
            "distinct_family_count",
            "exact_document_repeat_excess",
            "byte_identical_group_repeat_excess",
            "family_member_repeat_excess",
            "relevant_non_gold_duplicate_copy_count",
        )
    }
    for result in results:
        for failure in result.failures:
            failure_counts[failure.category.value] += 1
        for rank in result.document_match_ranks:
            if rank is None:
                rank_counts["unretrieved"] += 1
            elif rank == 1:
                rank_counts["rank_1"] += 1
            elif rank <= 5:
                rank_counts["rank_2_to_5"] += 1
            elif rank <= 10:
                rank_counts["rank_6_to_10"] += 1
            else:
                rank_counts["rank_over_10"] += 1
        diagnostics = result.duplicate_diagnostics
        if diagnostics is not None:
            for name in duplicate_totals:
                duplicate_totals[name] += getattr(diagnostics, name)
    return HeldOutChannelSummary(
        channel=channel,
        stage=stage,
        recall_cutoff=selected_count,
        actual_item_count_min=min(len(item.hits) for item in results),
        actual_item_count_max=max(len(item.hits) for item in results),
        actual_item_count_total=sum(len(item.hits) for item in results),
        document_recall=document,
        passage_recall=passage,
        recall_interval_lower=lower,
        recall_interval_upper=upper,
        failed_case_count=sum(item.status is ResultStatus.FAILED for item in results),
        degraded_case_count=sum(item.degraded for item in entries),
        no_result_case_count=sum(item.is_empty for item in results),
        failure_category_counts=tuple(sorted(failure_counts.items())),
        document_rank_distribution=tuple(rank_counts.items()),
        duplicate_diagnostic_totals=tuple(duplicate_totals.items()),
        categories=tuple(categories),
    )


def _document_metric_at(
    result: RetrievalCaseResult,
    cutoff: int,
) -> MetricFraction | None:
    """Calculate document recall from EVAL-03 target ranks at one cutoff."""
    if not result.document_match_ranks:
        return None
    return MetricFraction(
        sum(
            rank is not None and rank <= cutoff for rank in result.document_match_ranks
        ),
        len(result.document_match_ranks),
    )


def _compare_pair(
    observations: tuple[HeldOutCaseObservation, ...],
    candidate: HeldOutChannel,
    reference: HeldOutChannel,
    cutoff: int,
) -> HeldOutPairedComparison:
    """Compare paired micro recall counts without treating queries as unrelated."""
    candidate_wins = ties = reference_wins = non_applicable = 0
    paired = 0
    for observation in observations:
        candidate_result = observation.result_for(candidate).result
        reference_result = observation.result_for(reference).result
        candidate_metric = _document_metric_at(candidate_result, cutoff)
        reference_metric = _document_metric_at(reference_result, cutoff)
        if candidate_metric is None or reference_metric is None:
            non_applicable += 1
            continue
        if candidate_metric.denominator != reference_metric.denominator:
            raise RetrievalEvaluationContractError(
                "paired channel relevance denominators must agree."
            )
        paired += 1
        left = candidate_metric.numerator
        right = reference_metric.numerator
        if left > right:
            candidate_wins += 1
        elif left < right:
            reference_wins += 1
        else:
            ties += 1
    return HeldOutPairedComparison(
        candidate=candidate,
        reference=reference,
        cutoff=cutoff,
        paired_case_count=paired,
        candidate_wins=candidate_wins,
        ties=ties,
        reference_wins=reference_wins,
        non_applicable_case_count=non_applicable,
    )


def build_held_out_baseline_report(
    contract: HeldOutRunContract,
    observations: tuple[HeldOutCaseObservation, ...],
    *,
    limitations: tuple[str, ...] = (),
) -> HeldOutBaselineReport:
    """Aggregate one frozen held-out run into a deterministic local report."""
    if not isinstance(contract, HeldOutRunContract):
        raise TypeError("contract must be a HeldOutRunContract.")
    if not isinstance(observations, tuple) or not observations:
        raise TypeError("observations must be a non-empty tuple.")
    if any(not isinstance(item, HeldOutCaseObservation) for item in observations):
        raise TypeError("observations must contain HeldOutCaseObservation values.")
    case_ids = tuple(item.case_id for item in observations)
    if case_ids != contract.evaluation_identity.selected_case_ids:
        raise RetrievalEvaluationContractError(
            "observations must match the frozen case order and membership."
        )
    corpus_id = contract.evaluation_identity.corpus_id
    corpus_version = contract.evaluation_identity.corpus_version
    for observation in observations:
        for channel_result in observation.channel_results:
            if (
                channel_result.result.corpus_id != corpus_id
                or channel_result.result.corpus_version != corpus_version
            ):
                raise RetrievalEvaluationContractError(
                    "channel results must use the frozen corpus version."
                )
    summaries = tuple(
        _summarise_channel(observations, channel=channel, contract=contract)
        for channel in sorted(HeldOutChannel, key=lambda item: item.value)
    )
    comparisons = tuple(
        _compare_pair(
            observations,
            candidate,
            reference,
            (
                contract.final_evidence_count
                if candidate is HeldOutChannel.FUSED_RERANKED
                else contract.candidate_recall_cutoff
            ),
        )
        for candidate, reference in _CHANNEL_COMPARISON_PAIRS
    )
    gate_summary = next(
        item for item in summaries if item.channel is HeldOutChannel.FUSED_PRE_RERANK
    )
    gate_outcome, gate_reasons = evaluate_held_out_gate(
        policy=contract.gate_policy,
        categories=gate_summary.categories,
        overall_document_recall_at_10=gate_summary.document_recall.micro,
        overall_recall_interval_lower=gate_summary.recall_interval_lower,
        overall_recall_interval_upper=gate_summary.recall_interval_upper,
        case_count=len(observations),
        failed_case_count=gate_summary.failed_case_count,
        degraded_case_count=gate_summary.degraded_case_count,
    )
    candidate_document_denominator = gate_summary.document_recall.relevant_target_count
    candidate_passage_denominator = gate_summary.passage_recall.relevant_target_count
    for summary in summaries:
        if summary.channel is HeldOutChannel.FUSED_PRE_RERANK:
            continue
        if summary.stage == "candidate" and (
            summary.document_recall.relevant_target_count
            != candidate_document_denominator
            or summary.passage_recall.relevant_target_count
            != candidate_passage_denominator
        ):
            raise RetrievalEvaluationContractError(
                "candidate channel relevance denominators must be comparable."
            )
    final_summary = next(
        item for item in summaries if item.channel is HeldOutChannel.FUSED_RERANKED
    )
    final_measurement_incomplete = (
        final_summary.document_recall.relevant_target_count
        != candidate_document_denominator
        or final_summary.passage_recall.relevant_target_count
        != candidate_passage_denominator
    )
    selected_limitations = list(limitations)
    selected_reasons = list(gate_reasons)
    if final_measurement_incomplete:
        selected_limitations.append("final_evidence_recall_incomplete")
        selected_reasons.append("incomplete_final_evidence_measurement")
        gate_outcome = HeldOutGateOutcome.INCONCLUSIVE
    if len(selected_limitations) != len(set(selected_limitations)):
        raise RetrievalEvaluationContractError("limitations must be unique.")
    return HeldOutBaselineReport(
        contract=contract,
        case_count=len(observations),
        channel_summaries=summaries,
        paired_comparisons=comparisons,
        outcome=gate_outcome,
        outcome_reasons=tuple(selected_reasons),
        limitations=tuple(selected_limitations),
    )


def serialise_held_out_baseline_report(report: HeldOutBaselineReport) -> bytes:
    """Return canonical UTF-8 JSON bytes for an aggregate-only report."""
    if not isinstance(report, HeldOutBaselineReport):
        raise TypeError("report must be a HeldOutBaselineReport.")
    return canonical_json_bytes(cast(JsonObject, report.as_dict()))


def render_held_out_baseline_report(report: HeldOutBaselineReport) -> str:
    """Render a stable Markdown summary without case or evidence payloads."""
    if not isinstance(report, HeldOutBaselineReport):
        raise TypeError("report must be a HeldOutBaselineReport.")
    lines = [
        "# Held-out retrieval baseline",
        "",
        f"- Outcome: **{report.outcome.value}**",
        f"- Cases: {report.case_count}",
        f"- Run fingerprint: `{report.contract.fingerprint}`",
        f"- Report digest: `{report.digest}`",
        "",
        "## Channel results",
        "",
        "| Channel | Stage | Recall cutoff | Items min/max/total | Document recall (micro) | Passage recall (micro) | Failed | Degraded | No result |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in report.channel_summaries:
        lines.append(
            "| {channel} | {stage} | {cutoff} | {items} | {document} | {passage} | "
            "{failed} | {degraded} | {empty} |".format(
                channel=summary.channel.value,
                stage=summary.stage,
                cutoff=summary.recall_cutoff,
                items=(
                    f"{summary.actual_item_count_min}/"
                    f"{summary.actual_item_count_max}/"
                    f"{summary.actual_item_count_total}"
                ),
                document=_render_fraction(summary.document_recall.micro),
                passage=_render_fraction(summary.passage_recall.micro),
                failed=summary.failed_case_count,
                degraded=summary.degraded_case_count,
                empty=summary.no_result_case_count,
            )
        )
    lines.extend(
        (
            "",
            "## Category coverage",
            "",
            "| Channel | Category | Cases | Answerable | Expected abstention | Leakage groups | Recall cutoff | Recall | Interval |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for summary in report.channel_summaries:
        for category in summary.categories:
            interval = "—"
            if category.recall_interval_lower and category.recall_interval_upper:
                interval = (
                    f"{_render_fraction(category.recall_interval_lower)}–"
                    f"{_render_fraction(category.recall_interval_upper)}"
                )
            lines.append(
                "| {channel} | {category} | {cases} | {answerable} | {abstention} | "
                "{groups} | {cutoff} | {recall} | {interval} |".format(
                    channel=summary.channel.value,
                    category=category.question_type,
                    cases=category.case_count,
                    answerable=category.answerable_case_count,
                    abstention=category.expected_abstention_case_count,
                    groups=category.leakage_group_count,
                    cutoff=category.document_recall_cutoff,
                    recall=_render_fraction(category.document_recall),
                    interval=interval,
                )
            )
    lines.extend(
        (
            "",
            "## Paired candidate comparisons",
            "",
            "| Candidate | Reference | Cutoff | Paired | Candidate wins | Ties | Reference wins | Non-applicable |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for comparison in report.paired_comparisons:
        lines.append(
            f"| {comparison.candidate.value} | {comparison.reference.value} | "
            f"{comparison.cutoff} | {comparison.paired_case_count} | "
            f"{comparison.candidate_wins} | "
            f"{comparison.ties} | {comparison.reference_wins} | "
            f"{comparison.non_applicable_case_count} |"
        )
    lines.extend(
        ("", "## Gate", "", f"- Reasons: {', '.join(report.outcome_reasons) or 'none'}")
    )
    if report.limitations:
        lines.extend(("", "## Limitations", ""))
        lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"


def _render_fraction(value: MetricFraction | None) -> str:
    if value is None:
        return "not applicable"
    return f"{value.numerator}/{value.denominator}"


def log_held_out_baseline_report(report: HeldOutBaselineReport) -> None:
    """Emit one payload-free aggregate lifecycle event for a completed report."""
    if not isinstance(report, HeldOutBaselineReport):
        raise TypeError("report must be a HeldOutBaselineReport.")
    try:
        _LOGGER.info(
            "held_out_evaluation_completed",
            pipeline_stage="evaluation",
            corpus_id=report.contract.evaluation_identity.corpus_id.as_str(),
            corpus_version=report.contract.evaluation_identity.corpus_version.as_str(),
            run_fingerprint=report.contract.fingerprint,
            report_digest=report.digest,
            selected_case_count=report.case_count,
            gate_outcome=report.outcome.value,
            failed_channel_result_count=sum(
                item.failed_case_count for item in report.channel_summaries
            ),
            degraded_channel_result_count=sum(
                item.degraded_case_count for item in report.channel_summaries
            ),
        )
    except Exception:
        pass
