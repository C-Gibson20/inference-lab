"""EVAL-04 frozen gate policy and coverage decision contract."""

import hashlib
from dataclasses import replace
from typing import cast

import pytest

from inference_lab.evaluation.cases import QUESTION_TYPES
from inference_lab.evaluation.contracts import SplitName
from inference_lab.evaluation.heldout_gate import (
    HELD_OUT_GATE_POLICY_VERSION,
    HELD_OUT_RUN_CONTRACT_VERSION,
    HELD_OUT_UNCERTAINTY_METHOD,
    HeldOutCategoryCoverage,
    HeldOutChannel,
    HeldOutChannelIdentity,
    HeldOutGateOutcome,
    HeldOutGatePolicy,
    HeldOutRunContract,
    evaluate_held_out_gate,
)
from inference_lab.evaluation.retrieval_contracts import (
    MetricFraction,
    RetrievalComponentIdentity,
    RetrievalEvaluationContractError,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRunIdentity,
    RetrievalSystemIdentity,
)
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import FusionPolicy
from inference_lab.retrieval.reranking import RerankingIdentity


def _policy() -> HeldOutGatePolicy:
    return HeldOutGatePolicy(
        minimum_document_recall_at_10=MetricFraction(8, 10),
        maximum_failure_rate=MetricFraction(1, 20),
        maximum_degraded_rate=MetricFraction(1, 10),
        minimum_category_cases=2,
        minimum_category_leakage_groups=2,
        confidence_level_percent=95,
        bootstrap_replicates=2_000,
        rationale="Synthetic fixture values; not a production threshold decision.",
    )


def _checksum(seed: str) -> str:
    return f"sha256:{hashlib.sha256(seed.encode()).hexdigest()}"


def _system_identity(seed: str) -> RetrievalSystemIdentity:
    return RetrievalSystemIdentity(
        adapter_implementation="local-adapter",
        adapter_contract_version="1.0.0",
        query_construction_algorithm="verbatim",
        query_construction_version="1.0.0",
        retrieval_algorithm="frozen-retrieval",
        retrieval_algorithm_version="1.0.0",
        configuration_digest=_checksum(seed),
        components=(
            RetrievalComponentIdentity(
                component="index",
                implementation="fixture-index",
                format_version="1.0.0",
                contract_version="1.0.0",
                configuration_digest=_checksum(f"index-{seed}"),
                generation_id=f"generation-{seed}",
                source_manifest_digest=_checksum(f"manifest-{seed}"),
            ),
        ),
    )


def _run_identity(split: SplitName = "held_out") -> RetrievalEvaluationRunIdentity:
    return RetrievalEvaluationRunIdentity(
        dataset_id="dataset-eval04",
        dataset_version="1.0.0",
        dataset_schema_version="3.0.0",
        case_file_checksum=_checksum("cases"),
        corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000001"),
        corpus_version=CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002"),
        selected_split=split,
        selected_case_ids=("held-out-001",),
        policy=RetrievalEvaluationPolicy(passage_cutoffs=(10,), candidate_depth=10),
        relationship_snapshot_digest=_checksum("relationships"),
        system_identity=_system_identity("fused"),
    )


def _run_contract(*, final_count: int = 8) -> HeldOutRunContract:
    fusion_policy = FusionPolicy()
    return HeldOutRunContract(
        evaluation_identity=_run_identity(),
        gate_policy=_policy(),
        channel_identities=tuple(
            HeldOutChannelIdentity(
                channel,
                _system_identity(channel.value),
                fusion_policy=(
                    fusion_policy
                    if channel
                    in {
                        HeldOutChannel.FUSED_PRE_RERANK,
                        HeldOutChannel.FUSED_RERANKED,
                    }
                    else None
                ),
                reranking_identity=(
                    RerankingIdentity(
                        backend_name="fixture",
                        model_name="fixture-reranker",
                        model_revision="1.0.0",
                        preprocessing_profile_version="1.0.0",
                        evidence_count=final_count,
                        candidate_limit=100,
                    )
                    if channel is HeldOutChannel.FUSED_RERANKED
                    else None
                ),
            )
            for channel in HeldOutChannel
        ),
        final_evidence_count=final_count,
    )


def _categories(
    *,
    recall: MetricFraction = MetricFraction(9, 10),
    lower: MetricFraction = MetricFraction(85, 100),
    upper: MetricFraction = MetricFraction(95, 100),
) -> tuple[HeldOutCategoryCoverage, ...]:
    return tuple(
        HeldOutCategoryCoverage(
            question_type=question_type,
            case_count=2,
            answerable_case_count=1,
            expected_abstention_case_count=1,
            leakage_group_count=2,
            failed_case_count=0,
            degraded_case_count=0,
            document_recall_cutoff=10,
            document_recall=recall,
            recall_interval_lower=lower,
            recall_interval_upper=upper,
        )
        for question_type in QUESTION_TYPES
    )


def _decide(
    *,
    categories: tuple[HeldOutCategoryCoverage, ...] | None = None,
    recall: MetricFraction | None = MetricFraction(9, 10),
    lower: MetricFraction | None = MetricFraction(85, 100),
    upper: MetricFraction | None = MetricFraction(95, 100),
    failed: int = 0,
    degraded: int = 0,
) -> tuple[HeldOutGateOutcome, tuple[str, ...]]:
    selected_categories = _categories() if categories is None else categories
    if categories is None and (failed or degraded):
        selected_categories = (
            replace(
                selected_categories[0],
                failed_case_count=failed,
                degraded_case_count=degraded,
            ),
            *selected_categories[1:],
        )
    return evaluate_held_out_gate(
        policy=_policy(),
        categories=selected_categories,
        overall_document_recall_at_10=recall,
        overall_recall_interval_lower=lower,
        overall_recall_interval_upper=upper,
        case_count=2 * len(QUESTION_TYPES),
        failed_case_count=failed,
        degraded_case_count=degraded,
    )


def test_policy_persists_thresholds_rationale_and_uncertainty_identity() -> None:
    policy = _policy()

    assert policy.version == HELD_OUT_GATE_POLICY_VERSION
    assert policy.uncertainty_method == HELD_OUT_UNCERTAINTY_METHOD
    assert _run_contract().as_dict()["contract_version"] == (
        HELD_OUT_RUN_CONTRACT_VERSION
    )
    assert policy.as_dict() == {
        "version": "1.0.0",
        "minimum_document_recall_at_10": {"numerator": 8, "denominator": 10},
        "maximum_failure_rate": {"numerator": 1, "denominator": 20},
        "maximum_degraded_rate": {"numerator": 1, "denominator": 10},
        "minimum_category_cases": 2,
        "minimum_category_leakage_groups": 2,
        "confidence_level_percent": 95,
        "uncertainty_method": "leakage-cluster-bootstrap",
        "bootstrap_replicates": 2_000,
        "rationale": "Synthetic fixture values; not a production threshold decision.",
    }


def test_frozen_run_contract_binds_holdout_policy_channels_and_cutoffs() -> None:
    contract = _run_contract()
    permuted = replace(
        contract, channel_identities=tuple(reversed(contract.channel_identities))
    )
    changed_count = _run_contract(final_count=7)

    assert contract.fingerprint == permuted.fingerprint
    assert contract.fingerprint != changed_count.fingerprint
    assert contract.as_dict()["candidate_recall_cutoff"] == 10
    assert contract.as_dict()["final_evidence_count"] == 8
    identities = cast(list[object], contract.as_dict()["channel_identities"])
    assert len(identities) == len(HeldOutChannel)


def test_frozen_run_contract_rejects_non_heldout_or_incomplete_channel_identity() -> (
    None
):
    with pytest.raises(RetrievalEvaluationContractError, match="held_out"):
        HeldOutRunContract(
            evaluation_identity=_run_identity("development"),
            gate_policy=_policy(),
            channel_identities=_run_contract().channel_identities,
        )

    with pytest.raises(RetrievalEvaluationContractError, match="every Phase 6"):
        HeldOutRunContract(
            evaluation_identity=_run_identity(),
            gate_policy=_policy(),
            channel_identities=_run_contract().channel_identities[:-1],
        )


def test_gate_passes_only_when_lower_bound_clears_predeclared_minimum() -> None:
    outcome, reasons = _decide()

    assert outcome is HeldOutGateOutcome.PASS
    assert reasons == ()


def test_gate_rejects_when_upper_bound_is_below_minimum() -> None:
    outcome, reasons = _decide(
        recall=MetricFraction(65, 100),
        lower=MetricFraction(6, 10),
        upper=MetricFraction(7, 10),
    )

    assert outcome is HeldOutGateOutcome.REJECT
    assert "primary_metric_below_minimum" in reasons


def test_gate_is_inconclusive_for_crossing_interval_or_missing_coverage() -> None:
    crossing, crossing_reasons = _decide(
        lower=MetricFraction(7, 10),
        upper=MetricFraction(9, 10),
    )
    missing, missing_reasons = _decide(
        categories=_categories()[:-1],
    )

    assert crossing is HeldOutGateOutcome.INCONCLUSIVE
    assert "primary_metric_uncertain_at_threshold" in crossing_reasons
    assert missing is HeldOutGateOutcome.INCONCLUSIVE
    assert "missing_question_categories" in missing_reasons


def test_gate_rejects_failure_rate_over_the_fixed_limit() -> None:
    outcome, reasons = _decide(failed=1)

    assert outcome is HeldOutGateOutcome.REJECT
    assert "failure_rate_exceeds_policy" in reasons


def test_gate_is_inconclusive_when_a_category_lacks_case_or_group_coverage() -> None:
    categories = list(_categories())
    categories[0] = replace(
        categories[0],
        case_count=1,
        answerable_case_count=1,
        expected_abstention_case_count=0,
        leakage_group_count=1,
    )
    categories[1] = replace(
        categories[1],
        case_count=3,
        answerable_case_count=2,
        leakage_group_count=3,
    )

    outcome, reasons = _decide(categories=tuple(categories))

    assert outcome is HeldOutGateOutcome.INCONCLUSIVE
    assert "underpowered_question_categories" in reasons


def test_gate_requires_both_answerable_and_expected_abstention_cases() -> None:
    no_abstentions = tuple(
        replace(category, answerable_case_count=2, expected_abstention_case_count=0)
        for category in _categories()
    )

    outcome, reasons = _decide(categories=no_abstentions)

    assert outcome is HeldOutGateOutcome.INCONCLUSIVE
    assert "missing_answerability_status" in reasons


def test_category_contract_rejects_incoherent_population_and_interval() -> None:
    with pytest.raises(RetrievalEvaluationContractError, match="reconcile"):
        HeldOutCategoryCoverage(
            question_type=QUESTION_TYPES[0],
            case_count=2,
            answerable_case_count=2,
            expected_abstention_case_count=1,
            leakage_group_count=2,
            failed_case_count=0,
            degraded_case_count=0,
            document_recall_cutoff=10,
            document_recall=MetricFraction(9, 10),
            recall_interval_lower=MetricFraction(8, 10),
            recall_interval_upper=MetricFraction(95, 100),
        )

    with pytest.raises(RetrievalEvaluationContractError, match="upper bound"):
        HeldOutCategoryCoverage(
            question_type=QUESTION_TYPES[0],
            case_count=2,
            answerable_case_count=1,
            expected_abstention_case_count=1,
            leakage_group_count=2,
            failed_case_count=0,
            degraded_case_count=0,
            document_recall_cutoff=10,
            document_recall=MetricFraction(9, 10),
            recall_interval_lower=MetricFraction(99, 100),
            recall_interval_upper=MetricFraction(95, 100),
        )


def test_policy_rejects_unreasoned_thresholds_and_unsupported_uncertainty() -> None:
    with pytest.raises(RetrievalEvaluationContractError, match="rationale"):
        HeldOutGatePolicy(
            minimum_document_recall_at_10=MetricFraction(8, 10),
            maximum_failure_rate=MetricFraction(1, 20),
            maximum_degraded_rate=MetricFraction(1, 10),
            minimum_category_cases=2,
            minimum_category_leakage_groups=2,
            confidence_level_percent=95,
            bootstrap_replicates=2_000,
            rationale=" ",
        )
