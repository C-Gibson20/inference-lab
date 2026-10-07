"""EVAL-04 metric aggregation, report repeatability, and logging privacy."""

import hashlib
import json
from dataclasses import replace
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_lab.evaluation.cases import QUESTION_TYPES
from inference_lab.evaluation.heldout_gate import (
    HeldOutChannel,
    HeldOutChannelIdentity,
    HeldOutGatePolicy,
    HeldOutRunContract,
)
from inference_lab.evaluation.heldout_reporting import (
    HeldOutCaseObservation,
    HeldOutChannelCaseResult,
    build_held_out_baseline_report,
    log_held_out_baseline_report,
    render_held_out_baseline_report,
    serialise_held_out_baseline_report,
)
from inference_lab.evaluation.retrieval_contracts import (
    DuplicateRelationshipSnapshot,
    MetricFraction,
    RetrievalComponentIdentity,
    RetrievalEvaluationContractError,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRunIdentity,
    RetrievalSystemIdentity,
)
from inference_lab.evaluation.retrieval_scoring import score_retrieval_case
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    FailureCategory,
    FusionPolicy,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalFailure,
    RetrievalLimit,
)
from inference_lab.retrieval.reranking import RerankingIdentity

from .fixtures import CORPUS_ID, CORPUS_VERSION, coverage_case  # noqa: TID252


def _checksum(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _uuid(value: int) -> str:
    return f"00000000-0000-4000-8000-{value:012x}"


def _system_identity(seed: str) -> RetrievalSystemIdentity:
    return RetrievalSystemIdentity(
        adapter_implementation="fixture-adapter",
        adapter_contract_version="1.0.0",
        query_construction_algorithm="verbatim",
        query_construction_version="1.0.0",
        retrieval_algorithm="fixture-ranking",
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


def _contract(case_ids: tuple[str, ...]) -> HeldOutRunContract:
    systems = {channel: _system_identity(channel.value) for channel in HeldOutChannel}
    fusion_policy = FusionPolicy()
    base_identity = RetrievalEvaluationRunIdentity(
        dataset_id="dataset-eval04",
        dataset_version="1.0.0",
        dataset_schema_version="3.0.0",
        case_file_checksum=_checksum("frozen-cases"),
        corpus_id=CorpusId.from_str(CORPUS_ID),
        corpus_version=CorpusVersionId.from_str(CORPUS_VERSION),
        selected_split="held_out",
        selected_case_ids=case_ids,
        policy=RetrievalEvaluationPolicy(
            passage_cutoffs=(8, 10),
            candidate_depth=10,
        ),
        relationship_snapshot_digest=_checksum("relationship-snapshot"),
        system_identity=systems[HeldOutChannel.FUSED_PRE_RERANK],
    )
    return HeldOutRunContract(
        evaluation_identity=base_identity,
        gate_policy=HeldOutGatePolicy(
            minimum_document_recall_at_10=MetricFraction(1, 1),
            maximum_failure_rate=MetricFraction(0, 1),
            maximum_degraded_rate=MetricFraction(0, 1),
            minimum_category_cases=2,
            minimum_category_leakage_groups=2,
            confidence_level_percent=95,
            bootstrap_replicates=128,
            rationale="Synthetic fixture only; not a production threshold decision.",
        ),
        channel_identities=tuple(
            HeldOutChannelIdentity(
                channel,
                systems[channel],
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
                        evidence_count=8,
                        candidate_limit=100,
                    )
                    if channel is HeldOutChannel.FUSED_RERANKED
                    else None
                ),
            )
            for channel in HeldOutChannel
        ),
        final_evidence_count=8,
    )


def _observation_fixture(
    *,
    fail_metadata_case: str | None = None,
    degrade_dense_case: str | None = None,
) -> tuple[HeldOutCaseObservation, ...]:
    cases = tuple(
        coverage_case(
            index,
            question_type,
            split="held_out",
            abstention=position == 1,
        )
        for question_type in QUESTION_TYPES
        for position, index in enumerate(
            (
                2 * QUESTION_TYPES.index(question_type) + 1,
                2 * QUESTION_TYPES.index(question_type) + 2,
            )
        )
    )
    corpus_id = CorpusId.from_str(CORPUS_ID)
    corpus_version = CorpusVersionId.from_str(CORPUS_VERSION)
    memberships = tuple(
        CorpusDocumentVersionMembership(
            corpus_id,
            DocumentId.from_str(evidence.document_id),
            DocumentVersionId.from_str(cast(str, evidence.document_version_id)),
        )
        for case in cases
        for evidence in case.evidence
    )
    snapshot = DuplicateRelationshipSnapshot(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        memberships=memberships,
    )
    scoring_policy = RetrievalEvaluationPolicy(
        passage_cutoffs=(8, 10),
        candidate_depth=10,
    )
    observations = []
    for ordinal, case in enumerate(cases):
        evidence = case.evidence[0]
        channel_results = []
        for channel_index, channel in enumerate(HeldOutChannel):
            should_hit = not case.abstention.expected and (
                channel
                in {
                    HeldOutChannel.FUSED_PRE_RERANK,
                    HeldOutChannel.FUSED_RERANKED,
                    HeldOutChannel.DENSE_ONLY,
                }
                or (
                    channel is HeldOutChannel.LEXICAL_ONLY
                    and QUESTION_TYPES.index(case.question_type) % 2 == 0
                )
            )
            status = ResultStatus.COMPLETED
            hits = ()
            failures = ()
            if (
                fail_metadata_case == case.case_id
                and channel is HeldOutChannel.METADATA
            ):
                status = ResultStatus.FAILED
                failures = (
                    RetrievalFailure(
                        FailureCategory.CHANNEL_UNAVAILABLE,
                        retryable=False,
                    ),
                )
            elif should_hit:
                hits = (
                    EvidenceHit(
                        candidate=RetrievalCandidate(
                            record_id=RecordId.from_str(
                                _uuid(10_000 + ordinal * 10 + channel_index)
                            ),
                            kind=CandidateKind.ORIGINAL_PASSAGE,
                            content="synthetic private evidence text",
                            sources=(
                                EvidenceSource(
                                    DocumentId.from_str(evidence.document_id),
                                    DocumentVersionId.from_str(
                                        cast(str, evidence.document_version_id)
                                    ),
                                    SourceSpanId.from_str(
                                        cast(str, evidence.source_span_id)
                                    ),
                                ),
                            ),
                        ),
                        contributions=(ChannelScore(RetrievalChannel.LEXICAL, 1),),
                        rank=1,
                    ),
                )
            retrieval_result = EvidenceResult(
                corpus_version=corpus_version,
                requested_limit=RetrievalLimit(
                    8 if channel is HeldOutChannel.FUSED_RERANKED else 10
                ),
                hits=hits,
                status=status,
                failures=failures,
            )
            scored = score_retrieval_case(
                case=case,
                result=retrieval_result,
                corpus_id=corpus_id,
                snapshot=snapshot,
                policy=scoring_policy,
                case_ordinal=ordinal,
            )
            channel_results.append(
                HeldOutChannelCaseResult(
                    channel=channel,
                    result=scored,
                    degraded=(
                        degrade_dense_case == case.case_id
                        and channel is HeldOutChannel.DENSE_ONLY
                    ),
                )
            )
        group_id = _checksum(f"leakage-{case.case_id}")
        observations.append(
            HeldOutCaseObservation(
                case_id=case.case_id,
                question_type=case.question_type,
                expected_abstention=case.abstention.expected,
                leakage_group_id=group_id,
                channel_results=tuple(channel_results),
            )
        )
    return tuple(observations)


def _report(
    *,
    fail_metadata_case: str | None = None,
    degrade_dense_case: str | None = None,
):
    observations = _observation_fixture(
        fail_metadata_case=fail_metadata_case,
        degrade_dense_case=degrade_dense_case,
    )
    return build_held_out_baseline_report(
        _contract(tuple(item.case_id for item in observations)), observations
    )


def test_aggregate_reports_candidate_and_final_recall_at_their_cutoffs() -> None:
    report = _report()
    summaries = {item.channel: item for item in report.channel_summaries}
    fused_candidates = summaries[HeldOutChannel.FUSED_PRE_RERANK]
    reranked = summaries[HeldOutChannel.FUSED_RERANKED]
    comparison = next(
        item
        for item in report.paired_comparisons
        if item.candidate is HeldOutChannel.FUSED_RERANKED
    )

    assert report.outcome.value == "pass"
    assert fused_candidates.document_recall.micro == MetricFraction(6, 6)
    assert fused_candidates.passage_recall.cutoff == 10
    assert reranked.document_recall.cutoff == 8
    assert reranked.passage_recall.cutoff == 8
    assert reranked.actual_item_count_min == 0
    assert reranked.actual_item_count_max == 1
    assert reranked.actual_item_count_total == 6
    assert comparison.cutoff == 8
    assert comparison.paired_case_count == 6
    assert comparison.ties == 6
    assert comparison.non_applicable_case_count == 6


def test_failures_degradation_no_results_and_channel_pairs_are_retained() -> None:
    report = _report(fail_metadata_case="case-001", degrade_dense_case="case-003")
    summaries = {item.channel: item for item in report.channel_summaries}
    metadata = summaries[HeldOutChannel.METADATA]
    dense = summaries[HeldOutChannel.DENSE_ONLY]
    fused = summaries[HeldOutChannel.FUSED_PRE_RERANK]
    lexical_comparison = next(
        item
        for item in report.paired_comparisons
        if item.reference is HeldOutChannel.LEXICAL_ONLY
    )

    assert metadata.failed_case_count == 1
    assert metadata.failure_category_counts == (("channel_unavailable", 1),)
    assert metadata.document_recall.micro == MetricFraction(0, 6)
    assert dense.degraded_case_count == 1
    assert fused.no_result_case_count == 6
    assert lexical_comparison.candidate_wins == 3
    assert lexical_comparison.ties == 3
    assert lexical_comparison.non_applicable_case_count == 6


def test_report_serialisation_rendering_and_log_are_repeatable_and_payload_free() -> (
    None
):
    report = _report()
    repeated = _report()
    json_bytes = serialise_held_out_baseline_report(report)
    markdown = render_held_out_baseline_report(report)

    assert json_bytes == serialise_held_out_baseline_report(repeated)
    assert report.digest == repeated.digest
    assert json_bytes.endswith(b"\n")
    assert not json_bytes.endswith(b"\n\n")
    assert b"What result is recorded for benchmark item" not in json_bytes
    assert "synthetic private evidence text" not in markdown
    payload = json.loads(json_bytes)
    assert payload["gate"]["outcome"] == "pass"
    assert payload["case_count"] == 12

    with capture_logs() as logs:
        log_held_out_baseline_report(report)
    rendered_logs = repr(logs)
    assert logs[0]["event"] == "held_out_evaluation_completed"
    assert logs[0]["gate_outcome"] == "pass"
    assert "What result is recorded for benchmark item" not in rendered_logs
    assert "synthetic private evidence text" not in rendered_logs


def test_report_preflight_rejects_case_order_or_corpus_version_mismatch() -> None:
    observations = _observation_fixture()
    contract = _contract(tuple(item.case_id for item in observations))

    with pytest.raises(RetrievalEvaluationContractError, match="case order"):
        build_held_out_baseline_report(contract, tuple(reversed(observations)))

    bad_channel = replace(
        observations[0].channel_results[0],
        result=replace(
            observations[0].channel_results[0].result,
            corpus_version=CorpusVersionId.from_str(_uuid(999)),
        ),
    )
    bad_observation = replace(
        observations[0],
        channel_results=(bad_channel, *observations[0].channel_results[1:]),
    )
    with pytest.raises(RetrievalEvaluationContractError, match="frozen corpus version"):
        build_held_out_baseline_report(contract, (bad_observation, *observations[1:]))
