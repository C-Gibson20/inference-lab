"""Contract tests for reproducible retrieval evaluation values."""

import hashlib
from typing import Callable, TypeVar

import pytest

from inference_lab.evaluation import (
    PUBLISHED_ARTIFACT_NAMES,
    AggregateMetric,
    AnswerClaim,
    DuplicateDiagnostics,
    DuplicateRelationshipSnapshot,
    EvidenceReference,
    MetricFraction,
    PublishedArtifact,
    PublishedRetrievalEvaluationManifest,
    RetrievalCaseResult,
    RetrievalComponentIdentity,
    RetrievalEvaluationAggregate,
    RetrievalEvaluationContractError,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRunIdentity,
    RetrievalEvaluationScoringError,
    RetrievalSystemIdentity,
    SanitisedFailure,
    SanitisedHitReference,
    SanitisedSourceReference,
    aggregate_retrieval_case_results,
    build_relevance_targets,
    score_retrieval_case,
)
from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.schemas import AbstentionExpectation, ExpectedAnswer
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
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
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalFailure,
    RetrievalLimit,
)


def checksum(seed: str) -> str:
    """Return a deterministic test checksum."""
    return f"sha256:{hashlib.sha256(seed.encode()).hexdigest()}"


_Identifier = TypeVar("_Identifier")


def identifier(
    value: int, identifier_type: Callable[[str], _Identifier]
) -> _Identifier:
    """Build a fixed canonical UUID for one identifier domain."""
    return identifier_type(f"00000000-0000-0000-0000-{value:012d}")


def base_system() -> RetrievalSystemIdentity:
    """Return one valid path-free retriever identity."""
    component = RetrievalComponentIdentity(
        component="lexical-index",
        implementation="bm25",
        format_version="1.0.0",
        contract_version="1.0.0",
        configuration_digest=checksum("lexical-config"),
        generation_id="generation-001",
        source_manifest_digest=checksum("source-manifest"),
    )
    return RetrievalSystemIdentity(
        adapter_implementation="fake-adapter",
        adapter_contract_version="1.0.0",
        query_construction_algorithm="verbatim",
        query_construction_version="1.0.0",
        retrieval_algorithm="lexical-top-k",
        retrieval_algorithm_version="1.0.0",
        configuration_digest=checksum("adapter-config"),
        components=(component,),
    )


def base_policy() -> RetrievalEvaluationPolicy:
    """Return the explicit fixture metric policy."""
    return RetrievalEvaluationPolicy(passage_cutoffs=(1, 5, 10), candidate_depth=10)


def test_metric_fraction_uses_exact_counts_and_explicit_decimal_rounding() -> None:
    metric = MetricFraction(1, 3)

    assert metric.as_dict() == {"numerator": 1, "denominator": 3}
    assert metric.decimal_value(precision=4, rounding_mode="ROUND_HALF_UP") == "0.3333"

    with pytest.raises(RetrievalEvaluationContractError):
        MetricFraction(2, 1)
    with pytest.raises(TypeError):
        MetricFraction(True, 1)  # type: ignore[arg-type]


def test_policy_requires_explicit_sorted_cutoffs_and_sufficient_depth() -> None:
    with pytest.raises(RetrievalEvaluationContractError):
        RetrievalEvaluationPolicy(passage_cutoffs=(5, 1), candidate_depth=10)
    with pytest.raises(RetrievalEvaluationContractError):
        RetrievalEvaluationPolicy(passage_cutoffs=(1, 5), candidate_depth=5)
    with pytest.raises(RetrievalEvaluationContractError):
        RetrievalEvaluationPolicy(
            passage_cutoffs=(1,),
            candidate_depth=10,
            document_cutoff=5,
        )


def test_relationship_snapshot_is_canonical_and_corpus_scoped() -> None:
    corpus_id = identifier(1, CorpusId)
    corpus_version = identifier(2, CorpusVersionId)
    document_a = identifier(3, DocumentId)
    document_b = identifier(4, DocumentId)
    version_a = identifier(5, DocumentVersionId)
    version_b = identifier(6, DocumentVersionId)
    family_id = identifier(7, DocumentFamilyId)
    memberships = (
        CorpusDocumentVersionMembership(corpus_id, document_b, version_b),
        CorpusDocumentVersionMembership(corpus_id, document_a, version_a),
    )
    duplicate = DuplicateRelationship(
        version_a, version_b, DuplicateClass.BYTE_IDENTICAL
    )
    family = DocumentFamilyMembership(family_id, document_a, version_a)

    first = DuplicateRelationshipSnapshot(
        corpus_id,
        corpus_version,
        memberships,
        (duplicate,),
        (family,),
    )
    second = DuplicateRelationshipSnapshot(
        corpus_id,
        corpus_version,
        tuple(reversed(memberships)),
        (duplicate,),
        (family,),
    )

    assert first.as_dict() == second.as_dict()
    assert first.content_digest == second.content_digest

    foreign_corpus = identifier(8, CorpusId)
    with pytest.raises(RetrievalEvaluationContractError):
        DuplicateRelationshipSnapshot(
            corpus_id,
            corpus_version,
            (
                CorpusDocumentVersionMembership(
                    foreign_corpus,
                    document_a,
                    version_a,
                ),
            ),
        )


def test_sanitised_hit_reference_does_not_retain_candidate_content() -> None:
    source = EvidenceSource(
        identifier(10, DocumentId),
        identifier(11, DocumentVersionId),
        identifier(12, SourceSpanId),
    )
    candidate = RetrievalCandidate(
        identifier(13, RecordId),
        CandidateKind.ORIGINAL_PASSAGE,
        "private candidate sentinel",
        (source,),
    )
    hit = EvidenceHit(
        candidate,
        (ChannelScore(RetrievalChannel.LEXICAL, 1),),
        1,
    )

    safe = SanitisedHitReference.from_hit(hit)

    assert "private candidate sentinel" not in repr(safe)
    assert "private candidate sentinel" not in repr(safe.as_dict())
    assert safe.sources == (
        SanitisedSourceReference(
            source.document_id, source.document_version_id, source.source_span_id
        ),
    )


def test_case_result_preserves_empty_partial_and_failed_semantics() -> None:
    corpus_id = identifier(20, CorpusId)
    corpus_version = identifier(21, CorpusVersionId)
    case = EvaluationCase(
        case_id="case-001",
        question="private question sentinel",
        question_type="lookup",
        expected_answer=ExpectedAnswer(),
        abstention=AbstentionExpectation(
            expected=True,
            reason="evidence_absent",
        ),
        split="development",
    )
    limit = RetrievalLimit(10)
    failure = RetrievalFailure(FailureCategory.CHANNEL_UNAVAILABLE, True)
    empty = EvidenceResult(
        corpus_version,
        limit,
        (),
        ResultStatus.COMPLETED,
        (),
    )
    empty_result = RetrievalCaseResult.from_result(
        case=case,
        case_ordinal=0,
        corpus_id=corpus_id,
        result=empty,
    )
    assert empty_result.is_empty
    assert "private question sentinel" not in repr(empty_result)

    with pytest.raises(RetrievalEvaluationContractError):
        RetrievalCaseResult(
            case_id="case-001",
            case_ordinal=0,
            split="development",
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            requested_limit=10,
            status=ResultStatus.COMPLETED,
            hits=(),
            failures=(
                # The invalid state is intentionally checked at the evaluator boundary.
                # It must not be confused with RET-01's valid empty result.
                SanitisedFailure(failure.category, failure.retryable, failure.channel),
            ),
        )

    source = EvidenceSource(
        identifier(22, DocumentId),
        identifier(23, DocumentVersionId),
        identifier(24, SourceSpanId),
    )
    hit = EvidenceHit(
        RetrievalCandidate(
            identifier(25, RecordId),
            CandidateKind.ORIGINAL_PASSAGE,
            "candidate",
            (source,),
        ),
        (ChannelScore(RetrievalChannel.LEXICAL, 1),),
        1,
    )
    partial = EvidenceResult(
        corpus_version,
        limit,
        (hit,),
        ResultStatus.PARTIAL,
        (failure,),
    )
    partial_result = RetrievalCaseResult.from_result(
        case=case,
        case_ordinal=1,
        corpus_id=corpus_id,
        result=partial,
    )
    assert partial_result.status is ResultStatus.PARTIAL


def test_run_identity_and_published_manifest_are_configuration_sensitive() -> None:
    corpus_id = identifier(30, CorpusId)
    corpus_version = identifier(31, CorpusVersionId)
    base = RetrievalEvaluationRunIdentity(
        dataset_id="dataset-001",
        dataset_version="1.0.0",
        dataset_schema_version="3.0.0",
        case_file_checksum=checksum("cases"),
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        selected_split="development",
        selected_case_ids=("case-001", "case-002"),
        policy=base_policy(),
        relationship_snapshot_digest=checksum("relationships"),
        system_identity=base_system(),
    )
    changed = RetrievalEvaluationRunIdentity(
        dataset_id=base.dataset_id,
        dataset_version=base.dataset_version,
        dataset_schema_version=base.dataset_schema_version,
        case_file_checksum=base.case_file_checksum,
        corpus_id=base.corpus_id,
        corpus_version=base.corpus_version,
        selected_split=base.selected_split,
        selected_case_ids=base.selected_case_ids,
        policy=RetrievalEvaluationPolicy(passage_cutoffs=(1, 5), candidate_depth=10),
        relationship_snapshot_digest=base.relationship_snapshot_digest,
        system_identity=base.system_identity,
    )

    assert base.run_fingerprint != changed.run_fingerprint
    assert "private" not in repr(base)

    manifest = PublishedRetrievalEvaluationManifest(
        run_fingerprint=base.run_fingerprint or checksum("run"),
        result_digest=checksum("result"),
        artefacts=tuple(
            PublishedArtifact(name, index + 1, checksum(name))
            for index, name in enumerate(PUBLISHED_ARTIFACT_NAMES)
        ),
    )
    assert tuple(item.name for item in manifest.artefacts) == PUBLISHED_ARTIFACT_NAMES


def test_aggregate_reconciles_counts_and_metric_applicability() -> None:
    metric = AggregateMetric(
        cutoff=10,
        macro=MetricFraction(1, 2),
        micro=MetricFraction(2, 3),
        applicable_case_count=2,
        non_applicable_case_count=1,
        completed_case_count=2,
        partial_case_count=0,
        failed_case_count=1,
        empty_case_count=1,
        matched_target_count=2,
        relevant_target_count=3,
    )
    aggregate = RetrievalEvaluationAggregate(
        case_count=3,
        document=metric,
        passage=(metric,),
    )
    assert aggregate.as_dict()["case_count"] == 3

    with pytest.raises(RetrievalEvaluationContractError):
        AggregateMetric(
            cutoff=10,
            macro=None,
            micro=None,
            applicable_case_count=1,
            non_applicable_case_count=0,
            completed_case_count=1,
            partial_case_count=0,
            failed_case_count=0,
            empty_case_count=0,
            matched_target_count=0,
            relevant_target_count=0,
        )


def scoring_snapshot() -> DuplicateRelationshipSnapshot:
    """Return a corpus fixture with a duplicate copy and a family."""
    corpus_id = identifier(100, CorpusId)
    corpus_version = identifier(101, CorpusVersionId)
    document_a = identifier(102, DocumentId)
    document_b = identifier(103, DocumentId)
    document_c = identifier(104, DocumentId)
    version_a = identifier(105, DocumentVersionId)
    version_b = identifier(106, DocumentVersionId)
    version_c = identifier(107, DocumentVersionId)
    family_id = identifier(108, DocumentFamilyId)
    return DuplicateRelationshipSnapshot(
        corpus_id,
        corpus_version,
        (
            CorpusDocumentVersionMembership(corpus_id, document_a, version_a),
            CorpusDocumentVersionMembership(corpus_id, document_b, version_b),
            CorpusDocumentVersionMembership(corpus_id, document_c, version_c),
        ),
        (DuplicateRelationship(version_a, version_b, DuplicateClass.BYTE_IDENTICAL),),
        (
            DocumentFamilyMembership(family_id, document_a),
            DocumentFamilyMembership(family_id, document_b),
        ),
    )


def scoring_case(snapshot: DuplicateRelationshipSnapshot) -> EvaluationCase:
    """Return two authoritative spans plus context-only evidence."""
    document_a = identifier(102, DocumentId)
    document_c = identifier(104, DocumentId)
    version_a = identifier(105, DocumentVersionId)
    version_c = identifier(107, DocumentVersionId)
    span_a = identifier(109, SourceSpanId)
    span_c = identifier(110, SourceSpanId)
    evidence = (
        EvidenceReference(
            evidence_id="evidence-a",
            kind="source_span",
            role="authoritative",
            corpus_version=snapshot.corpus_version.as_str(),
            document_id=document_a.as_str(),
            document_version_id=version_a.as_str(),
            source_span_id=span_a.as_str(),
        ),
        EvidenceReference(
            evidence_id="evidence-c",
            kind="source_span",
            role="authoritative",
            corpus_version=snapshot.corpus_version.as_str(),
            document_id=document_c.as_str(),
            document_version_id=version_c.as_str(),
            source_span_id=span_c.as_str(),
        ),
        EvidenceReference(
            evidence_id="evidence-context",
            kind="document",
            role="context",
            corpus_version=snapshot.corpus_version.as_str(),
            document_id=document_a.as_str(),
        ),
    )
    return EvaluationCase(
        case_id="case-101",
        question="private retrieval question",
        question_type="lookup",
        expected_answer=ExpectedAnswer(
            claims=(
                AnswerClaim(
                    key="claim_a",
                    evidence_ids=("evidence-a",),
                    description="private expected answer A",
                ),
                AnswerClaim(
                    key="claim_c",
                    evidence_ids=("evidence-c",),
                    description="private expected answer C",
                ),
            )
        ),
        evidence=evidence,
        abstention=AbstentionExpectation(expected=False, reason="not_applicable"),
        split="development",
    )


def retrieval_hit(
    *,
    record_number: int,
    rank: int,
    kind: CandidateKind,
    sources: tuple[EvidenceSource, ...],
) -> EvidenceHit:
    """Build one ranked hit whose candidate content is intentionally private."""
    return EvidenceHit(
        RetrievalCandidate(
            identifier(record_number, RecordId),
            kind,
            "private candidate content sentinel",
            sources,
        ),
        (ChannelScore(RetrievalChannel.LEXICAL, rank),),
        rank,
    )


def test_relevance_targets_exclude_context_and_non_passage_records() -> None:
    snapshot = scoring_snapshot()
    targets = build_relevance_targets(scoring_case(snapshot), snapshot)

    assert len(targets.documents) == 2
    assert len(targets.passages) == 2


def test_authoritative_record_evidence_creates_document_target_only() -> None:
    snapshot = scoring_snapshot()
    document_c = identifier(104, DocumentId)
    case = EvaluationCase(
        case_id="case-record",
        question="private record question",
        question_type="lookup",
        expected_answer=ExpectedAnswer(
            claims=(
                AnswerClaim(
                    key="claim_record",
                    evidence_ids=("evidence-record",),
                    description="private expected record",
                ),
            )
        ),
        evidence=(
            EvidenceReference(
                evidence_id="evidence-record",
                kind="record",
                role="authoritative",
                corpus_version=snapshot.corpus_version.as_str(),
                document_id=document_c.as_str(),
                record_id=identifier(130, RecordId).as_str(),
            ),
        ),
        abstention=AbstentionExpectation(expected=False, reason="not_applicable"),
        split="development",
    )

    targets = build_relevance_targets(case, snapshot)

    assert len(targets.documents) == 1
    assert targets.documents[0].document_id == document_c
    assert targets.passages == ()


def test_transitive_duplicate_components_collapse_logical_documents() -> None:
    snapshot = scoring_snapshot()
    document_a = identifier(102, DocumentId)
    document_c = identifier(104, DocumentId)
    version_a = identifier(105, DocumentVersionId)
    version_b = identifier(106, DocumentVersionId)
    version_c = identifier(107, DocumentVersionId)
    transitive_snapshot = DuplicateRelationshipSnapshot(
        snapshot.corpus_id,
        snapshot.corpus_version,
        snapshot.memberships,
        (
            DuplicateRelationship(version_b, version_c, DuplicateClass.BYTE_IDENTICAL),
            DuplicateRelationship(version_a, version_b, DuplicateClass.BYTE_IDENTICAL),
        ),
        snapshot.family_memberships,
    )

    targets = build_relevance_targets(
        scoring_case(transitive_snapshot), transitive_snapshot
    )

    assert len(targets.documents) == 1
    assert targets.documents[0].document_id in {document_a, document_c}


def test_one_original_passage_hit_can_match_multiple_labelled_spans() -> None:
    snapshot = scoring_snapshot()
    case = scoring_case(snapshot)
    result = EvidenceResult(
        snapshot.corpus_version,
        RetrievalLimit(10),
        (
            retrieval_hit(
                record_number=131,
                rank=1,
                kind=CandidateKind.ORIGINAL_PASSAGE,
                sources=(
                    EvidenceSource(
                        identifier(102, DocumentId),
                        identifier(105, DocumentVersionId),
                        identifier(109, SourceSpanId),
                    ),
                    EvidenceSource(
                        identifier(104, DocumentId),
                        identifier(107, DocumentVersionId),
                        identifier(110, SourceSpanId),
                    ),
                ),
            ),
        ),
        ResultStatus.COMPLETED,
        (),
    )

    scored = score_retrieval_case(
        case=case,
        result=result,
        corpus_id=snapshot.corpus_id,
        snapshot=snapshot,
        policy=base_policy(),
    )

    assert scored.document_recall is not None
    assert scored.document_recall.value == MetricFraction(2, 2)
    assert scored.document_match_ranks == (1, 2)
    assert tuple(metric.value for metric in scored.passage_recall if metric) == (
        MetricFraction(2, 2),
        MetricFraction(2, 2),
        MetricFraction(2, 2),
    )
    assert tuple(match.ranks for match in scored.passage_match_ranks) == (
        (1, 1),
        (1, 1),
        (1, 1),
    )


def test_family_members_do_not_satisfy_another_document_target() -> None:
    snapshot = scoring_snapshot()
    family_snapshot = DuplicateRelationshipSnapshot(
        snapshot.corpus_id,
        snapshot.corpus_version,
        snapshot.memberships,
        (),
        snapshot.family_memberships,
    )
    result = EvidenceResult(
        family_snapshot.corpus_version,
        RetrievalLimit(10),
        (
            retrieval_hit(
                record_number=132,
                rank=1,
                kind=CandidateKind.ORIGINAL_PASSAGE,
                sources=(
                    EvidenceSource(
                        identifier(103, DocumentId),
                        identifier(106, DocumentVersionId),
                        identifier(133, SourceSpanId),
                    ),
                ),
            ),
            retrieval_hit(
                record_number=134,
                rank=2,
                kind=CandidateKind.ORIGINAL_PASSAGE,
                sources=(
                    EvidenceSource(
                        identifier(104, DocumentId),
                        identifier(107, DocumentVersionId),
                        identifier(110, SourceSpanId),
                    ),
                ),
            ),
        ),
        ResultStatus.COMPLETED,
        (),
    )

    scored = score_retrieval_case(
        case=scoring_case(family_snapshot),
        result=result,
        corpus_id=family_snapshot.corpus_id,
        snapshot=family_snapshot,
        policy=base_policy(),
    )

    assert scored.document_recall is not None
    assert scored.document_recall.value == MetricFraction(1, 2)
    assert scored.document_match_ranks == (None, 2)


def test_scoring_collapses_duplicate_documents_but_requires_exact_passages() -> None:
    snapshot = scoring_snapshot()
    case = scoring_case(snapshot)
    document_a = identifier(102, DocumentId)
    document_b = identifier(103, DocumentId)
    document_c = identifier(104, DocumentId)
    version_a = identifier(105, DocumentVersionId)
    version_b = identifier(106, DocumentVersionId)
    version_c = identifier(107, DocumentVersionId)
    span_a = identifier(109, SourceSpanId)
    span_c = identifier(110, SourceSpanId)
    result = EvidenceResult(
        snapshot.corpus_version,
        RetrievalLimit(5),
        (
            retrieval_hit(
                record_number=111,
                rank=1,
                kind=CandidateKind.ORIGINAL_PASSAGE,
                sources=(
                    EvidenceSource(
                        document_b, version_b, identifier(112, SourceSpanId)
                    ),
                ),
            ),
            retrieval_hit(
                record_number=113,
                rank=2,
                kind=CandidateKind.ORIGINAL_PASSAGE,
                sources=(
                    EvidenceSource(
                        document_b, version_b, identifier(114, SourceSpanId)
                    ),
                ),
            ),
            retrieval_hit(
                record_number=115,
                rank=3,
                kind=CandidateKind.ORIGINAL_PASSAGE,
                sources=(EvidenceSource(document_c, version_c, span_c),),
            ),
            retrieval_hit(
                record_number=116,
                rank=4,
                kind=CandidateKind.ORIGINAL_PASSAGE,
                sources=(EvidenceSource(document_a, version_a, span_a),),
            ),
            retrieval_hit(
                record_number=117,
                rank=5,
                kind=CandidateKind.GENERATED_SUMMARY,
                sources=(
                    EvidenceSource(
                        document_a, version_a, identifier(118, SourceSpanId)
                    ),
                ),
            ),
        ),
        ResultStatus.COMPLETED,
        (),
    )
    policy = RetrievalEvaluationPolicy(passage_cutoffs=(1, 5, 10), candidate_depth=10)

    scored = score_retrieval_case(
        case=case,
        result=result,
        corpus_id=snapshot.corpus_id,
        snapshot=snapshot,
        policy=policy,
    )

    assert scored.document_recall is not None
    assert scored.document_recall.value == MetricFraction(2, 2)
    assert tuple(
        metric.value if metric is not None else None for metric in scored.passage_recall
    ) == (
        MetricFraction(0, 2),
        MetricFraction(2, 2),
        MetricFraction(2, 2),
    )
    assert scored.document_match_ranks == (1, 2)
    assert scored.document_depth_exhausted is True
    assert tuple(match.ranks for match in scored.passage_match_ranks) == (
        (None, None),
        (4, 3),
        (4, 3),
    )
    assert scored.duplicate_diagnostics == DuplicateDiagnostics(
        raw_hit_count=5,
        distinct_record_count=5,
        distinct_document_count=3,
        distinct_duplicate_group_count=2,
        distinct_family_count=1,
        unknown_family_document_count=1,
        exact_document_repeat_excess=2,
        byte_identical_group_repeat_excess=3,
        family_member_repeat_excess=3,
        relevant_non_gold_duplicate_copy_count=1,
    )


def test_scoring_fails_closed_for_foreign_candidate_versions() -> None:
    snapshot = scoring_snapshot()
    case = scoring_case(snapshot)
    foreign_document = identifier(119, DocumentId)
    foreign_version = identifier(120, DocumentVersionId)
    result = EvidenceResult(
        snapshot.corpus_version,
        RetrievalLimit(10),
        (
            retrieval_hit(
                record_number=121,
                rank=1,
                kind=CandidateKind.ORIGINAL_PASSAGE,
                sources=(
                    EvidenceSource(
                        foreign_document,
                        foreign_version,
                        identifier(122, SourceSpanId),
                    ),
                ),
            ),
        ),
        ResultStatus.COMPLETED,
        (),
    )

    with pytest.raises(RetrievalEvaluationScoringError):
        score_retrieval_case(
            case=case,
            result=result,
            corpus_id=snapshot.corpus_id,
            snapshot=snapshot,
            policy=base_policy(),
        )


def test_aggregation_keeps_failed_cases_in_recall_denominators() -> None:
    snapshot = scoring_snapshot()
    case = scoring_case(snapshot)
    policy = RetrievalEvaluationPolicy(passage_cutoffs=(1, 5, 10), candidate_depth=10)
    successful = score_retrieval_case(
        case=case,
        result=EvidenceResult(
            snapshot.corpus_version,
            RetrievalLimit(10),
            (
                retrieval_hit(
                    record_number=123,
                    rank=1,
                    kind=CandidateKind.ORIGINAL_PASSAGE,
                    sources=(
                        EvidenceSource(
                            identifier(102, DocumentId),
                            identifier(105, DocumentVersionId),
                            identifier(109, SourceSpanId),
                        ),
                    ),
                ),
            ),
            ResultStatus.COMPLETED,
            (),
        ),
        corpus_id=snapshot.corpus_id,
        snapshot=snapshot,
        policy=policy,
    )
    failed = score_retrieval_case(
        case=case.model_copy(update={"case_id": "case-102"}),
        result=EvidenceResult(
            snapshot.corpus_version,
            RetrievalLimit(10),
            (),
            ResultStatus.FAILED,
            (RetrievalFailure(FailureCategory.CHANNEL_UNAVAILABLE, True),),
        ),
        corpus_id=snapshot.corpus_id,
        snapshot=snapshot,
        policy=policy,
    )
    aggregate = aggregate_retrieval_case_results((successful, failed), policy)

    assert aggregate.document.macro == MetricFraction(1, 4)
    assert aggregate.document.micro == MetricFraction(1, 4)
    assert aggregate.document.applicable_case_count == 2
    assert aggregate.document.failed_case_count == 1
    assert aggregate.document.empty_case_count == 0
