"""Contract tests for reproducible retrieval evaluation values."""

import hashlib
from typing import Callable, TypeVar

import pytest

from inference_lab.evaluation import (
    PUBLISHED_ARTIFACT_NAMES,
    AggregateMetric,
    DuplicateRelationshipSnapshot,
    MetricFraction,
    PublishedArtifact,
    PublishedRetrievalEvaluationManifest,
    RetrievalCaseResult,
    RetrievalComponentIdentity,
    RetrievalEvaluationAggregate,
    RetrievalEvaluationContractError,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRunIdentity,
    RetrievalSystemIdentity,
    SanitisedFailure,
    SanitisedHitReference,
    SanitisedSourceReference,
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
