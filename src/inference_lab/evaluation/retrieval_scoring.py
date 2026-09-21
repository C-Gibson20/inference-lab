"""Pure duplicate-aware retrieval relevance matching and aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Final

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.retrieval_contracts import (
    AggregateMetric,
    DuplicateDiagnostics,
    DuplicateRelationshipSnapshot,
    MetricFraction,
    RecallMatchRanks,
    RecallMetric,
    RetrievalCaseResult,
    RetrievalEvaluationAggregate,
    RetrievalEvaluationContractError,
    RetrievalEvaluationPolicy,
    SanitisedHitReference,
    SanitisedSourceReference,
)
from inference_lab.provenance import (
    CorpusId,
    DocumentFamilyId,
    DocumentId,
    DocumentVersionId,
    DuplicateClass,
    SourceSpanId,
)
from inference_lab.retrieval import CandidateKind, EvidenceResult, ResultStatus


class RetrievalEvaluationScoringError(RetrievalEvaluationContractError):
    """Raised when relevance inputs cannot be scored safely."""


_DocumentGroupKey = tuple[str, str]
_SourceTriple = tuple[DocumentId, DocumentVersionId, SourceSpanId]


@dataclass(frozen=True, slots=True)
class DocumentRelevanceTarget:
    """One authoritative document-level relevance target."""

    document_id: DocumentId
    document_version_id: DocumentVersionId | None
    group_key: _DocumentGroupKey

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the target's provenance identity."""
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if self.document_version_id is not None and not isinstance(
            self.document_version_id,
            DocumentVersionId,
        ):
            raise TypeError("document_version_id must be a DocumentVersionId or None.")
        if not isinstance(self.group_key, tuple) or len(self.group_key) != 2:
            raise TypeError("group_key must be a two-item tuple.")
        if any(not isinstance(item, str) or not item for item in self.group_key):
            raise RetrievalEvaluationScoringError("group_key must contain safe values.")


@dataclass(frozen=True, slots=True)
class PassageRelevanceTarget:
    """One authoritative exact source-span relevance target."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    source_span_id: SourceSpanId

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the exact source triple."""
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")

    @property
    def key(self) -> _SourceTriple:
        """Return the exact source triple used for matching."""
        return self.document_id, self.document_version_id, self.source_span_id


@dataclass(frozen=True, slots=True)
class RetrievalRelevanceTargets:
    """Deterministically ordered document and passage targets for one case."""

    documents: tuple[DocumentRelevanceTarget, ...]
    passages: tuple[PassageRelevanceTarget, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate target types and uniqueness."""
        if not isinstance(self.documents, tuple):
            raise TypeError("documents must be a tuple.")
        if not isinstance(self.passages, tuple):
            raise TypeError("passages must be a tuple.")
        if any(
            not isinstance(target, DocumentRelevanceTarget) for target in self.documents
        ):
            raise TypeError(
                "documents must contain only DocumentRelevanceTarget values."
            )
        if any(
            not isinstance(target, PassageRelevanceTarget) for target in self.passages
        ):
            raise TypeError("passages must contain only PassageRelevanceTarget values.")
        document_keys = tuple(target.group_key for target in self.documents)
        if len(document_keys) != len(set(document_keys)):
            raise RetrievalEvaluationScoringError("document targets must be unique.")
        passage_keys = tuple(target.key for target in self.passages)
        if len(passage_keys) != len(set(passage_keys)):
            raise RetrievalEvaluationScoringError("passage targets must be unique.")


@dataclass(frozen=True, slots=True)
class _SnapshotIndex:
    """Private lookup index for one validated relationship snapshot."""

    version_to_document: dict[DocumentVersionId, DocumentId]
    version_to_duplicate_group: dict[DocumentVersionId, str]
    version_to_document_group: dict[DocumentVersionId, _DocumentGroupKey]
    document_families: dict[DocumentId, frozenset[DocumentFamilyId]]
    version_families: dict[DocumentVersionId, frozenset[DocumentFamilyId]]


class _UnionFind:
    """Small deterministic union-find for duplicate components."""

    def __init__(self, values: tuple[DocumentVersionId, ...]) -> None:
        self._parent = {value.as_str(): value.as_str() for value in values}

    def find(self, value: str) -> str:
        """Return the canonical component root with path compression."""
        parent = self._parent[value]
        if parent != value:
            parent = self.find(parent)
            self._parent[value] = parent
        return parent

    def union(self, first: str, second: str) -> None:
        """Union two values while retaining the lowest root identity."""
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        lower, higher = sorted((first_root, second_root))
        self._parent[higher] = lower


def _snapshot_index(snapshot: DuplicateRelationshipSnapshot) -> _SnapshotIndex:
    """Build private lookup tables from one validated snapshot."""
    version_to_document = {
        membership.document_version_id: membership.document_id
        for membership in snapshot.memberships
    }
    versions = tuple(version_to_document)
    union_find = _UnionFind(versions)
    for relationship in snapshot.duplicate_relationships:
        if relationship.duplicate_class is not DuplicateClass.BYTE_IDENTICAL:
            raise RetrievalEvaluationScoringError("unsupported duplicate class.")
        union_find.union(
            relationship.first_version_id.as_str(),
            relationship.second_version_id.as_str(),
        )

    component_versions: dict[str, list[DocumentVersionId]] = defaultdict(list)
    for version in versions:
        component_versions[union_find.find(version.as_str())].append(version)
    component_documents = {
        root: {version_to_document[version] for version in members}
        for root, members in component_versions.items()
    }
    version_to_duplicate_group = {
        version: root
        for root, members in component_versions.items()
        for version in members
    }
    version_to_document_group: dict[DocumentVersionId, _DocumentGroupKey] = {}
    for version, document_id in version_to_document.items():
        root = version_to_duplicate_group[version]
        if len(component_documents[root]) > 1:
            version_to_document_group[version] = ("duplicate", root)
        else:
            version_to_document_group[version] = ("document", document_id.as_str())

    document_families: dict[DocumentId, set[DocumentFamilyId]] = defaultdict(set)
    version_families: dict[DocumentVersionId, set[DocumentFamilyId]] = defaultdict(set)
    for membership in snapshot.family_memberships:
        document_families[membership.document_id].add(membership.family_id)
        if membership.document_version_id is not None:
            version_families[membership.document_version_id].add(membership.family_id)

    return _SnapshotIndex(
        version_to_document=version_to_document,
        version_to_duplicate_group=version_to_duplicate_group,
        version_to_document_group=version_to_document_group,
        document_families={
            document_id: frozenset(families)
            for document_id, families in document_families.items()
        },
        version_families={
            version: frozenset(families)
            for version, families in version_families.items()
        },
    )


def _validate_scoring_boundary(
    case: EvaluationCase,
    snapshot: DuplicateRelationshipSnapshot,
) -> None:
    """Ensure all case evidence belongs to the pinned snapshot."""
    if not isinstance(case, EvaluationCase):
        raise TypeError("case must be an EvaluationCase.")
    if not isinstance(snapshot, DuplicateRelationshipSnapshot):
        raise TypeError("snapshot must be a DuplicateRelationshipSnapshot.")
    for evidence in case.evidence:
        if evidence.corpus_version != snapshot.corpus_version.as_str():
            raise RetrievalEvaluationScoringError(
                "case evidence belongs to another corpus version."
            )


def _validate_evidence_membership(
    *,
    document_id: DocumentId,
    document_version_id: DocumentVersionId | None,
    index: _SnapshotIndex,
) -> None:
    """Reject gold evidence that cannot be proven visible in the snapshot."""
    if document_version_id is None:
        if document_id not in set(index.version_to_document.values()):
            raise RetrievalEvaluationScoringError(
                "document evidence is outside the selected corpus version."
            )
        return
    observed_document = index.version_to_document.get(document_version_id)
    if observed_document is None or observed_document != document_id:
        raise RetrievalEvaluationScoringError(
            "versioned evidence is outside the selected corpus version."
        )


def build_relevance_targets(
    case: EvaluationCase,
    snapshot: DuplicateRelationshipSnapshot,
) -> RetrievalRelevanceTargets:
    """Build authoritative document and exact-passage targets for one case."""
    _validate_scoring_boundary(case, snapshot)
    index = _snapshot_index(snapshot)
    documents_by_group: dict[_DocumentGroupKey, DocumentRelevanceTarget] = {}
    passages: list[PassageRelevanceTarget] = []

    for evidence in case.evidence:
        if evidence.role != "authoritative":
            continue
        document_id = DocumentId.from_str(evidence.document_id)
        document_version_id = (
            DocumentVersionId.from_str(evidence.document_version_id)
            if evidence.document_version_id is not None
            else None
        )
        _validate_evidence_membership(
            document_id=document_id,
            document_version_id=document_version_id,
            index=index,
        )
        if document_version_id is None:
            group_key = ("document", document_id.as_str())
        else:
            group_key = index.version_to_document_group[document_version_id]
        documents_by_group.setdefault(
            group_key,
            DocumentRelevanceTarget(document_id, document_version_id, group_key),
        )

        if evidence.kind == "source_span":
            assert document_version_id is not None
            assert evidence.source_span_id is not None
            passages.append(
                PassageRelevanceTarget(
                    document_id,
                    document_version_id,
                    SourceSpanId.from_str(evidence.source_span_id),
                )
            )

    documents = tuple(
        sorted(
            documents_by_group.values(),
            key=lambda target: (
                target.document_id.as_str(),
                target.document_version_id.as_str()
                if target.document_version_id is not None
                else "",
            ),
        )
    )
    passages.sort(
        key=lambda target: (
            target.document_id.as_str(),
            target.document_version_id.as_str(),
            target.source_span_id.as_str(),
        )
    )
    return RetrievalRelevanceTargets(documents, tuple(passages))


def _source_key(source: SanitisedSourceReference) -> _SourceTriple:
    """Extract the exact source triple from a sanitised source reference."""
    return (
        source.document_id,
        source.document_version_id,
        source.source_span_id,
    )


def _document_occurrences(
    hits: tuple[SanitisedHitReference, ...],
    index: _SnapshotIndex,
) -> tuple[
    tuple[_DocumentGroupKey, DocumentId, DocumentVersionId, int],
    ...,
]:
    """Return first occurrences of visible document groups in ranked order."""
    occurrences: list[tuple[_DocumentGroupKey, DocumentId, DocumentVersionId, int]] = []
    seen: set[_DocumentGroupKey] = set()
    for hit in hits:
        for source in hit.sources:
            observed_document = index.version_to_document.get(
                source.document_version_id
            )
            if observed_document is None or observed_document != source.document_id:
                raise RetrievalEvaluationScoringError(
                    "retrieval candidate is outside the selected corpus version."
                )
            group_key = index.version_to_document_group[source.document_version_id]
            if group_key in seen:
                continue
            seen.add(group_key)
            occurrences.append(
                (
                    group_key,
                    source.document_id,
                    source.document_version_id,
                    hit.rank,
                )
            )
    return tuple(occurrences)


def _passage_match_ranks(
    hits: tuple[SanitisedHitReference, ...],
    targets: tuple[PassageRelevanceTarget, ...],
    cutoffs: tuple[int, ...],
) -> tuple[RecallMatchRanks, ...]:
    """Match exact source triples only in original-passage candidates."""
    earliest: dict[_SourceTriple, int] = {}
    target_keys = {target.key for target in targets}
    for hit in hits:
        if hit.kind is not CandidateKind.ORIGINAL_PASSAGE:
            continue
        for source in hit.sources:
            key = _source_key(source)
            if key in target_keys:
                earliest.setdefault(key, hit.rank)
    return tuple(
        RecallMatchRanks(
            cutoff,
            tuple(
                rank if rank is not None and rank <= cutoff else None
                for target in targets
                for rank in (earliest.get(target.key),)
            ),
        )
        for cutoff in cutoffs
    )


def _duplicate_diagnostics(
    hits: tuple[SanitisedHitReference, ...],
    index: _SnapshotIndex,
    targets: RetrievalRelevanceTargets,
    occurrences: tuple[
        tuple[_DocumentGroupKey, DocumentId, DocumentVersionId, int], ...
    ],
) -> DuplicateDiagnostics:
    """Calculate bounded duplicate and family crowding diagnostics."""
    record_ids = {hit.record_id for hit in hits}
    source_records = [source for hit in hits for source in hit.sources]
    document_ids = [source.document_id for source in source_records]
    duplicate_groups = [
        index.version_to_duplicate_group[source.document_version_id]
        for source in source_records
    ]
    family_ids: list[DocumentFamilyId] = []
    known_documents: set[DocumentId] = set()
    unknown_documents: set[DocumentId] = set()
    for source in source_records:
        families = set(index.document_families.get(source.document_id, frozenset()))
        families.update(
            index.version_families.get(source.document_version_id, frozenset())
        )
        if families:
            family_ids.extend(families)
            known_documents.add(source.document_id)
        else:
            unknown_documents.add(source.document_id)

    def repeat_excess(values: Sequence[object]) -> int:
        """Return occurrences beyond one for each group."""
        return sum(max(0, count - 1) for count in Counter(values).values())

    matched_sources = {
        target.group_key: (source_document, source_version)
        for target in targets.documents
        for group_key, source_document, source_version, _rank in occurrences
        if group_key == target.group_key
    }
    non_gold_copy_count = sum(
        1
        for target in targets.documents
        if target.document_version_id is not None
        and target.group_key[0] == "duplicate"
        and target.group_key in matched_sources
        and matched_sources[target.group_key][1] != target.document_version_id
    )
    return DuplicateDiagnostics(
        raw_hit_count=len(hits),
        distinct_record_count=len(record_ids),
        distinct_document_count=len(set(document_ids)),
        distinct_duplicate_group_count=len(set(duplicate_groups)),
        distinct_family_count=len(set(family_ids)),
        unknown_family_document_count=len(unknown_documents - known_documents),
        exact_document_repeat_excess=repeat_excess(document_ids),
        byte_identical_group_repeat_excess=repeat_excess(duplicate_groups),
        family_member_repeat_excess=repeat_excess(family_ids),
        relevant_non_gold_duplicate_copy_count=non_gold_copy_count,
    )


def score_retrieval_case(
    *,
    case: EvaluationCase,
    result: EvidenceResult,
    corpus_id: CorpusId,
    snapshot: DuplicateRelationshipSnapshot,
    policy: RetrievalEvaluationPolicy,
    case_ordinal: int = 0,
) -> RetrievalCaseResult:
    """Score one ordered RET-01 result against authoritative case evidence."""
    if not isinstance(result, EvidenceResult):
        raise TypeError("result must be an EvidenceResult.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if result.corpus_version != snapshot.corpus_version:
        raise RetrievalEvaluationScoringError(
            "retrieval result belongs to another corpus version."
        )
    if not isinstance(policy, RetrievalEvaluationPolicy):
        raise TypeError("policy must be a RetrievalEvaluationPolicy.")

    targets = build_relevance_targets(case, snapshot)
    base_result = RetrievalCaseResult.from_result(
        case=case,
        case_ordinal=case_ordinal,
        corpus_id=corpus_id,
        result=result,
    )
    inspected_hits = base_result.hits[: policy.candidate_depth]
    index = _snapshot_index(snapshot)
    occurrences = _document_occurrences(inspected_hits, index)
    document_match_ranks = tuple(
        next(
            (
                derived_rank
                for derived_rank, occurrence in enumerate(occurrences, start=1)
                if occurrence[0] == target.group_key
            ),
            None,
        )
        for target in targets.documents
    )
    document_recall = (
        RecallMetric(
            10,
            MetricFraction(
                sum(rank is not None for rank in document_match_ranks),
                len(document_match_ranks),
            ),
        )
        if document_match_ranks
        else None
    )
    passage_match_ranks = _passage_match_ranks(
        inspected_hits,
        targets.passages,
        policy.passage_cutoffs,
    )
    passage_recall = tuple(
        RecallMetric(
            match_ranks.cutoff,
            MetricFraction(
                sum(rank is not None for rank in match_ranks.ranks),
                len(match_ranks.ranks),
            ),
        )
        if match_ranks.ranks
        else None
        for match_ranks in passage_match_ranks
    )
    diagnostics = _duplicate_diagnostics(
        inspected_hits,
        index,
        targets,
        occurrences,
    )
    return replace(
        base_result,
        document_recall=document_recall,
        passage_recall=passage_recall,
        document_match_ranks=document_match_ranks,
        passage_match_ranks=passage_match_ranks,
        document_depth_exhausted=(
            result.status is not ResultStatus.FAILED
            and len(occurrences) < policy.document_cutoff
        ),
        duplicate_diagnostics=diagnostics,
    )


def _aggregate_metric(
    metrics: tuple[RecallMetric | None, ...],
    results: tuple[RetrievalCaseResult, ...],
    cutoff: int,
) -> AggregateMetric:
    """Aggregate one cut-off with exact macro and micro fractions."""
    selected = tuple(metric for metric in metrics if metric is not None)
    applicable_count = len(selected)
    relevant_total = sum(metric.value.denominator for metric in selected)
    matched_total = sum(metric.value.numerator for metric in selected)
    macro = None
    if selected:
        mean = sum(
            (
                Fraction(metric.value.numerator, metric.value.denominator)
                for metric in selected
            ),
            Fraction(0),
        ) / len(selected)
        macro = MetricFraction(mean.numerator, mean.denominator)
    micro = MetricFraction(matched_total, relevant_total) if relevant_total else None
    completed_count = sum(result.status is ResultStatus.COMPLETED for result in results)
    partial_count = sum(result.status is ResultStatus.PARTIAL for result in results)
    failed_count = sum(result.status is ResultStatus.FAILED for result in results)
    empty_count = sum(result.is_empty for result in results)
    return AggregateMetric(
        cutoff=cutoff,
        macro=macro,
        micro=micro,
        applicable_case_count=applicable_count,
        non_applicable_case_count=len(results) - applicable_count,
        completed_case_count=completed_count,
        partial_case_count=partial_count,
        failed_case_count=failed_count,
        empty_case_count=empty_count,
        matched_target_count=matched_total,
        relevant_target_count=relevant_total,
    )


def aggregate_retrieval_case_results(
    results: tuple[RetrievalCaseResult, ...],
    policy: RetrievalEvaluationPolicy,
) -> RetrievalEvaluationAggregate:
    """Aggregate scored cases while retaining failures and empty cases."""
    if not isinstance(results, tuple):
        raise TypeError("results must be a tuple.")
    if not results:
        raise RetrievalEvaluationScoringError("results must not be empty.")
    if not isinstance(policy, RetrievalEvaluationPolicy):
        raise TypeError("policy must be a RetrievalEvaluationPolicy.")
    if any(not isinstance(result, RetrievalCaseResult) for result in results):
        raise TypeError("results must contain only RetrievalCaseResult values.")
    case_ids = tuple(result.case_id for result in results)
    if len(case_ids) != len(set(case_ids)):
        raise RetrievalEvaluationScoringError("case IDs must be unique.")
    corpus_pairs = {(result.corpus_id, result.corpus_version) for result in results}
    if len(corpus_pairs) != 1:
        raise RetrievalEvaluationScoringError(
            "aggregate results must use one corpus version."
        )

    document_metrics = tuple(result.document_recall for result in results)
    document = _aggregate_metric(document_metrics, results, 10)
    passage_metrics: list[AggregateMetric] = []
    for index, cutoff in enumerate(policy.passage_cutoffs):
        per_case: list[RecallMetric | None] = []
        for result in results:
            if len(result.passage_recall) != len(policy.passage_cutoffs):
                raise RetrievalEvaluationScoringError(
                    "case passage metrics do not match the evaluation policy."
                )
            metric = result.passage_recall[index]
            if metric is not None and metric.cutoff != cutoff:
                raise RetrievalEvaluationScoringError(
                    "case passage metric cut-off does not match the policy."
                )
            per_case.append(metric)
        passage_metrics.append(_aggregate_metric(tuple(per_case), results, cutoff))

    diagnostics = [result.duplicate_diagnostics for result in results]
    duplicate_diagnostics = None
    if any(diagnostic is not None for diagnostic in diagnostics):
        values = [
            diagnostic or DuplicateDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            for diagnostic in diagnostics
        ]
        duplicate_diagnostics = DuplicateDiagnostics(
            *(
                sum(getattr(value, field) for value in values)
                for field in (
                    "raw_hit_count",
                    "distinct_record_count",
                    "distinct_document_count",
                    "distinct_duplicate_group_count",
                    "distinct_family_count",
                    "unknown_family_document_count",
                    "exact_document_repeat_excess",
                    "byte_identical_group_repeat_excess",
                    "family_member_repeat_excess",
                    "relevant_non_gold_duplicate_copy_count",
                )
            )
        )
    return RetrievalEvaluationAggregate(
        case_count=len(results),
        document=document,
        passage=tuple(passage_metrics),
        duplicate_diagnostics=duplicate_diagnostics,
    )


__all__: Final = [
    "DocumentRelevanceTarget",
    "PassageRelevanceTarget",
    "RetrievalEvaluationScoringError",
    "RetrievalRelevanceTargets",
    "aggregate_retrieval_case_results",
    "build_relevance_targets",
    "score_retrieval_case",
]
