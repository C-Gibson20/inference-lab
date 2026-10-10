"""Single-snapshot execution and validation for hybrid retrieval."""

from collections.abc import Hashable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import structlog

from inference_lab.backends.runtime import EmbeddingBackend
from inference_lab.profiling import FailureMapping, ProfilingContext
from inference_lab.provenance import (
    CorpusId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    SourceSpanId,
)
from inference_lab.retrieval.dense_retrieval import execute_dense_retrieval
from inference_lab.retrieval.dense_store import DenseGeneration
from inference_lab.retrieval.errors import (
    InvalidRetrievalContractError,
    InvalidRetrievalResultError,
)
from inference_lab.retrieval.evidence import RetrievalCandidate
from inference_lab.retrieval.fusion import FusionPolicy, fuse_evidence_results
from inference_lab.retrieval.lexical_retrieval import execute_lexical_retrieval
from inference_lab.retrieval.lexical_store import LexicalGeneration
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.results import (
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalFailure,
)
from inference_lab.retrieval.values import RetrievalChannel
from inference_lab.storage import (
    CorpusSnapshot,
    MetadataRepository,
    StorageError,
)

_LOGGER = structlog.get_logger("inference_lab.retrieval")
FUSION_DEDUPLICATION_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class _CandidateSource:
    """A validated source location used only for deterministic deduplication."""

    document_version_id: DocumentVersionId
    source_span_id: SourceSpanId
    start_offset: int
    end_offset: int
    page_number: int | None
    content_origin: str


@dataclass(frozen=True, slots=True)
class _SnapshotEvidence:
    """Bounded identity data resolved from one selected corpus snapshot."""

    sources: Mapping[SourceSpanId, _CandidateSource]
    duplicate_roots: Mapping[DocumentVersionId, DocumentVersionId]
    duplicate_component_sizes: Mapping[DocumentVersionId, int]
    family_ids: frozenset[DocumentFamilyId]


def execute_hybrid_retrieval(
    repository: MetadataRepository,
    index_dir: Path,
    corpus_id: CorpusId,
    query: RetrievalQuery,
    embedder: EmbeddingBackend,
    *,
    policy: FusionPolicy | None = None,
    profiling: ProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str = "request",
    snapshot: CorpusSnapshot | None = None,
    lexical_generation: LexicalGeneration | None = None,
    dense_generation: DenseGeneration | None = None,
) -> EvidenceResult:
    """Execute lexical and dense retrieval against one pinned snapshot."""
    if not callable(getattr(repository, "open_corpus_snapshot", None)):
        raise TypeError("repository must provide the metadata repository contract.")
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(query, RetrievalQuery):
        raise TypeError("query must be a RetrievalQuery.")
    if not callable(getattr(embedder, "embed", None)):
        raise TypeError("embedder must implement the embedding contract.")
    if policy is not None and not isinstance(policy, FusionPolicy):
        raise TypeError("policy must be a FusionPolicy or None.")
    if profiling is not None and not isinstance(profiling, ProfilingContext):
        raise TypeError("profiling must be a ProfilingContext or None.")
    if parent_stage_id is not None and not isinstance(parent_stage_id, str):
        raise TypeError("parent_stage_id must be a string or None.")
    if not isinstance(profiling_identity, str) or not profiling_identity:
        raise ValueError("profiling_identity must be a non-empty string.")
    if snapshot is not None and not _is_hybrid_snapshot(snapshot):
        raise TypeError("snapshot must provide the corpus snapshot contract or None.")

    selected_policy = FusionPolicy() if policy is None else policy
    selected_profiling = profiling or ProfilingContext.create()
    stage_id = selected_profiling.stage_id("retrieval_fusion", profiling_identity)
    result: EvidenceResult | None = None
    channel_counts: dict[RetrievalChannel, int] = {}
    family_count = 0
    with selected_profiling.observed_stage(
        stage="retrieval_fusion",
        stage_id=stage_id,
        document_id=None,
        corpus_id=corpus_id.as_str(),
        corpus_version=query.corpus_version.as_str(),
        parent_stage_id=parent_stage_id,
        module="inference_lab.retrieval",
        failure_mapper=_map_fusion_exception,
    ) as stage:
        try:
            snapshot_context = (
                nullcontext(snapshot)
                if snapshot is not None
                else repository.open_corpus_snapshot(corpus_id, query.corpus_version)
            )
            with snapshot_context as selected_snapshot:
                if selected_snapshot is None:
                    result = _failed_result(
                        query,
                        FailureCategory.CHANNEL_UNAVAILABLE,
                    )
                elif selected_snapshot.corpus_id != corpus_id:
                    result = _failed_result(
                        query,
                        FailureCategory.INVALID_CHANNEL_RESPONSE,
                    )
                elif selected_snapshot.version.corpus_version != query.corpus_version:
                    result = EvidenceResult.corpus_version_mismatch(
                        query, selected_snapshot.version.corpus_version
                    )
                else:
                    result, channel_counts, family_count = _retrieve_and_fuse(
                        repository,
                        index_dir,
                        corpus_id,
                        query,
                        embedder,
                        selected_snapshot,
                        selected_policy,
                        selected_profiling,
                        stage_id,
                        profiling_identity,
                        lexical_generation,
                        dense_generation,
                    )
        except StorageError:
            result = _failed_result(
                query,
                FailureCategory.CHANNEL_UNAVAILABLE,
            )
        except (InvalidRetrievalContractError, InvalidRetrievalResultError):
            result = _failed_result(
                query,
                FailureCategory.INVALID_CHANNEL_RESPONSE,
            )

        assert result is not None
        stage.set_counts(
            items_processed=sum(channel_counts.values()),
            records_emitted=len(result.hits),
        )
        failure = _map_result_failure(result)
        if failure is not None:
            stage.mark_failure(failure)

    assert result is not None
    _log_fusion_outcome(
        result,
        corpus_id=corpus_id,
        policy=selected_policy,
        channel_counts=channel_counts,
        family_count=family_count,
        run_id=selected_profiling.run_id,
        request_id=selected_profiling.request_id,
    )
    return result


def _retrieve_and_fuse(
    repository: MetadataRepository,
    index_dir: Path,
    corpus_id: CorpusId,
    query: RetrievalQuery,
    embedder: EmbeddingBackend,
    snapshot: CorpusSnapshot,
    policy: FusionPolicy,
    profiling: ProfilingContext,
    stage_id: str,
    profiling_identity: str,
    lexical_generation: LexicalGeneration | None,
    dense_generation: DenseGeneration | None,
) -> tuple[EvidenceResult, dict[RetrievalChannel, int], int]:
    """Run both ranked channels, validate provenance, then fuse candidates."""
    channel_results = {
        RetrievalChannel.LEXICAL: execute_lexical_retrieval(
            repository,
            index_dir,
            corpus_id,
            query,
            profiling=profiling,
            parent_stage_id=stage_id,
            profiling_identity=f"{profiling_identity}-lexical",
            snapshot=snapshot,
            generation=lexical_generation,
        ),
        RetrievalChannel.DENSE: execute_dense_retrieval(
            repository,
            index_dir,
            corpus_id,
            query,
            embedder,
            profiling=profiling,
            parent_stage_id=stage_id,
            profiling_identity=f"{profiling_identity}-dense",
            snapshot=snapshot,
            generation=dense_generation,
        ),
    }
    channel_counts: dict[RetrievalChannel, int] = {}
    for channel, result in tuple(channel_results.items()):
        if not _channel_candidates_visible(snapshot, result):
            result = _failed_channel_result(
                query,
                channel,
                FailureCategory.INVALID_CHANNEL_RESPONSE,
            )
            channel_results[channel] = result
        channel_counts[channel] = len(result.hits)

    snapshot_evidence = _resolve_snapshot_evidence(snapshot, channel_results)
    result = fuse_evidence_results(
        query,
        channel_results,
        policy=policy,
        candidate_deduplication_key=lambda candidate: _candidate_key(
            candidate,
            snapshot_evidence,
        ),
    )
    return result, channel_counts, len(snapshot_evidence.family_ids)


def _channel_candidates_visible(
    snapshot: CorpusSnapshot,
    result: EvidenceResult,
) -> bool:
    """Require every channel source link to resolve inside the pinned snapshot."""
    for hit in result.hits:
        for source in hit.candidate.sources:
            document_version = snapshot.get_document_version(source.document_version_id)
            if (
                document_version is None
                or document_version.document_id != source.document_id
            ):
                return False
            resolved = snapshot.resolve_source_span(source.source_span_id)
            if (
                resolved is None
                or resolved.document_id != source.document_id
                or resolved.location.document_version_id != source.document_version_id
                or resolved.location.source_span_id != source.source_span_id
            ):
                return False
    return True


def _resolve_snapshot_evidence(
    snapshot: CorpusSnapshot,
    channel_results: Mapping[RetrievalChannel, EvidenceResult],
) -> _SnapshotEvidence:
    """Resolve exact locations and visible duplicate/family relationships."""
    source_versions: dict[SourceSpanId, DocumentVersionId] = {}
    source_documents: dict[SourceSpanId, DocumentId] = {}
    locations: dict[SourceSpanId, _CandidateSource] = {}
    versions: set[DocumentVersionId] = set()
    family_ids: set[DocumentFamilyId] = set()
    for result in channel_results.values():
        for hit in result.hits:
            for source in hit.candidate.sources:
                source_versions[source.source_span_id] = source.document_version_id
                source_documents[source.source_span_id] = source.document_id
                versions.add(source.document_version_id)

    relationships: set[DuplicateRelationship] = set()
    queried_versions: set[DocumentVersionId] = set()
    pending_versions = list(sorted(versions, key=lambda item: item.as_str()))
    while pending_versions:
        version_id = pending_versions.pop(0)
        if version_id in queried_versions:
            continue
        queried_versions.add(version_id)
        for relationship in snapshot.list_duplicate_relationships(version_id):
            if not isinstance(relationship, DuplicateRelationship):
                raise InvalidRetrievalContractError(
                    "snapshot returned an invalid duplicate relationship."
                )
            if relationship.duplicate_class is not DuplicateClass.BYTE_IDENTICAL:
                continue
            for endpoint in (
                relationship.first_version_id,
                relationship.second_version_id,
            ):
                if snapshot.get_document_version(endpoint) is None:
                    raise InvalidRetrievalContractError(
                        "duplicate relationship endpoint is outside the snapshot."
                    )
                if endpoint not in queried_versions:
                    pending_versions.append(endpoint)
            relationships.add(relationship)
        pending_versions.sort(key=lambda item: item.as_str())

    for span_id, version_id in source_versions.items():
        resolved = snapshot.resolve_source_span(span_id)
        if resolved is None:
            raise InvalidRetrievalContractError(
                "candidate source span is outside the selected snapshot."
            )
        if (
            resolved.document_id != source_documents[span_id]
            or resolved.location.document_version_id != version_id
            or resolved.location.source_span_id != span_id
        ):
            raise InvalidRetrievalContractError(
                "candidate source span ownership is inconsistent."
            )
        locations[span_id] = _CandidateSource(
            document_version_id=version_id,
            source_span_id=span_id,
            start_offset=resolved.location.start_offset,
            end_offset=resolved.location.end_offset,
            page_number=resolved.location.page_number,
            content_origin=resolved.location.content_origin.value,
        )
        for membership in snapshot.list_family_memberships_by_document_version(
            version_id
        ):
            if _validate_family_membership(
                snapshot, membership, source_documents[span_id], version_id
            ):
                family_ids.add(membership.family_id)
        for membership in snapshot.list_family_memberships_by_document(
            source_documents[span_id]
        ):
            if _validate_family_membership(
                snapshot, membership, source_documents[span_id], version_id
            ):
                family_ids.add(membership.family_id)

    roots, component_sizes = _duplicate_components(versions, relationships)
    return _SnapshotEvidence(
        sources=locations,
        duplicate_roots=roots,
        duplicate_component_sizes=component_sizes,
        family_ids=frozenset(family_ids),
    )


def _validate_family_membership(
    snapshot: CorpusSnapshot,
    membership: DocumentFamilyMembership,
    document_id: DocumentId,
    version_id: DocumentVersionId,
) -> bool:
    """Reject a family association whose identity escapes its selected source."""
    if not isinstance(membership, DocumentFamilyMembership):
        raise InvalidRetrievalContractError(
            "snapshot returned an invalid family membership."
        )
    if membership.document_id != document_id:
        raise InvalidRetrievalContractError(
            "family membership does not match its visible source."
        )
    if membership.document_version_id is None:
        return True
    member_version = snapshot.get_document_version(membership.document_version_id)
    if member_version is None or member_version.document_id != membership.document_id:
        raise InvalidRetrievalContractError(
            "family membership does not match a visible document version."
        )
    return membership.document_version_id == version_id


def _duplicate_components(
    versions: set[DocumentVersionId],
    relationships: set[DuplicateRelationship],
) -> tuple[
    dict[DocumentVersionId, DocumentVersionId],
    dict[DocumentVersionId, int],
]:
    """Build deterministic connected components from visible duplicate edges."""
    parents = {version: version for version in versions}

    def find(version: DocumentVersionId) -> DocumentVersionId:
        parent = parents.setdefault(version, version)
        if parent != version:
            parents[version] = find(parent)
        return parents[version]

    for relationship in sorted(relationships, key=lambda item: item.key()):
        first_root = find(relationship.first_version_id)
        second_root = find(relationship.second_version_id)
        if first_root != second_root:
            lower, upper = sorted(
                (first_root, second_root),
                key=lambda item: item.as_str(),
            )
            parents[upper] = lower

    components: dict[DocumentVersionId, list[DocumentVersionId]] = {}
    for version in parents:
        components.setdefault(find(version), []).append(version)
    roots: dict[DocumentVersionId, DocumentVersionId] = {}
    sizes: dict[DocumentVersionId, int] = {}
    for members in components.values():
        root = min(members, key=lambda item: item.as_str())
        for member in members:
            roots[member] = root
            sizes[member] = len(members)
    return roots, sizes


def _candidate_key(
    candidate: RetrievalCandidate,
    evidence: _SnapshotEvidence,
) -> Hashable:
    """Identify exact repeated source spans without collapsing family members."""
    source_keys: list[tuple[object, ...]] = []
    for source in candidate.sources:
        resolved = evidence.sources[source.source_span_id]
        root = evidence.duplicate_roots[resolved.document_version_id]
        if evidence.duplicate_component_sizes[resolved.document_version_id] > 1:
            source_keys.append(
                (
                    "byte_identical_source_range",
                    root.as_str(),
                    resolved.start_offset,
                    resolved.end_offset,
                    (-1 if resolved.page_number is None else resolved.page_number),
                    resolved.content_origin,
                )
            )
        else:
            source_keys.append(("source_span", resolved.source_span_id.as_str()))
    return candidate.kind.value, tuple(sorted(source_keys))


def _is_hybrid_snapshot(snapshot: object) -> bool:
    """Check the narrow snapshot operations required by hybrid fusion."""
    return all(
        callable(getattr(snapshot, name, None))
        for name in (
            "get_document_version",
            "resolve_source_span",
            "list_duplicate_relationships",
            "list_family_memberships_by_document_version",
            "list_family_memberships_by_document",
        )
    ) and all(hasattr(snapshot, name) for name in ("corpus_id", "version", "manifest"))


def _failed_result(
    query: RetrievalQuery,
    category: FailureCategory,
) -> EvidenceResult:
    """Return one payload-free unscoped fusion failure."""
    return EvidenceResult(
        corpus_version=query.corpus_version,
        requested_limit=query.limit,
        hits=(),
        status=ResultStatus.FAILED,
        failures=(RetrievalFailure(category=category, retryable=False),),
    )


def _failed_channel_result(
    query: RetrievalQuery,
    channel: RetrievalChannel,
    category: FailureCategory,
) -> EvidenceResult:
    """Return one payload-free channel failure after snapshot validation."""
    return EvidenceResult(
        corpus_version=query.corpus_version,
        requested_limit=query.limit,
        hits=(),
        status=ResultStatus.FAILED,
        failures=(
            RetrievalFailure(category=category, retryable=False, channel=channel),
        ),
    )


def _map_fusion_exception(error: BaseException) -> FailureMapping:
    """Map a stage exception to bounded profiling diagnostics."""
    del error
    return FailureMapping(
        failure_category="internal",
        failure_code="retrieval_fusion_failed",
    )


def _map_result_failure(result: EvidenceResult) -> FailureMapping | None:
    """Map failed aggregate results without retaining channel payloads."""
    if result.status is ResultStatus.COMPLETED:
        return None
    if result.status is ResultStatus.PARTIAL:
        return None
    return FailureMapping(
        failure_category="dependency",
        failure_code="retrieval_channels_failed",
    )


def _log_fusion_outcome(
    result: EvidenceResult,
    *,
    corpus_id: CorpusId,
    policy: FusionPolicy,
    channel_counts: Mapping[RetrievalChannel, int],
    family_count: int,
    run_id: str,
    request_id: str | None,
) -> None:
    """Emit a bounded, payload-free fusion completion event."""
    log = (
        _LOGGER.info
        if result.status is ResultStatus.COMPLETED
        else _LOGGER.warning
        if result.status is ResultStatus.PARTIAL
        else _LOGGER.error
    )
    log(
        "retrieval_fusion_completed",
        module="inference_lab.retrieval",
        pipeline_stage="retrieval_fusion",
        run_id=run_id,
        request_id=request_id,
        document_id=None,
        corpus_id=corpus_id.as_str(),
        corpus_version=result.corpus_version.as_str(),
        fusion_policy=policy.identity,
        deduplication_policy=FUSION_DEDUPLICATION_VERSION,
        result_status=result.status.value,
        lexical_candidate_count=channel_counts.get(RetrievalChannel.LEXICAL, 0),
        dense_candidate_count=channel_counts.get(RetrievalChannel.DENSE, 0),
        result_count=len(result.hits),
        family_count=family_count,
        failure_categories=tuple(failure.category.value for failure in result.failures),
    )
