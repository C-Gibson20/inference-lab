"""Pinned-snapshot execution for the lexical retrieval channel."""

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from inference_lab.profiling import FailureMapping, ProfilingContext
from inference_lab.provenance import CorpusId
from inference_lab.retrieval.errors import (
    InvalidRetrievalContractError,
    InvalidRetrievalResultError,
)
from inference_lab.retrieval.evidence import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceSource,
    RetrievalCandidate,
)
from inference_lab.retrieval.lexical import (
    LexicalError,
    LexicalFailureCategory,
    LexicalMatchMode,
    LexicalQuery,
    lexical_score_label,
)
from inference_lab.retrieval.lexical_store import (
    LexicalGeneration,
    LexicalIndexedChunk,
    map_lexical_failure,
    open_lexical_generation,
)
from inference_lab.retrieval.lexical_store_models import LEXICAL_COMPONENT_KIND
from inference_lab.retrieval.metadata_adapter import (
    resolve_lexical_candidate_metadata,
)
from inference_lab.retrieval.metadata_filters import (
    InvalidMetadataFilterError,
    MetadataFilterValidation,
    evaluate_metadata_filter,
    validate_metadata_filter,
)
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
    CorruptStorageError,
    MetadataRepository,
    StorageError,
    generation_matches_reference,
    select_ready_component_generation,
)


@dataclass(slots=True)
class _MetadataFilterObservation:
    """Carry exact filter counts from execution to its observed stage."""

    items_processed: int | None = None
    records_emitted: int | None = None


def _failed_result(
    query: RetrievalQuery,
    category: FailureCategory,
) -> EvidenceResult:
    """Return one payload-free failed lexical result."""
    return EvidenceResult.from_query(
        query,
        observed_corpus_version=query.corpus_version,
        hits=(),
        status=ResultStatus.FAILED,
        failures=(
            RetrievalFailure(
                category=category,
                retryable=False,
                channel=RetrievalChannel.LEXICAL,
            ),
        ),
    )


def _map_lexical_failure(error: LexicalError) -> FailureCategory:
    """Map one bounded lexical failure to the shared channel vocabulary."""
    if error.category in {
        LexicalFailureCategory.GENERATION_UNAVAILABLE,
        LexicalFailureCategory.CORRUPT_GENERATION,
    }:
        return FailureCategory.CHANNEL_UNAVAILABLE
    if error.category is LexicalFailureCategory.INCOMPATIBLE_GENERATION:
        return FailureCategory.INVALID_CHANNEL_RESPONSE
    return FailureCategory.CHANNEL_EXECUTION_FAILURE


def _map_result_failure(result: EvidenceResult) -> FailureMapping | None:
    """Map a returned retrieval failure to the shared profiling vocabulary."""
    if result.status is not ResultStatus.FAILED:
        return None
    if not result.failures:
        return FailureMapping(
            failure_category="internal",
            failure_code="stage_failed",
        )
    category = result.failures[0].category
    if category is FailureCategory.CHANNEL_UNAVAILABLE:
        return FailureMapping(
            failure_category="dependency",
            failure_code="generation_unavailable",
        )
    if category is FailureCategory.INVALID_CHANNEL_RESPONSE:
        return FailureMapping(
            failure_category="dependency",
            failure_code="incompatible_generation",
        )
    return FailureMapping(
        failure_category="internal",
        failure_code="execution_failure",
    )


def _map_metadata_filter_failure(error: BaseException) -> FailureMapping:
    """Map a raised metadata-filter failure without retaining its payload."""
    if isinstance(error, InvalidMetadataFilterError):
        return FailureMapping(
            failure_category="input",
            failure_code="invalid_metadata_filter",
        )
    if isinstance(error, asyncio.CancelledError):
        return FailureMapping(
            failure_category="cancelled",
            failure_code="metadata_filter_cancelled",
            status="cancelled",
        )
    return FailureMapping(
        failure_category="internal",
        failure_code="metadata_filter_failed",
    )


def _map_metadata_result_failure(result: EvidenceResult) -> FailureMapping:
    """Map one returned filtered-retrieval failure to profiling diagnostics."""
    if (
        result.failures
        and result.failures[0].category is FailureCategory.CHANNEL_UNAVAILABLE
    ):
        return FailureMapping(
            failure_category="dependency",
            failure_code="metadata_dependency_unavailable",
        )
    if (
        result.failures
        and result.failures[0].category is FailureCategory.INVALID_CHANNEL_RESPONSE
    ):
        return FailureMapping(
            failure_category="internal",
            failure_code="metadata_state_invalid",
        )
    return FailureMapping(
        failure_category="internal",
        failure_code="metadata_filter_failed",
    )


def execute_lexical_retrieval(
    repository: MetadataRepository,
    index_dir: Path,
    corpus_id: CorpusId,
    request: LexicalQuery | RetrievalQuery,
    *,
    mode: LexicalMatchMode | None = None,
    profiling: ProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str = "request",
    snapshot: CorpusSnapshot | None = None,
    generation: LexicalGeneration | None = None,
) -> EvidenceResult:
    """Execute one lexical query; supplied handles stay owned by the caller."""
    if profiling is not None and not isinstance(profiling, ProfilingContext):
        raise TypeError("profiling must be a ProfilingContext or None.")
    if parent_stage_id is not None and not isinstance(parent_stage_id, str):
        raise TypeError("parent_stage_id must be a string or None.")
    if not isinstance(profiling_identity, str) or not profiling_identity:
        raise ValueError("profiling_identity must be a non-empty string.")
    if snapshot is not None and not _is_snapshot_reader(snapshot):
        raise TypeError("snapshot must provide corpus snapshot reads or be None.")
    if not callable(getattr(repository, "open_corpus_snapshot", None)):
        raise TypeError("repository must provide corpus snapshots.")
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(request, LexicalQuery):
        if not isinstance(request, RetrievalQuery):
            raise TypeError("request must be a LexicalQuery or RetrievalQuery.")
    if mode is not None and not isinstance(mode, LexicalMatchMode):
        raise TypeError("mode must be a LexicalMatchMode or None.")

    query, lexical_request = _normalise_request(request, mode)

    selected_profiling = (
        profiling if profiling is not None else ProfilingContext.create()
    )
    stage_id = selected_profiling.stage_id("lexical_retrieval", profiling_identity)
    result: EvidenceResult | None = None
    with selected_profiling.observed_stage(
        stage="lexical_retrieval",
        stage_id=stage_id,
        document_id=None,
        corpus_id=corpus_id.as_str(),
        corpus_version=query.corpus_version.as_str(),
        parent_stage_id=parent_stage_id,
        module="inference_lab.retrieval",
        failure_mapper=map_lexical_failure,
    ) as stage:
        if query.filter is None:
            result = _execute_lexical_retrieval(
                repository,
                index_dir,
                corpus_id,
                query,
                lexical_request,
                snapshot=snapshot,
                generation=generation,
            )
        else:
            metadata_observation = _MetadataFilterObservation()
            metadata_stage_id = selected_profiling.stage_id(
                "metadata_filter", profiling_identity
            )
            with selected_profiling.observed_stage(
                stage="metadata_filter",
                stage_id=metadata_stage_id,
                document_id=None,
                corpus_id=corpus_id.as_str(),
                corpus_version=query.corpus_version.as_str(),
                parent_stage_id=stage_id,
                module="inference_lab.retrieval",
                failure_mapper=_map_metadata_filter_failure,
            ) as metadata_stage:
                result = _execute_lexical_retrieval(
                    repository,
                    index_dir,
                    corpus_id,
                    query,
                    lexical_request,
                    metadata_observation=metadata_observation,
                    snapshot=snapshot,
                    generation=generation,
                )
                if result.status is ResultStatus.COMPLETED:
                    metadata_stage.set_counts(
                        items_processed=metadata_observation.items_processed,
                        records_emitted=metadata_observation.records_emitted,
                    )
                else:
                    metadata_stage.mark_failure(_map_metadata_result_failure(result))
        assert result is not None
        # SQLite FTS5 does not expose a cheap, reliable count of candidates
        # examined by its scorer; leave items_processed unavailable rather than
        # performing a second payload-bearing scan.
        stage.set_counts(items_processed=None, records_emitted=len(result.hits))
        failure = _map_result_failure(result)
        if failure is not None:
            stage.mark_failure(failure)
    assert result is not None
    return result


def _execute_lexical_retrieval(
    repository: MetadataRepository,
    index_dir: Path,
    corpus_id: CorpusId,
    query: RetrievalQuery,
    lexical_request: LexicalQuery,
    *,
    metadata_observation: _MetadataFilterObservation | None = None,
    snapshot: CorpusSnapshot | None = None,
    generation: LexicalGeneration | None = None,
) -> EvidenceResult:
    """Execute one lexical query against exactly one pinned corpus snapshot."""
    if not callable(getattr(repository, "open_corpus_snapshot", None)):
        raise TypeError("repository must provide corpus snapshots.")
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(query, RetrievalQuery):
        raise TypeError("query must be a RetrievalQuery.")
    if not isinstance(lexical_request, LexicalQuery):
        raise TypeError("lexical_request must be a LexicalQuery.")
    if snapshot is not None and not _is_snapshot_reader(snapshot):
        raise TypeError("snapshot must provide corpus snapshot reads or be None.")

    metadata_validation = (
        None if query.filter is None else validate_metadata_filter(query.filter)
    )

    try:
        snapshot_context = (
            nullcontext(snapshot)
            if snapshot is not None
            else repository.open_corpus_snapshot(
                corpus_id,
                query.corpus_version,
            )
        )
        with snapshot_context as selected_snapshot:
            if selected_snapshot is None:
                return _failed_result(query, FailureCategory.CHANNEL_UNAVAILABLE)
            if selected_snapshot.version.corpus_version != query.corpus_version:
                return EvidenceResult.from_query(
                    query,
                    observed_corpus_version=selected_snapshot.version.corpus_version,
                    hits=(),
                    status=ResultStatus.FAILED,
                    failures=(),
                )
            reference = select_ready_component_generation(
                selected_snapshot.manifest,
                LEXICAL_COMPONENT_KIND,
                corpus_id=corpus_id,
                corpus_version=query.corpus_version,
            )
            if reference is None:
                return _failed_result(query, FailureCategory.INVALID_CHANNEL_RESPONSE)

            try:
                generation_context = (
                    nullcontext(generation)
                    if generation is not None
                    else open_lexical_generation(index_dir, reference.generation_id)
                )
                with generation_context as selected_generation:
                    if not generation_matches_reference(
                        reference,
                        selected_generation.manifest.component_generation(),
                    ):
                        return _failed_result(
                            query,
                            FailureCategory.INVALID_CHANNEL_RESPONSE,
                        )
                    if metadata_validation is None:
                        matches = selected_generation.search(lexical_request)
                    else:
                        if metadata_observation is None:
                            matches = _filter_ranked_matches(
                                selected_generation,
                                lexical_request,
                                query,
                                metadata_validation,
                                selected_snapshot,
                            )
                        else:
                            matches, items_processed = (
                                _filter_ranked_matches_with_count(
                                    selected_generation,
                                    lexical_request,
                                    query,
                                    metadata_validation,
                                    selected_snapshot,
                                )
                            )
                            metadata_observation.items_processed = items_processed
                            metadata_observation.records_emitted = len(matches)
                    score_label = (
                        selected_generation.manifest.configuration.identifier_score_label
                        if lexical_request.mode is LexicalMatchMode.IDENTIFIER
                        else selected_generation.manifest.configuration.score_label
                    )
                    hits = _build_hits(
                        lexical_request, matches, score_label=score_label
                    )
                    return EvidenceResult.from_query(
                        query,
                        observed_corpus_version=selected_snapshot.version.corpus_version,
                        hits=hits,
                        status=ResultStatus.COMPLETED,
                        failures=(),
                    )
            except (InvalidRetrievalContractError, InvalidRetrievalResultError):
                return _failed_result(
                    query,
                    FailureCategory.INVALID_CHANNEL_RESPONSE,
                )
            except LexicalError as error:
                failure_category = _map_lexical_failure(error)
                if (
                    metadata_observation is not None
                    and error.category is LexicalFailureCategory.CORRUPT_GENERATION
                ):
                    failure_category = FailureCategory.CHANNEL_EXECUTION_FAILURE
                return _failed_result(query, failure_category)
    except StorageError as error:
        if metadata_observation is not None and isinstance(error, CorruptStorageError):
            return _failed_result(query, FailureCategory.CHANNEL_EXECUTION_FAILURE)
        return _failed_result(query, FailureCategory.CHANNEL_UNAVAILABLE)
    return _failed_result(query, FailureCategory.CHANNEL_EXECUTION_FAILURE)


def _is_snapshot_reader(snapshot: object) -> bool:
    """Check the narrow borrowed snapshot surface used by this executor."""
    return all(
        callable(getattr(snapshot, method_name, None))
        for method_name in (
            "get_source_artefact",
            "get_parsed_representation",
            "resolve_source_span",
        )
    ) and all(
        hasattr(snapshot, property_name) for property_name in ("version", "manifest")
    )


def _normalise_request(
    request: LexicalQuery | RetrievalQuery,
    mode: LexicalMatchMode | None,
) -> tuple[RetrievalQuery, LexicalQuery]:
    """Keep the public query envelope separate from the lexical request."""
    if isinstance(request, LexicalQuery):
        if mode is not None and mode is not request.mode:
            raise InvalidRetrievalContractError(
                "mode must match the supplied LexicalQuery."
            )
        return request.query, request
    selected_mode = LexicalMatchMode.TERMS if mode is None else mode
    lexical_query = RetrievalQuery(
        request.text,
        request.corpus_version,
        request.limit,
    )
    return request, LexicalQuery(lexical_query, selected_mode)


def _filter_ranked_matches(
    generation: LexicalGeneration,
    lexical_request: LexicalQuery,
    query: RetrievalQuery,
    validation: MetadataFilterValidation,
    snapshot: CorpusSnapshot,
) -> tuple[tuple[LexicalIndexedChunk, int | float], ...]:
    """Retain eligible ranked matches before applying the final result limit."""
    matches, _ = _filter_ranked_matches_with_count(
        generation,
        lexical_request,
        query,
        validation,
        snapshot,
    )
    return matches


def _filter_ranked_matches_with_count(
    generation: LexicalGeneration,
    lexical_request: LexicalQuery,
    query: RetrievalQuery,
    validation: MetadataFilterValidation,
    snapshot: CorpusSnapshot,
) -> tuple[tuple[tuple[LexicalIndexedChunk, int | float], ...], int]:
    """Retain ranked matches and count candidates whose metadata was evaluated."""
    matches: list[tuple[LexicalIndexedChunk, int | float]] = []
    items_processed = 0
    for chunk, score in generation.iter_ranked_matches(lexical_request):
        metadata = resolve_lexical_candidate_metadata(snapshot, chunk)
        items_processed += 1
        if evaluate_metadata_filter(validation, metadata.values):
            matches.append((chunk, score))
            if len(matches) == query.limit.value:
                break
    return tuple(matches), items_processed


def _build_hits(
    request: LexicalQuery,
    matches: tuple[tuple[LexicalIndexedChunk, int | float], ...],
    *,
    score_label: str | None = None,
) -> tuple[EvidenceHit, ...]:
    """Build contiguous lexical hits while preserving candidate provenance."""
    return tuple(
        EvidenceHit(
            candidate=RetrievalCandidate(
                chunk.record_id,
                CandidateKind.ORIGINAL_PASSAGE,
                chunk.content,
                tuple(
                    EvidenceSource(
                        chunk.document_id,
                        chunk.document_version_id,
                        source.source_span_id,
                    )
                    for source in chunk.source_links
                ),
            ),
            contributions=(
                ChannelScore(
                    RetrievalChannel.LEXICAL,
                    rank,
                    score,
                    (
                        lexical_score_label(request.mode)
                        if score_label is None
                        else score_label
                    ),
                ),
            ),
            rank=rank,
        )
        for rank, (chunk, score) in enumerate(matches, start=1)
    )
