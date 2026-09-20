"""Pinned-snapshot execution for the lexical retrieval channel."""

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
    LEXICAL_COMPONENT_KIND,
    LexicalError,
    LexicalFailureCategory,
    LexicalQuery,
    lexical_score_label,
)
from inference_lab.retrieval.lexical_store import (
    map_lexical_failure,
    open_lexical_generation,
)
from inference_lab.retrieval.results import (
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalFailure,
)
from inference_lab.retrieval.values import RetrievalChannel
from inference_lab.storage import (
    ComponentGenerationState,
    MetadataRepository,
    StorageError,
)


def _failed_result(
    request: LexicalQuery,
    category: FailureCategory,
) -> EvidenceResult:
    """Return one payload-free failed lexical result."""
    return EvidenceResult.from_query(
        request.query,
        observed_corpus_version=request.query.corpus_version,
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
    """Map one bounded lexical failure to RET-01's channel vocabulary."""
    if error.category in {
        LexicalFailureCategory.GENERATION_UNAVAILABLE,
        LexicalFailureCategory.CORRUPT_GENERATION,
    }:
        return FailureCategory.CHANNEL_UNAVAILABLE
    if error.category is LexicalFailureCategory.INCOMPATIBLE_GENERATION:
        return FailureCategory.INVALID_CHANNEL_RESPONSE
    return FailureCategory.CHANNEL_EXECUTION_FAILURE


def _map_result_failure(result: EvidenceResult) -> FailureMapping | None:
    """Map a returned RET-01 failure to the shared profiling vocabulary."""
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


def execute_lexical_retrieval(
    repository: MetadataRepository,
    index_dir: Path,
    corpus_id: CorpusId,
    request: LexicalQuery,
    *,
    profiling: ProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str = "request",
) -> EvidenceResult:
    """Execute one lexical query and instrument its bounded retrieval stage."""
    if profiling is not None and not isinstance(profiling, ProfilingContext):
        raise TypeError("profiling must be a ProfilingContext or None.")
    if parent_stage_id is not None and not isinstance(parent_stage_id, str):
        raise TypeError("parent_stage_id must be a string or None.")
    if not isinstance(profiling_identity, str) or not profiling_identity:
        raise ValueError("profiling_identity must be a non-empty string.")
    if not callable(getattr(repository, "open_corpus_snapshot", None)):
        raise TypeError("repository must provide corpus snapshots.")
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(request, LexicalQuery):
        raise TypeError("request must be a LexicalQuery.")

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
        corpus_version=request.query.corpus_version.as_str(),
        parent_stage_id=parent_stage_id,
        module="inference_lab.retrieval",
        failure_mapper=map_lexical_failure,
    ) as stage:
        result = _execute_lexical_retrieval(repository, index_dir, corpus_id, request)
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
    request: LexicalQuery,
) -> EvidenceResult:
    """Execute one lexical query against exactly one pinned corpus snapshot."""
    if not callable(getattr(repository, "open_corpus_snapshot", None)):
        raise TypeError("repository must provide corpus snapshots.")
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(request, LexicalQuery):
        raise TypeError("request must be a LexicalQuery.")

    try:
        with repository.open_corpus_snapshot(
            corpus_id,
            request.query.corpus_version,
        ) as snapshot:
            if snapshot.version.corpus_version != request.query.corpus_version:
                return EvidenceResult.from_query(
                    request.query,
                    observed_corpus_version=snapshot.version.corpus_version,
                    hits=(),
                    status=ResultStatus.FAILED,
                    failures=(),
                )
            references = tuple(
                generation
                for generation in snapshot.manifest.component_generations
                if generation.component_kind.as_str() == LEXICAL_COMPONENT_KIND
            )
            if len(references) != 1:
                return _failed_result(request, FailureCategory.INVALID_CHANNEL_RESPONSE)
            reference = references[0]
            if (
                reference.state is not ComponentGenerationState.READY
                or reference.corpus_id != corpus_id
                or reference.corpus_version != request.query.corpus_version
            ):
                return _failed_result(request, FailureCategory.INVALID_CHANNEL_RESPONSE)

            try:
                with open_lexical_generation(
                    index_dir, reference.generation_id
                ) as generation:
                    if (
                        generation.manifest.corpus_id != corpus_id
                        or generation.manifest.corpus_version
                        != request.query.corpus_version
                        or generation.manifest.generation_id != reference.generation_id
                    ):
                        return _failed_result(
                            request,
                            FailureCategory.INVALID_CHANNEL_RESPONSE,
                        )
                    matches = generation.search(request)
                    hits = tuple(
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
                                    lexical_score_label(request.mode),
                                ),
                            ),
                            rank=rank,
                        )
                        for rank, (chunk, score) in enumerate(matches, start=1)
                    )
                    return EvidenceResult.from_query(
                        request.query,
                        observed_corpus_version=snapshot.version.corpus_version,
                        hits=hits,
                        status=ResultStatus.COMPLETED,
                        failures=(),
                    )
            except (InvalidRetrievalContractError, InvalidRetrievalResultError):
                return _failed_result(
                    request,
                    FailureCategory.INVALID_CHANNEL_RESPONSE,
                )
            except LexicalError as error:
                return _failed_result(request, _map_lexical_failure(error))
    except StorageError:
        return _failed_result(request, FailureCategory.CHANNEL_UNAVAILABLE)
    return _failed_result(request, FailureCategory.CHANNEL_EXECUTION_FAILURE)
