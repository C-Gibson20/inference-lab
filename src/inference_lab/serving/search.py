"""Corpus-scoped lexical search orchestration for the local serving facade."""

from __future__ import annotations

from typing import NoReturn, cast

from inference_lab.profiling import FailureMapping, ProfilingContext
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import (
    CandidateKind,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    LexicalError,
    LexicalFailureCategory,
    LexicalGeneration,
    ResultStatus,
    RetrievalChannel,
    execute_lexical_retrieval,
    open_lexical_generation,
    validate_metadata_filter,
)
from inference_lab.retrieval.metadata_filters import InvalidMetadataFilterError
from inference_lab.serving.search_contracts import (
    ServingSearchError,
    ServingSearchFailureCategory,
    ServingSearchHit,
    ServingSearchRequest,
    ServingSearchResult,
    ServingSearchSourceLink,
)
from inference_lab.serving.search_locations import (
    _lexical_reference,
    resolve_source_locations,
)
from inference_lab.settings import Settings
from inference_lab.storage import (
    CorpusSnapshot,
    CorruptStorageError,
    MetadataRepository,
    RepositoryClosedError,
    RepositoryUnavailableError,
    StorageConcurrencyError,
    StorageConflictError,
    StorageError,
)


class ServingSearchRunner:
    """Resolve one corpus version and execute a bounded lexical search."""

    def __init__(self, repository: MetadataRepository, settings: Settings) -> None:
        if not callable(getattr(repository, "get_corpus", None)):
            raise TypeError("repository must provide corpus reads.")
        if not callable(getattr(repository, "open_active_corpus_snapshot", None)):
            raise TypeError("repository must provide active corpus snapshots.")
        if not isinstance(settings, Settings):
            raise TypeError("settings must be Settings.")
        self._repository = repository
        self._settings = settings

    def search(
        self,
        request: ServingSearchRequest,
        *,
        profiling: ProfilingContext | None = None,
    ) -> ServingSearchResult:
        """Execute one search against exactly one resolved corpus snapshot."""
        if not isinstance(request, ServingSearchRequest):
            raise TypeError("request must be a ServingSearchRequest.")
        if profiling is not None and not isinstance(profiling, ProfilingContext):
            raise TypeError("profiling must be a ProfilingContext or None.")
        selected_profiling = (
            profiling if profiling is not None else ProfilingContext.create()
        )
        try:
            self._validate_filter(request)
        except ServingSearchError as error:
            _raise_profiled_failure(request, selected_profiling, error)
        try:
            corpus = self._repository.get_corpus(request.corpus_id)
        except StorageError as error:
            _raise_profiled_failure(
                request,
                selected_profiling,
                _storage_error(error),
            )
        if corpus is None:
            _raise_profiled_failure(
                request,
                selected_profiling,
                ServingSearchError(ServingSearchFailureCategory.CORPUS_NOT_FOUND),
            )

        if request.corpus_version is None:
            return self._search_active(request, selected_profiling)
        try:
            self._validate_historical_version(request.corpus_id, request.corpus_version)
        except ServingSearchError as error:
            _raise_profiled_failure(request, selected_profiling, error)
        except StorageError as error:
            _raise_profiled_failure(
                request,
                selected_profiling,
                _storage_error(error),
            )
        return self._search_historical(request, selected_profiling)

    def _validate_filter(self, request: ServingSearchRequest) -> None:
        """Reject unsupported or oversized filter trees before opening storage."""
        if request.filter is None:
            return
        try:
            validation = validate_metadata_filter(request.filter)
        except InvalidMetadataFilterError:
            raise ServingSearchError(
                ServingSearchFailureCategory.UNSUPPORTED_FILTER
            ) from None
        if validation.registry_version != "1.0.0":
            raise ServingSearchError(ServingSearchFailureCategory.UNSUPPORTED_FILTER)

    def _validate_historical_version(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> None:
        """Require the requested historical version to belong to the corpus."""
        if self._repository.get_corpus_version(corpus_id, corpus_version) is not None:
            return
        for corpus in self._repository.list_corpora():
            if corpus.corpus_id == corpus_id:
                continue
            if (
                self._repository.get_corpus_version(corpus.corpus_id, corpus_version)
                is not None
            ):
                raise ServingSearchError(
                    ServingSearchFailureCategory.FOREIGN_CORPUS_VERSION
                )
        raise ServingSearchError(
            ServingSearchFailureCategory.HISTORICAL_VERSION_NOT_FOUND
        )

    def _search_active(
        self,
        request: ServingSearchRequest,
        profiling: ProfilingContext,
    ) -> ServingSearchResult:
        """Open and pin the active version for the complete request."""
        try:
            with self._repository.open_active_corpus_snapshot(
                request.corpus_id
            ) as snapshot:
                return self._execute_in_snapshot(request, snapshot, profiling)
        except StorageConflictError:
            _raise_profiled_failure(
                request,
                profiling,
                ServingSearchError(ServingSearchFailureCategory.NO_PUBLISHED_VERSION),
            )
        except ServingSearchError:
            raise
        except StorageError as error:
            _raise_profiled_failure(request, profiling, _storage_error(error))
        raise ServingSearchError(ServingSearchFailureCategory.STORAGE_UNAVAILABLE)

    def _search_historical(
        self,
        request: ServingSearchRequest,
        profiling: ProfilingContext,
    ) -> ServingSearchResult:
        """Open one caller-selected owned historical version."""
        assert request.corpus_version is not None
        try:
            with self._repository.open_corpus_snapshot(
                request.corpus_id,
                request.corpus_version,
            ) as snapshot:
                return self._execute_in_snapshot(request, snapshot, profiling)
        except ServingSearchError:
            raise
        except StorageError as error:
            _raise_profiled_failure(request, profiling, _storage_error(error))
        raise ServingSearchError(ServingSearchFailureCategory.STORAGE_UNAVAILABLE)

    def _execute_in_snapshot(
        self,
        request: ServingSearchRequest,
        snapshot: CorpusSnapshot,
        profiling: ProfilingContext,
    ) -> ServingSearchResult:
        """Run RET-02 and validate every returned identity before conversion."""
        selected_version = snapshot.version.corpus_version
        if request.corpus_version is not None and (
            selected_version != request.corpus_version
        ):
            raise ServingSearchError(
                ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
            )
        stage_id = profiling.stage_id("serving", "request")
        with profiling.observed_stage(
            stage="serving",
            stage_id=stage_id,
            document_id=None,
            corpus_id=request.corpus_id.as_str(),
            corpus_version=selected_version.as_str(),
            parent_stage_id=None,
            module="inference_lab.serving",
            failure_mapper=map_serving_search_failure,
        ) as stage:
            retrieval_query = request.lexical_query(selected_version)
            try:
                reference = _lexical_reference(snapshot)
                # One full integrity check belongs to the serving request;
                # the borrowed retrieval stage then measures ranked execution.
                with open_lexical_generation(
                    self._settings.storage.index_dir, reference.generation_id
                ) as generation:
                    evidence = execute_lexical_retrieval(
                        self._repository,
                        self._settings.storage.index_dir,
                        request.corpus_id,
                        retrieval_query,
                        profiling=profiling,
                        parent_stage_id=stage_id,
                        snapshot=snapshot,
                        generation=generation,
                    )
                    result = self._convert_result(
                        request, snapshot, generation, evidence
                    )
            except InvalidMetadataFilterError:
                raise ServingSearchError(
                    ServingSearchFailureCategory.UNSUPPORTED_FILTER
                ) from None
            except LexicalError as error:
                category = (
                    ServingSearchFailureCategory.CORRUPT_STATE
                    if error.category is LexicalFailureCategory.CORRUPT_GENERATION
                    else ServingSearchFailureCategory.RETRIEVAL_FAILED
                )
                raise ServingSearchError(category) from None
            except ServingSearchError:
                raise
            except (TypeError, ValueError):
                raise ServingSearchError(
                    ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
                ) from None
            except StorageError as error:
                raise _storage_error(error) from None
            stage.set_counts(items_processed=1, records_emitted=len(result.hits))
            return result
        raise ServingSearchError(ServingSearchFailureCategory.STORAGE_UNAVAILABLE)

    def _convert_result(
        self,
        request: ServingSearchRequest,
        snapshot: CorpusSnapshot,
        generation: LexicalGeneration,
        evidence: EvidenceResult,
    ) -> ServingSearchResult:
        """Convert a successful RET-01 result into the serving contract."""
        if not isinstance(evidence, EvidenceResult):
            raise ServingSearchError(
                ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
            )
        if evidence.corpus_version != snapshot.version.corpus_version:
            raise ServingSearchError(
                ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
            )
        if evidence.requested_limit.value != request.limit:
            raise ServingSearchError(
                ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
            )
        if evidence.status is not ResultStatus.COMPLETED:
            raise ServingSearchError(ServingSearchFailureCategory.RETRIEVAL_FAILED)
        if evidence.failures:
            raise ServingSearchError(ServingSearchFailureCategory.RETRIEVAL_FAILED)

        hits = tuple(
            self._convert_hit(snapshot, generation, hit) for hit in evidence.hits
        )
        return ServingSearchResult(
            corpus_id=request.corpus_id,
            corpus_version=snapshot.version.corpus_version,
            requested_limit=request.limit,
            hits=hits,
        )

    def _convert_hit(
        self,
        snapshot: CorpusSnapshot,
        generation: LexicalGeneration,
        hit: object,
    ) -> ServingSearchHit:
        """Validate one original-passage hit and its corpus ownership."""
        evidence_hit = cast(EvidenceHit, hit)
        candidate = evidence_hit.candidate
        if candidate.kind is not CandidateKind.ORIGINAL_PASSAGE:
            raise ServingSearchError(
                ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
            )
        lexical_scores = tuple(
            contribution
            for contribution in evidence_hit.contributions
            if contribution.channel is RetrievalChannel.LEXICAL
        )
        if len(lexical_scores) != 1:
            raise ServingSearchError(
                ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
            )
        score = lexical_scores[0]
        if score.raw_score is None or score.rank != evidence_hit.rank:
            raise ServingSearchError(
                ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
            )
        links = tuple(
            self._convert_source_link(snapshot, source) for source in candidate.sources
        )
        source_locations = resolve_source_locations(
            snapshot,
            generation,
            candidate.record_id,
            candidate.content,
            tuple(candidate.sources),
        )
        try:
            return ServingSearchHit(
                record_id=candidate.record_id,
                document_id=links[0].document_id,
                document_version_id=links[0].document_version_id,
                rank=evidence_hit.rank,
                score=score.raw_score,
                score_label=cast(str, score.score_label),
                text=candidate.content,
                source_links=links,
                source_locations=source_locations,
            )
        except (TypeError, ValueError):
            raise ServingSearchError(
                ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
            ) from None

    @staticmethod
    def _convert_source_link(
        snapshot: CorpusSnapshot,
        source: object,
    ) -> ServingSearchSourceLink:
        """Require each evidence source to belong to the pinned snapshot."""
        source_link = cast(EvidenceSource, source)
        document_version = snapshot.get_document_version(
            source_link.document_version_id
        )
        if (
            document_version is None
            or document_version.document_id != source_link.document_id
        ):
            raise ServingSearchError(
                ServingSearchFailureCategory.INVALID_RETRIEVAL_RESULT
            )
        return ServingSearchSourceLink(
            document_id=source_link.document_id,
            document_version_id=source_link.document_version_id,
            source_span_id=source_link.source_span_id,
        )


def _storage_error(error: StorageError) -> ServingSearchError:
    """Map storage failures without retaining storage payloads."""
    if isinstance(error, CorruptStorageError):
        category = ServingSearchFailureCategory.CORRUPT_STATE
    elif isinstance(
        error,
        (RepositoryUnavailableError, RepositoryClosedError, StorageConcurrencyError),
    ):
        category = ServingSearchFailureCategory.STORAGE_UNAVAILABLE
    else:
        category = ServingSearchFailureCategory.STORAGE_UNAVAILABLE
    return ServingSearchError(category)


def _raise_profiled_failure(
    request: ServingSearchRequest,
    profiling: ProfilingContext,
    error: ServingSearchError,
) -> NoReturn:
    """Emit one failed serving stage for work that cannot reach RET-02."""
    with profiling.observed_stage(
        stage="serving",
        stage_id=profiling.stage_id("serving", "request"),
        document_id=None,
        corpus_id=request.corpus_id.as_str(),
        corpus_version=(
            None if request.corpus_version is None else request.corpus_version.as_str()
        ),
        parent_stage_id=None,
        module="inference_lab.serving",
        failure_mapper=map_serving_search_failure,
    ):
        raise error
    raise error


def map_serving_search_failure(error: BaseException) -> FailureMapping:
    """Map serving failures to the shared payload-free profiling vocabulary."""
    if isinstance(error, ServingSearchError):
        if error.category in {
            ServingSearchFailureCategory.INVALID_REQUEST,
            ServingSearchFailureCategory.UNSUPPORTED_FILTER,
            ServingSearchFailureCategory.CORPUS_NOT_FOUND,
            ServingSearchFailureCategory.NO_PUBLISHED_VERSION,
            ServingSearchFailureCategory.HISTORICAL_VERSION_NOT_FOUND,
            ServingSearchFailureCategory.FOREIGN_CORPUS_VERSION,
        }:
            return FailureMapping(
                failure_category="input",
                failure_code=error.category.value,
            )
        if error.category in {
            ServingSearchFailureCategory.CORRUPT_STATE,
            ServingSearchFailureCategory.STORAGE_UNAVAILABLE,
        }:
            return FailureMapping(
                failure_category="dependency",
                failure_code=error.category.value,
            )
        return FailureMapping(
            failure_category="internal",
            failure_code=error.category.value,
        )
    return FailureMapping(
        failure_category="internal",
        failure_code="stage_failed",
    )
