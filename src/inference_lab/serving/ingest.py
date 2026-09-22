"""Local serving facade for resumable baseline corpus ingest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from inference_lab.ingest import (
    BaselineChunkGenerationManifest,
    BaselineDocumentOutcomeKind,
    BaselineExecutionStatus,
    BaselineFailureCategory,
    BaselineIngestSummary,
    BaselineManifestState,
    BaselinePublicationStatus,
    BaselineRequestExecutionError,
    BaselineStage,
    BaselineStageCounts,
    DiscoveryStatus,
    plan_baseline_membership,
)
from inference_lab.ingest.baseline_profiling import (
    BaselineProfilingContext,
    map_baseline_result_failure,
)
from inference_lab.profiling import ProfilingContext
from inference_lab.provenance import CorpusId, CorpusMetadata
from inference_lab.serving.ingest_contracts import (
    ServingIngestError,
    ServingIngestFailureCategory,
    ServingIngestRequest,
    _ServingRepository,
)
from inference_lab.serving.ingest_discovery import discover_sources
from inference_lab.serving.ingest_documents import ServingDocumentRunner
from inference_lab.serving.ingest_helpers import stage_count
from inference_lab.serving.ingest_publication import ServingPublicationCoordinator
from inference_lab.serving.search import ServingSearchRunner
from inference_lab.serving.search_contracts import (
    ServingSearchRequest,
    ServingSearchResult,
)
from inference_lab.settings import Settings
from inference_lab.storage import (
    MetadataRepository,
    StorageConcurrencyError,
    StorageConflictError,
    WorkerRepositoryOpener,
)


@dataclass(frozen=True, slots=True)
class _IngestFailureContext:
    """Keep a workflow boundary's reporting stage and fallback category together."""

    stage: BaselineStage
    fallback_category: BaselineFailureCategory

    def __post_init__(self) -> None:
        if not isinstance(self.stage, BaselineStage):
            raise TypeError("stage must be a BaselineStage.")
        if not isinstance(self.fallback_category, BaselineFailureCategory):
            raise TypeError("fallback_category must be a BaselineFailureCategory.")

    @property
    def contention_category(self) -> BaselineFailureCategory:
        """Classify storage contention by the active workflow stage."""
        if self.stage is BaselineStage.PUBLICATION:
            return BaselineFailureCategory.PUBLICATION_CONCURRENCY
        return BaselineFailureCategory.STORAGE_UNAVAILABLE


_DISCOVERY_FAILURE = _IngestFailureContext(
    BaselineStage.DISCOVERY, BaselineFailureCategory.DISCOVERY_FAILED
)
_MANIFEST_FAILURE = _IngestFailureContext(
    BaselineStage.PUBLICATION, BaselineFailureCategory.PUBLICATION_CONFLICT
)
_PARENT_MEMBERSHIP_FAILURE = _IngestFailureContext(
    BaselineStage.MEMBERSHIP, BaselineFailureCategory.STORAGE_UNAVAILABLE
)
_DOCUMENT_FAILURE = _IngestFailureContext(
    BaselineStage.PARSE, BaselineFailureCategory.PARSE_FAILED
)
_MEMBERSHIP_FAILURE = _IngestFailureContext(
    BaselineStage.MEMBERSHIP, BaselineFailureCategory.MEMBERSHIP_CONFLICT
)
_GENERATION_FAILURE = _IngestFailureContext(
    BaselineStage.PUBLICATION, BaselineFailureCategory.GENERATION_BUILD_FAILED
)


class LocalServingFacade:
    """Compose corpus selection and the first durable local ingest workflow."""

    def __init__(self, repository: MetadataRepository, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be Settings.")
        if not callable(getattr(repository, "transaction", None)):
            raise TypeError("repository must provide metadata transactions.")
        if settings.ingest.max_concurrent_documents > 1 and not isinstance(
            repository, WorkerRepositoryOpener
        ):
            raise TypeError("parallel ingest requires worker repository ownership.")
        self._repository: _ServingRepository = cast(_ServingRepository, repository)
        self._settings = settings
        self._documents = ServingDocumentRunner(self._repository, settings)
        self._publication = ServingPublicationCoordinator(
            self._repository,
            settings,
        )
        self._search = ServingSearchRunner(self._repository, settings)

    def list_corpora(self) -> tuple[CorpusMetadata, ...]:
        """Return the configured deployment's named corpora and opaque IDs."""
        return self._repository.list_corpora()

    def search(
        self,
        request: ServingSearchRequest,
        *,
        profiling: ProfilingContext | None = None,
    ) -> ServingSearchResult:
        """Search one explicitly selected corpus version for original passages."""
        return self._search.search(request, profiling=profiling)

    def create_corpus(
        self,
        display_name: str,
        *,
        corpus_id: CorpusId | None = None,
    ) -> CorpusMetadata:
        """Create one named corpus and return its stable metadata."""
        if not isinstance(display_name, str):
            raise TypeError("display_name must be a string.")
        selected_id = CorpusId.generate() if corpus_id is None else corpus_id
        if not isinstance(selected_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId or None.")
        corpus = CorpusMetadata(selected_id, display_name)
        with self._repository.transaction() as transaction:
            transaction.add_corpus(corpus)
        return corpus

    def ingest(
        self,
        request: ServingIngestRequest,
        *,
        profiling: BaselineProfilingContext | None = None,
    ) -> BaselineIngestSummary:
        """Ingest one local source scope and publish lexical state."""
        if not isinstance(request, ServingIngestRequest):
            raise TypeError("request must be a ServingIngestRequest.")
        if profiling is not None and not isinstance(
            profiling, BaselineProfilingContext
        ):
            raise TypeError("profiling must be a BaselineProfilingContext or None.")
        selected_profiling = (
            profiling if profiling is not None else BaselineProfilingContext.create()
        )
        stage_id = selected_profiling.stage_id("serving", "ingest-request")
        summary: BaselineIngestSummary | None = None
        with selected_profiling.observed_stage(
            stage="serving",
            stage_id=stage_id,
            document_id=None,
            corpus_id=request.corpus_id.as_str(),
            corpus_version=request.proposed_corpus_version.as_str(),
            parent_stage_id=None,
            module="inference_lab.serving",
        ) as stage:
            summary = self._ingest(request, selected_profiling, stage_id)
            stage.set_counts(
                items_processed=len(summary.document_outcomes),
                records_emitted=(
                    0
                    if summary.publication_status
                    is BaselinePublicationStatus.NOT_PUBLISHED
                    else 1
                ),
            )
            if summary.execution_status in {
                BaselineExecutionStatus.FAILED,
                BaselineExecutionStatus.COMPLETED_WITH_FAILURES,
            }:
                failure = (
                    summary.failures[0]
                    if summary.failures
                    else next(
                        outcome.failures[0]
                        for outcome in summary.document_outcomes
                        if outcome.failures
                    )
                )
                stage.mark_failure(
                    map_baseline_result_failure(
                        failure,
                        partial=(
                            summary.execution_status
                            is BaselineExecutionStatus.COMPLETED_WITH_FAILURES
                        ),
                    )
                )
        if summary is None:
            raise RuntimeError("serving profiling stage did not run")
        return summary

    def _ingest(
        self,
        request: ServingIngestRequest,
        profiling: BaselineProfilingContext,
        parent_stage_id: str,
    ) -> BaselineIngestSummary:
        """Run the durable workflow under one observed serving request."""
        if self._repository.get_corpus(request.corpus_id) is None:
            raise ServingIngestError(ServingIngestFailureCategory.CORPUS_NOT_FOUND)
        failure_context = _DISCOVERY_FAILURE
        replay_committed = False
        try:
            discovery = discover_sources(
                self._repository,
                self._settings,
                request,
                profiling=profiling,
                parent_stage_id=parent_stage_id,
            )
            failure_context = _MANIFEST_FAILURE
            existing = self._publication.ensure_manifest(request, discovery)
            if existing.state is BaselineManifestState.COMMITTED:
                replay_committed = True
                publication_status = self._publication.committed_publication_status(
                    request
                )
                self._publication.verify_published(request, publication_status)
                return BaselineIngestSummary(
                    corpus_id=request.corpus_id,
                    proposed_corpus_version=request.proposed_corpus_version,
                    expected_corpus_version=request.expected_corpus_version,
                    execution_status=BaselineExecutionStatus.COMPLETED,
                    publication_status=publication_status,
                )
            if existing.state is BaselineManifestState.ABORTED:
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                )
            if discovery.status is DiscoveryStatus.FAILED and not discovery.sources:
                return self._publication.failed_summary(
                    request,
                    BaselineFailureCategory.DISCOVERY_FAILED,
                    stage=BaselineStage.DISCOVERY,
                )

            failure_context = _PARENT_MEMBERSHIP_FAILURE
            parent_memberships = self._publication.parent_memberships(request)
            failure_context = _DOCUMENT_FAILURE
            outcomes = self._documents.execute(
                request,
                discovery,
                profiling,
                parent_memberships,
                parent_stage_id=parent_stage_id,
            )
            failure_context = _MEMBERSHIP_FAILURE
            membership_plan = plan_baseline_membership(
                request.corpus_id,
                parent_memberships,
                outcomes,
                profiling=profiling,
                parent_stage_id=parent_stage_id,
            )
            entries = self._publication.chunk_manifest_entries(
                request,
                membership_plan,
            )
            chunk_manifest = BaselineChunkGenerationManifest(
                corpus_id=request.corpus_id,
                corpus_version=request.proposed_corpus_version,
                configuration_identity=request.configuration_identity,
                generation_id=f"chunks-{request.proposed_corpus_version.as_str()}",
                entries=entries,
            )
            if existing.state is BaselineManifestState.PREPARING:
                self._publication.prepare_publication(
                    request,
                    membership_plan,
                    profiling,
                    parent_stage_id=parent_stage_id,
                )
            else:
                self._publication.verify_ready_memberships(request, membership_plan)
            failure_context = _GENERATION_FAILURE
            publication = self._publication.build_and_publish(
                request,
                chunk_manifest,
                profiling,
                parent_stage_id=parent_stage_id,
            )
        except (StorageConflictError, ServingIngestError):
            raise
        except BaselineRequestExecutionError as error:
            return BaselineIngestSummary(
                corpus_id=request.corpus_id,
                proposed_corpus_version=request.proposed_corpus_version,
                expected_corpus_version=request.expected_corpus_version,
                execution_status=BaselineExecutionStatus.FAILED,
                publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
                failures=(error.failure,),
            )
        except StorageConcurrencyError:
            if replay_committed:
                raise
            return self._publication.failed_summary(
                request,
                failure_context.contention_category,
                stage=failure_context.stage,
                retryable=True,
            )
        except Exception:
            if replay_committed:
                raise
            return self._publication.failed_summary(
                request,
                failure_context.fallback_category,
                stage=failure_context.stage,
            )

        publication_status = publication
        self._publication.verify_published(request, publication_status)
        execution_status = (
            BaselineExecutionStatus.COMPLETED_WITH_FAILURES
            if any(
                outcome.kind is BaselineDocumentOutcomeKind.FAILED
                for outcome in outcomes
            )
            else BaselineExecutionStatus.COMPLETED
        )
        return BaselineIngestSummary(
            corpus_id=request.corpus_id,
            proposed_corpus_version=request.proposed_corpus_version,
            expected_corpus_version=request.expected_corpus_version,
            execution_status=execution_status,
            publication_status=publication_status,
            document_outcomes=outcomes,
            stage_counts=(
                stage_count(
                    BaselineStage.PARSE,
                    outcomes,
                    lambda value: value.parse_disposition,
                ),
                stage_count(
                    BaselineStage.CHUNK,
                    outcomes,
                    lambda value: value.chunk_disposition,
                ),
                BaselineStageCounts(
                    stage=BaselineStage.MEMBERSHIP,
                    worked=len(membership_plan.changes),
                ),
                BaselineStageCounts(
                    stage=BaselineStage.PUBLICATION,
                    worked=0
                    if publication_status is BaselinePublicationStatus.NO_CHANGE
                    else 1,
                ),
            ),
        )
