"""Local serving facade for resumable baseline corpus ingest."""

from __future__ import annotations

from typing import cast

from inference_lab.ingest import (
    BaselineChunkGenerationManifest,
    BaselineDocumentOutcomeKind,
    BaselineExecutionStatus,
    BaselineFailureCategory,
    BaselineIngestSummary,
    BaselineManifestState,
    BaselinePublicationStatus,
    BaselineStage,
    BaselineStageCounts,
    DiscoveryStatus,
    plan_baseline_membership,
)
from inference_lab.ingest.baseline_profiling import BaselineProfilingContext
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
    NoCorpusChangeOutcome,
    StorageConcurrencyError,
    StorageConflictError,
)


class LocalServingFacade:
    """Compose corpus selection and the first durable local ingest workflow."""

    def __init__(self, repository: MetadataRepository, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be Settings.")
        if not callable(getattr(repository, "transaction", None)):
            raise TypeError("repository must provide metadata transactions.")
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
        if self._repository.get_corpus(request.corpus_id) is None:
            raise ServingIngestError(ServingIngestFailureCategory.CORPUS_NOT_FOUND)
        selected_profiling = (
            profiling if profiling is not None else BaselineProfilingContext.create()
        )
        discovery = discover_sources(self._repository, self._settings, request)
        existing = self._publication.ensure_manifest(request, discovery)
        if existing.state is BaselineManifestState.COMMITTED:
            self._publication.verify_published(
                request,
                BaselinePublicationStatus.COMMITTED,
            )
            return BaselineIngestSummary(
                corpus_id=request.corpus_id,
                proposed_corpus_version=request.proposed_corpus_version,
                expected_corpus_version=request.expected_corpus_version,
                execution_status=BaselineExecutionStatus.COMPLETED,
                publication_status=BaselinePublicationStatus.COMMITTED,
            )
        if existing.state is BaselineManifestState.ABORTED:
            raise ServingIngestError(ServingIngestFailureCategory.INCOMPATIBLE_RESUME)
        if discovery.status is DiscoveryStatus.FAILED and not discovery.sources:
            return self._publication.failed_summary(
                request,
                BaselineFailureCategory.DISCOVERY_FAILED,
            )

        try:
            outcomes = self._documents.execute(
                request,
                discovery,
                selected_profiling,
            )
            parent_memberships = self._publication.parent_memberships(request)
            membership_plan = plan_baseline_membership(
                request.corpus_id,
                parent_memberships,
                outcomes,
                profiling=selected_profiling,
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
                    selected_profiling,
                )
            publication = self._publication.build_and_publish(
                request,
                chunk_manifest,
                selected_profiling,
            )
        except (StorageConflictError, ServingIngestError):
            raise
        except StorageConcurrencyError:
            return self._publication.failed_summary(
                request,
                BaselineFailureCategory.PUBLICATION_CONCURRENCY,
                retryable=True,
            )
        except Exception:
            return self._publication.failed_summary(
                request,
                BaselineFailureCategory.GENERATION_BUILD_FAILED,
            )

        publication_status = (
            BaselinePublicationStatus.NO_CHANGE
            if isinstance(publication, NoCorpusChangeOutcome)
            else BaselinePublicationStatus.COMMITTED
        )
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
