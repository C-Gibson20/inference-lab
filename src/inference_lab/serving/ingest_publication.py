"""Manifest and publication orchestration for serving ingest."""

from __future__ import annotations

from inference_lab.ingest import (
    BaselineChunkGenerationEntry,
    BaselineChunkGenerationManifest,
    BaselineExecutionStatus,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineIngestSummary,
    BaselineManifestState,
    BaselinePublicationStatus,
    BaselineStage,
)
from inference_lab.ingest.baseline_checkpoint_mapping import baseline_manifest_to_record
from inference_lab.ingest.baseline_checkpoints import BaselineIngestManifest
from inference_lab.ingest.baseline_membership import (
    BaselineMembershipPlan,
    apply_baseline_membership_plan,
)
from inference_lab.ingest.baseline_profiling import BaselineProfilingContext
from inference_lab.ingest.baseline_publication import (
    BaselinePublicationOutcome,
    publish_baseline_lexical_generations,
)
from inference_lab.ingest.discovery_results import DiscoveryResult
from inference_lab.provenance import CorpusDocumentVersionMembership
from inference_lab.retrieval import (
    LexicalGenerationParticipant,
    build_lexical_generation_for_corpus_update,
)
from inference_lab.serving.ingest_contracts import (
    _BASELINE_REQUIRED_COMPONENT_KINDS,
    ServingIngestError,
    ServingIngestFailureCategory,
    ServingIngestRequest,
    _ServingRepository,
)
from inference_lab.serving.ingest_helpers import (
    digest,
    discovery_result_identity,
    failure,
    request_identity,
)
from inference_lab.settings import Settings
from inference_lab.storage import (
    BaselineManifestRecordState,
    ComponentKind,
    CorpusUpdateState,
)


class ServingPublicationCoordinator:
    """Own durable proposal, generation, and publication operations."""

    def __init__(self, repository: _ServingRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    def ensure_manifest(
        self,
        request: ServingIngestRequest,
        discovery: DiscoveryResult,
    ) -> BaselineIngestManifest:
        """Create or validate the exact durable proposal identity."""
        discovery_identity = digest(
            {
                "request": request_identity(request).as_str(),
                "discovery": discovery_result_identity(discovery).as_str(),
            }
        )
        expected = BaselineIngestManifest(
            corpus_id=request.corpus_id,
            proposed_corpus_version=request.proposed_corpus_version,
            expected_corpus_version=request.expected_corpus_version,
            configuration_identity=request.configuration_identity,
            required_component_kinds=_BASELINE_REQUIRED_COMPONENT_KINDS,
            discovery_summary_identity=discovery_identity,
            state=BaselineManifestState.PREPARING,
        )
        stored = self._repository.get_baseline_manifest(
            request.corpus_id,
            request.proposed_corpus_version,
        )
        if stored is not None:
            if (
                stored.corpus_id != expected.corpus_id
                or stored.expected_corpus_version != expected.expected_corpus_version
                or stored.configuration_identity != expected.configuration_identity
                or stored.required_component_kinds != expected.required_component_kinds
                or stored.discovery_summary_identity
                != expected.discovery_summary_identity
            ):
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                )
            return BaselineIngestManifest(
                corpus_id=stored.corpus_id,
                proposed_corpus_version=stored.proposed_corpus_version,
                expected_corpus_version=stored.expected_corpus_version,
                configuration_identity=stored.configuration_identity,
                required_component_kinds=stored.required_component_kinds,
                discovery_summary_identity=stored.discovery_summary_identity,
                state=BaselineManifestState(stored.state.value),
            )
        outcome = None
        with self._repository.transaction() as transaction:
            transaction.begin_corpus_update(
                request.corpus_id,
                request.proposed_corpus_version,
                expected_corpus_version=request.expected_corpus_version,
                required_component_kinds=_BASELINE_REQUIRED_COMPONENT_KINDS,
            )
            outcome = transaction.begin_baseline_manifest(
                baseline_manifest_to_record(expected)
            )
        assert outcome is not None
        return BaselineIngestManifest(
            corpus_id=outcome.manifest.corpus_id,
            proposed_corpus_version=outcome.manifest.proposed_corpus_version,
            expected_corpus_version=outcome.manifest.expected_corpus_version,
            configuration_identity=outcome.manifest.configuration_identity,
            required_component_kinds=outcome.manifest.required_component_kinds,
            discovery_summary_identity=outcome.manifest.discovery_summary_identity,
            state=BaselineManifestState(outcome.manifest.state.value),
        )

    def parent_memberships(
        self,
        request: ServingIngestRequest,
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        """Read the exact expected parent membership once."""
        if request.expected_corpus_version is None:
            return ()
        memberships: tuple[CorpusDocumentVersionMembership, ...] = ()
        snapshot = self._repository.open_corpus_snapshot(
            request.corpus_id,
            request.expected_corpus_version,
        )
        with snapshot:
            memberships = snapshot.manifest.document_memberships
        return memberships

    def chunk_manifest_entries(
        self,
        request: ServingIngestRequest,
        membership_plan: BaselineMembershipPlan,
    ) -> tuple[BaselineChunkGenerationEntry, ...]:
        """Resolve every final member to its exact durable chunk selection."""
        entries: list[BaselineChunkGenerationEntry] = []
        for membership in membership_plan.final_memberships:
            selection = self._repository.get_chunk_set_selection(
                request.corpus_id,
                membership.document_version_id,
                request.configuration_identity,
            )
            if selection is None:
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                )
            entries.append(
                BaselineChunkGenerationEntry(
                    membership=membership,
                    selection=selection,
                )
            )
        return tuple(entries)

    def prepare_publication(
        self,
        request: ServingIngestRequest,
        membership_plan: BaselineMembershipPlan,
        profiling: BaselineProfilingContext,
    ) -> None:
        """Create the exact storage proposal and seal the ingest manifest."""
        with self._repository.transaction() as transaction:
            transaction.begin_corpus_update(
                request.corpus_id,
                request.proposed_corpus_version,
                expected_corpus_version=request.expected_corpus_version,
                required_component_kinds=_BASELINE_REQUIRED_COMPONENT_KINDS,
            )
            current = transaction.list_corpus_update_document_memberships(
                request.corpus_id,
                request.proposed_corpus_version,
            )
            if current != membership_plan.final_memberships:
                apply_baseline_membership_plan(
                    transaction,
                    request.proposed_corpus_version,
                    membership_plan,
                    profiling=profiling,
                )
            transaction.transition_baseline_manifest(
                request.corpus_id,
                request.proposed_corpus_version,
                expected_state=BaselineManifestRecordState.PREPARING,
                next_state=BaselineManifestRecordState.READY,
            )

    def verify_ready_memberships(
        self,
        request: ServingIngestRequest,
        membership_plan: BaselineMembershipPlan,
    ) -> None:
        """Reject a resume whose sealed ingest plan differs from staged state."""
        update = self._repository.get_corpus_update(
            request.corpus_id,
            request.proposed_corpus_version,
        )
        if update is None or update.state is not CorpusUpdateState.PREPARING:
            raise ServingIngestError(ServingIngestFailureCategory.INCOMPATIBLE_RESUME)
        staged = self._repository.list_corpus_update_document_memberships(
            request.corpus_id,
            request.proposed_corpus_version,
        )
        if staged != membership_plan.final_memberships:
            raise ServingIngestError(ServingIngestFailureCategory.INCOMPATIBLE_RESUME)

    def committed_publication_status(
        self,
        request: ServingIngestRequest,
    ) -> BaselinePublicationStatus:
        """Distinguish a committed version from an unchanged terminal proposal."""
        update = self._repository.get_corpus_update(
            request.corpus_id,
            request.proposed_corpus_version,
        )
        if update is None:
            raise ServingIngestError(ServingIngestFailureCategory.INCOMPATIBLE_RESUME)
        if update.state is CorpusUpdateState.COMMITTED:
            return BaselinePublicationStatus.COMMITTED
        if (
            update.state is CorpusUpdateState.ABORTED
            and update.manifest_digest is not None
            and request.expected_corpus_version is not None
        ):
            return BaselinePublicationStatus.NO_CHANGE
        raise ServingIngestError(ServingIngestFailureCategory.INCOMPATIBLE_RESUME)

    def build_and_publish(
        self,
        request: ServingIngestRequest,
        chunk_manifest: BaselineChunkGenerationManifest,
        profiling: BaselineProfilingContext,
    ) -> BaselinePublicationOutcome:
        """Build immutable lexical data, then atomically publish both components."""
        build = build_lexical_generation_for_corpus_update(
            self._settings.storage.index_dir,
            chunk_manifest,
            self._repository,
            expected_corpus_version=request.expected_corpus_version,
            configuration=request.lexical_configuration,
            profiling=profiling,
        )
        participant = LexicalGenerationParticipant(
            self._settings.storage.index_dir,
            build,
            chunk_manifest,
        )
        result: BaselinePublicationOutcome | None = None
        with self._repository.transaction() as transaction:
            transaction.begin_corpus_update(
                request.corpus_id,
                request.proposed_corpus_version,
                expected_corpus_version=request.expected_corpus_version,
                required_component_kinds=_BASELINE_REQUIRED_COMPONENT_KINDS,
            )
            result = publish_baseline_lexical_generations(
                transaction,
                self._repository,
                chunk_manifest,
                participant,
                profiling=profiling,
            )
            transaction.transition_baseline_manifest(
                request.corpus_id,
                request.proposed_corpus_version,
                expected_state=BaselineManifestRecordState.READY,
                next_state=BaselineManifestRecordState.COMMITTED,
            )
        if result is None:
            raise RuntimeError("baseline publication returned no outcome")
        return result

    def verify_published(
        self,
        request: ServingIngestRequest,
        publication_status: BaselinePublicationStatus,
    ) -> None:
        """Verify the fresh active snapshot after publication or no-change."""
        with self._repository.open_active_corpus_snapshot(
            request.corpus_id
        ) as snapshot:
            expected_version = (
                request.expected_corpus_version
                if publication_status is BaselinePublicationStatus.NO_CHANGE
                else request.proposed_corpus_version
            )
            if snapshot.version.corpus_version != expected_version:
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                )
            kinds = tuple(
                component.component_kind
                for component in snapshot.manifest.component_generations
            )
            if tuple(sorted(kinds, key=ComponentKind.as_str)) != tuple(
                sorted(_BASELINE_REQUIRED_COMPONENT_KINDS, key=ComponentKind.as_str)
            ):
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                )

    def failed_summary(
        self,
        request: ServingIngestRequest,
        category: BaselineFailureCategory,
        *,
        retryable: bool = False,
    ) -> BaselineIngestSummary:
        """Return a stable failed result without copying exception details."""
        failure_stage = (
            BaselineStage.PUBLICATION
            if category
            in {
                BaselineFailureCategory.GENERATION_BUILD_FAILED,
                BaselineFailureCategory.PUBLICATION_CONCURRENCY,
            }
            else BaselineStage.DISCOVERY
        )
        return BaselineIngestSummary(
            corpus_id=request.corpus_id,
            proposed_corpus_version=request.proposed_corpus_version,
            expected_corpus_version=request.expected_corpus_version,
            execution_status=BaselineExecutionStatus.FAILED,
            publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
            failures=(
                failure(
                    stage=failure_stage,
                    category=category,
                    scope=BaselineFailureScope.REQUEST,
                    severity=BaselineFailureSeverity.FATAL,
                    retryable=retryable,
                ),
            ),
        )
