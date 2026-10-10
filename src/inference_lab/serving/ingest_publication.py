"""Manifest and publication orchestration for serving ingest."""

from __future__ import annotations

from dataclasses import replace

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
    publish_baseline_generations,
)
from inference_lab.ingest.discovery_results import DiscoveryResult
from inference_lab.provenance import CorpusDocumentVersionMembership
from inference_lab.retrieval import (
    DenseGenerationParticipant,
    LexicalGenerationParticipant,
    build_dense_generation_for_corpus_update,
    build_lexical_generation_for_corpus_update,
    derive_lexical_generation_id,
    open_lexical_generation,
)
from inference_lab.retrieval.lexical import (
    LEXICAL_COMPONENT_KIND,
    LEXICAL_CONTRACT_VERSION,
    LEXICAL_FORMAT_VERSION,
    LEXICAL_IMPLEMENTATION,
)
from inference_lab.serving.ingest_contracts import (
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
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusUpdateState,
    CorpusVersionManifest,
    NoCorpusChangeOutcome,
    StorageConcurrencyError,
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
            required_component_kinds=request.required_component_kinds,
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
                required_component_kinds=request.required_component_kinds,
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
        *,
        parent_stage_id: str | None = None,
    ) -> None:
        """Create the exact storage proposal and seal the ingest manifest."""
        with self._repository.transaction() as transaction:
            transaction.begin_corpus_update(
                request.corpus_id,
                request.proposed_corpus_version,
                expected_corpus_version=request.expected_corpus_version,
                required_component_kinds=request.required_component_kinds,
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
                    parent_stage_id=parent_stage_id,
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
        """Distinguish a committed version from either unchanged terminal path.

        A sealed storage no-change has a manifest digest; the pre-build fast
        path aborts before staging component generations and has none. Both
        atomically commit the baseline ingest manifest for replay.
        """
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
            and request.expected_corpus_version is not None
        ):
            return BaselinePublicationStatus.NO_CHANGE
        raise ServingIngestError(ServingIngestFailureCategory.INCOMPATIBLE_RESUME)

    def build_and_publish(
        self,
        request: ServingIngestRequest,
        chunk_manifest: BaselineChunkGenerationManifest,
        profiling: BaselineProfilingContext,
        *,
        parent_stage_id: str | None = None,
    ) -> BaselinePublicationStatus:
        """Build immutable lexical data, then atomically publish both components."""
        if request.dense_embedder is None and self._is_unchanged_parent(
            request, chunk_manifest
        ):
            # No generation is required: the committed parent already represents
            # this exact membership, chunk selection, and lexical configuration.
            with self._repository.transaction() as transaction:
                update = transaction.begin_corpus_update(
                    request.corpus_id,
                    request.proposed_corpus_version,
                    expected_corpus_version=request.expected_corpus_version,
                    required_component_kinds=request.required_component_kinds,
                ).update
                active = transaction.get_active_corpus_version(request.corpus_id)
                staged = transaction.list_corpus_update_document_memberships(
                    request.corpus_id, request.proposed_corpus_version
                )
                if (
                    active is None
                    or active.corpus_version != request.expected_corpus_version
                    or update.state is not CorpusUpdateState.PREPARING
                    or staged
                    != tuple(entry.membership for entry in chunk_manifest.entries)
                ):
                    raise StorageConcurrencyError()
                transaction.abort_corpus_update(
                    request.corpus_id, request.proposed_corpus_version
                )
                transaction.transition_baseline_manifest(
                    request.corpus_id,
                    request.proposed_corpus_version,
                    expected_state=BaselineManifestRecordState.READY,
                    next_state=BaselineManifestRecordState.COMMITTED,
                )
            return BaselinePublicationStatus.NO_CHANGE
        build = build_lexical_generation_for_corpus_update(
            self._settings.storage.index_dir,
            chunk_manifest,
            self._repository,
            expected_corpus_version=request.expected_corpus_version,
            configuration=request.lexical_configuration,
            profiling=profiling,
            parent_stage_id=parent_stage_id,
        )
        participant = LexicalGenerationParticipant(
            self._settings.storage.index_dir,
            build,
            chunk_manifest,
        )
        participants = [chunk_manifest, participant]
        if (
            request.dense_embedder is not None
            and request.dense_compatibility is not None
        ):
            dense_build = build_dense_generation_for_corpus_update(
                self._settings.storage.index_dir,
                chunk_manifest,
                request.dense_embedder,
                self._repository,
                expected_corpus_version=request.expected_corpus_version,
                metadata_db_path=self._settings.storage.metadata_db_path,
                source_dir=self._settings.storage.source_dir,
                compatibility=request.dense_compatibility,
            )
            participants.append(
                DenseGenerationParticipant(
                    self._settings.storage.index_dir,
                    dense_build,
                    chunk_manifest,
                )
            )
        result: BaselinePublicationOutcome | None = None
        with self._repository.transaction() as transaction:
            transaction.begin_corpus_update(
                request.corpus_id,
                request.proposed_corpus_version,
                expected_corpus_version=request.expected_corpus_version,
                required_component_kinds=request.required_component_kinds,
            )
            result = publish_baseline_generations(
                transaction,
                self._repository,
                tuple(participants),
                profiling=profiling,
                parent_stage_id=parent_stage_id,
            )
            transaction.transition_baseline_manifest(
                request.corpus_id,
                request.proposed_corpus_version,
                expected_state=BaselineManifestRecordState.READY,
                next_state=BaselineManifestRecordState.COMMITTED,
            )
        if result is None:
            raise RuntimeError("baseline publication returned no outcome")
        return (
            BaselinePublicationStatus.NO_CHANGE
            if isinstance(result, NoCorpusChangeOutcome)
            else BaselinePublicationStatus.COMMITTED
        )

    def _is_unchanged_parent(
        self,
        request: ServingIngestRequest,
        chunk_manifest: BaselineChunkGenerationManifest,
    ) -> bool:
        """Check exact parent semantics without creating a candidate generation."""
        if request.expected_corpus_version is None:
            return False
        with self._repository.open_active_corpus_snapshot(
            request.corpus_id
        ) as snapshot:
            if snapshot.version.corpus_version != request.expected_corpus_version:
                raise StorageConcurrencyError()
            parent = snapshot.manifest
            if not parent.has_required_components(request.required_component_kinds):
                return False
            proposed_lexical = ComponentGenerationReference(
                component_kind=LEXICAL_COMPONENT_KIND,
                generation_id=derive_lexical_generation_id(
                    request.corpus_id,
                    request.proposed_corpus_version,
                    request.lexical_configuration,
                    chunk_manifest.source_manifest_digest,
                ),
                implementation=LEXICAL_IMPLEMENTATION,
                format_version=LEXICAL_FORMAT_VERSION,
                contract_version=LEXICAL_CONTRACT_VERSION,
                configuration_digest=(
                    request.lexical_configuration.configuration_identity.digest
                ),
                source_manifest_digest=chunk_manifest.source_manifest_digest,
                corpus_id=request.corpus_id,
                corpus_version=request.proposed_corpus_version,
                state=ComponentGenerationState.READY,
            )
            proposed = CorpusVersionManifest(
                corpus_id=request.corpus_id,
                corpus_version=request.proposed_corpus_version,
                document_memberships=tuple(
                    entry.membership for entry in chunk_manifest.entries
                ),
                component_generations=(
                    replace(
                        chunk_manifest.component_generation(),
                        state=ComponentGenerationState.READY,
                    ),
                    proposed_lexical,
                ),
            )
            if not parent.has_same_semantic_state(proposed):
                return False
            lexical = next(
                ref
                for ref in parent.component_generations
                if ref.component_kind == proposed_lexical.component_kind
            )
            with open_lexical_generation(
                self._settings.storage.index_dir, lexical.generation_id
            ) as generation:
                manifest = generation.manifest
                if (
                    manifest.corpus_id != request.corpus_id
                    or manifest.corpus_version != request.expected_corpus_version
                    or manifest.configuration_digest != lexical.configuration_digest
                    or manifest.source_manifest_digest != lexical.source_manifest_digest
                    or manifest.contract_version != lexical.contract_version
                    or manifest.format_version != lexical.format_version
                ):
                    return False
        return True

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
                sorted(request.required_component_kinds, key=ComponentKind.as_str)
            ):
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                )

    def failed_summary(
        self,
        request: ServingIngestRequest,
        category: BaselineFailureCategory,
        *,
        stage: BaselineStage,
        retryable: bool = False,
    ) -> BaselineIngestSummary:
        """Return a stable failed result without copying exception details."""
        return BaselineIngestSummary(
            corpus_id=request.corpus_id,
            proposed_corpus_version=request.proposed_corpus_version,
            expected_corpus_version=request.expected_corpus_version,
            execution_status=BaselineExecutionStatus.FAILED,
            publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
            failures=(
                failure(
                    stage=stage,
                    category=category,
                    scope=BaselineFailureScope.REQUEST,
                    severity=BaselineFailureSeverity.FATAL,
                    retryable=retryable,
                ),
            ),
        )
