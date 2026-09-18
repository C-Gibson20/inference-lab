"""Tests for metadata storage protocols."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Self

import inference_lab.storage as storage
from inference_lab.provenance import (
    ConfigurationIdentity,
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateRelationship,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    KnownSourceSpan,
    RecordId,
    SourceObservationId,
    SourceObservationState,
    SourceSpanId,
)
from inference_lab.storage import (
    AbortCorpusUpdateOutcome,
    BaselineCheckpointReader,
    BaselineCheckpointRecord,
    BaselineCheckpointRecordStage,
    BaselineCheckpointRecordStatus,
    BaselineCheckpointRepository,
    BaselineCheckpointWriter,
    BaselineManifestRecord,
    BaselineManifestRecordState,
    BeginBaselineManifestOutcome,
    BeginCorpusUpdateOutcome,
    ChunkSetWriteRequest,
    CommitCorpusUpdateOutcome,
    ComponentGenerationReference,
    ComponentGenerationValidator,
    ComponentKind,
    CorpusSnapshot,
    CorpusUpdate,
    CorpusVersion,
    MetadataRepository,
    MetadataTransaction,
    NoCorpusChangeOutcome,
    ParsedRepresentation,
    ParseOutcome,
    ResolvedSourceSpan,
    RollbackCorpusVersionOutcome,
    SealCorpusUpdateOutcome,
    SourceArtefact,
    SourceFileIdentityHint,
    SourceLocatorRevision,
    TransitionBaselineManifestOutcome,
    WriteBaselineCheckpointOutcome,
    WriteChunkSetOutcome,
)


class StructuralTransaction:
    """Complete structural implementation independent of the protocols."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    def add_corpus(self, corpus: CorpusMetadata) -> None:
        pass

    def set_corpus_display_name(
        self, corpus_id: CorpusId, display_name: str
    ) -> CorpusMetadata:
        raise NotImplementedError

    def begin_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        *,
        expected_corpus_version: CorpusVersionId | None,
        required_component_kinds: tuple[ComponentKind, ...] = (),
        restores_corpus_version: CorpusVersionId | None = None,
    ) -> BeginCorpusUpdateOutcome:
        raise NotImplementedError

    def abort_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> AbortCorpusUpdateOutcome:
        raise NotImplementedError

    def register_corpus_update_component_generation(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        generation: ComponentGenerationReference,
        *,
        validator: ComponentGenerationValidator,
    ) -> ComponentGenerationReference:
        raise NotImplementedError

    def replace_corpus_update_component_generation(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        expected_generation: ComponentGenerationReference,
        replacement_generation: ComponentGenerationReference,
        *,
        validator: ComponentGenerationValidator,
    ) -> ComponentGenerationReference:
        raise NotImplementedError

    def seal_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> SealCorpusUpdateOutcome | NoCorpusChangeOutcome:
        raise NotImplementedError

    def commit_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CommitCorpusUpdateOutcome:
        raise NotImplementedError

    def commit_corpus_rollback(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> RollbackCorpusVersionOutcome:
        raise NotImplementedError

    def add_corpus_update_document_membership(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        membership: CorpusDocumentVersionMembership,
    ) -> None:
        pass

    def replace_corpus_update_document_membership(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        expected_membership: CorpusDocumentVersionMembership,
        replacement_membership: CorpusDocumentVersionMembership,
    ) -> None:
        pass

    def remove_corpus_update_document_membership(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        membership: CorpusDocumentVersionMembership,
    ) -> None:
        pass

    def add_corpus_document_version_membership(
        self, membership: CorpusDocumentVersionMembership
    ) -> None:
        pass

    def add_document_family(self, family_id: DocumentFamilyId) -> None:
        pass

    def add_document(self, document: DocumentIdentity) -> None:
        pass

    def add_document_version(self, document_version: DocumentVersion) -> None:
        pass

    def add_source_artefact(
        self,
        corpus_id: CorpusId,
        artefact: SourceArtefact,
    ) -> None:
        pass

    def add_parsed_representation(
        self,
        corpus_id: CorpusId,
        representation: ParsedRepresentation,
    ) -> None:
        pass

    def add_parse_outcome(
        self,
        corpus_id: CorpusId,
        outcome: ParseOutcome,
    ) -> None:
        pass

    def add_source_observation(self, state: SourceObservationState) -> None:
        pass

    def advance_source_observation(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
        *,
        expected_version_id: DocumentVersionId,
        new_version_id: DocumentVersionId,
    ) -> SourceObservationState:
        raise NotImplementedError

    def add_source_locator(self, revision: SourceLocatorRevision) -> None:
        pass

    def advance_source_locator(
        self,
        revision: SourceLocatorRevision,
    ) -> SourceLocatorRevision:
        raise NotImplementedError

    def add_external_identifier_assertions(
        self,
        corpus_id: CorpusId,
        assertions: tuple[ExternalIdentifierAssertion, ...],
    ) -> None:
        pass

    def add_duplicate_relationship(self, relationship: DuplicateRelationship) -> None:
        pass

    def add_family_membership(self, membership: DocumentFamilyMembership) -> None:
        pass

    def begin_baseline_manifest(
        self,
        manifest: BaselineManifestRecord,
    ) -> BeginBaselineManifestOutcome:
        raise NotImplementedError

    def transition_baseline_manifest(
        self,
        corpus_id: CorpusId,
        proposed_corpus_version: CorpusVersionId,
        *,
        expected_state: BaselineManifestRecordState,
        next_state: BaselineManifestRecordState,
    ) -> TransitionBaselineManifestOutcome:
        raise NotImplementedError

    def write_baseline_checkpoint(
        self,
        checkpoint: BaselineCheckpointRecord,
        *,
        expected_status: BaselineCheckpointRecordStatus | None,
    ) -> WriteBaselineCheckpointOutcome:
        raise NotImplementedError

    def write_chunk_set(
        self,
        request: ChunkSetWriteRequest,
    ) -> WriteChunkSetOutcome:
        raise NotImplementedError

    def get_corpus(self, corpus_id: CorpusId) -> CorpusMetadata | None:
        return None

    def list_corpora(self) -> tuple[CorpusMetadata, ...]:
        return ()

    def get_active_corpus_version(
        self,
        corpus_id: CorpusId,
    ) -> CorpusVersion | None:
        return None

    def get_corpus_version(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CorpusVersion | None:
        return None

    def list_corpus_versions(
        self,
        corpus_id: CorpusId,
    ) -> tuple[CorpusVersion, ...]:
        return ()

    def get_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CorpusUpdate | None:
        return None

    def list_non_terminal_corpus_updates(self) -> tuple[CorpusUpdate, ...]:
        return ()

    def list_corpus_update_document_memberships(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        return ()

    def list_corpus_update_component_generations(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> tuple[ComponentGenerationReference, ...]:
        return ()

    def get_corpus_document_version_membership(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> CorpusDocumentVersionMembership | None:
        return None

    def list_corpus_document_version_memberships(
        self, corpus_id: CorpusId
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        return ()

    def get_document_family(
        self, family_id: DocumentFamilyId
    ) -> DocumentFamilyId | None:
        return None

    def get_document(self, document_id: DocumentId) -> DocumentIdentity | None:
        return None

    def get_document_version(
        self, document_version_id: DocumentVersionId
    ) -> DocumentVersion | None:
        return None

    def list_document_versions(
        self, document_id: DocumentId
    ) -> tuple[DocumentVersion, ...]:
        return ()

    def get_source_artefact(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        return None

    def get_parse_outcome(
        self,
        corpus_id: CorpusId,
        parse_id: RecordId,
    ) -> ParseOutcome | None:
        return None

    def get_parsed_representation(
        self,
        corpus_id: CorpusId,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        return None

    def list_parsed_representations(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[ParsedRepresentation, ...]:
        return ()

    def resolve_source_span(
        self,
        corpus_id: CorpusId,
        source_span_id: SourceSpanId,
    ) -> ResolvedSourceSpan | None:
        return None

    def list_known_source_spans(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[KnownSourceSpan, ...]:
        return ()

    def find_document_versions_by_content_hash(
        self, corpus_id: CorpusId, content_hash: ContentHash
    ) -> tuple[DocumentVersion, ...]:
        return ()

    def find_document_versions_by_external_identifiers(
        self,
        corpus_id: CorpusId,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        return ()

    def get_source_observation(
        self, corpus_id: CorpusId, source_observation_id: SourceObservationId
    ) -> SourceObservationState | None:
        return None

    def list_source_observation_history(
        self, corpus_id: CorpusId, source_observation_id: SourceObservationId
    ) -> tuple[SourceObservationState, ...]:
        return ()

    def find_source_observations_by_document_version(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[SourceObservationState, ...]:
        return ()

    def get_source_locator(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> SourceLocatorRevision | None:
        return None

    def list_source_locator_history(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> tuple[SourceLocatorRevision, ...]:
        return ()

    def find_source_locator_by_path(
        self,
        corpus_id: CorpusId,
        path: Path,
    ) -> SourceLocatorRevision | None:
        return None

    def find_source_locators_by_identity_hint(
        self,
        corpus_id: CorpusId,
        identity_hint: SourceFileIdentityHint,
    ) -> tuple[SourceLocatorRevision, ...]:
        return ()

    def list_external_identifier_assertions(
        self, corpus_id: CorpusId, document_version_id: DocumentVersionId
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        return ()

    def list_duplicate_relationships(
        self, corpus_id: CorpusId, document_version_id: DocumentVersionId
    ) -> tuple[DuplicateRelationship, ...]:
        return ()

    def list_family_memberships_by_family(
        self, corpus_id: CorpusId, family_id: DocumentFamilyId
    ) -> tuple[DocumentFamilyMembership, ...]:
        return ()

    def list_family_memberships_by_document(
        self, corpus_id: CorpusId, document_id: DocumentId
    ) -> tuple[DocumentFamilyMembership, ...]:
        return ()

    def list_family_memberships_by_document_version(
        self, corpus_id: CorpusId, document_version_id: DocumentVersionId
    ) -> tuple[DocumentFamilyMembership, ...]:
        return ()

    def get_baseline_manifest(
        self,
        corpus_id: CorpusId,
        proposed_corpus_version: CorpusVersionId,
    ) -> BaselineManifestRecord | None:
        return None

    def get_baseline_checkpoint(
        self,
        corpus_id: CorpusId,
        proposed_corpus_version: CorpusVersionId,
        document_version_id: DocumentVersionId,
        stage: BaselineCheckpointRecordStage,
        configuration_identity: ConfigurationIdentity,
    ) -> BaselineCheckpointRecord | None:
        return None

    def list_baseline_checkpoints(
        self,
        corpus_id: CorpusId,
        proposed_corpus_version: CorpusVersionId,
    ) -> tuple[BaselineCheckpointRecord, ...]:
        return ()


class StructuralComponentValidator:
    """Complete structural component validator without protocol inheritance."""

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        return True


class StructuralRepository(StructuralTransaction):
    """Complete structural repository without protocol inheritance."""

    is_closed = False

    def transaction(self) -> MetadataTransaction:
        return StructuralTransaction()

    def open_active_corpus_snapshot(
        self,
        corpus_id: CorpusId,
    ) -> CorpusSnapshot:
        raise NotImplementedError

    def open_corpus_snapshot(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CorpusSnapshot:
        raise NotImplementedError

    def get_chunk_set_selection(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
        configuration_identity: ConfigurationIdentity,
    ) -> storage.ChunkSetSelectionRecord | None:
        del corpus_id, document_version_id, configuration_identity
        return None

    def read_source_bytes(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> bytes | None:
        return None

    def stream_source_bytes(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> Iterator[bytes] | None:
        return None

    def reconcile_source_storage(self, *, max_entries: int) -> None:
        pass

    def persist_source_artefact(
        self,
        corpus_id: CorpusId,
        artefact: SourceArtefact,
        chunks: Iterable[bytes],
        *,
        max_source_bytes: int,
        parse_outcome: ParseOutcome | None = None,
        parsed_representation: ParsedRepresentation | None = None,
    ) -> None:
        pass

    def close(self) -> None:
        pass


def test_storage_protocols_are_structural() -> None:
    transaction = StructuralTransaction()
    repository = StructuralRepository()

    assert isinstance(transaction, MetadataTransaction)
    assert isinstance(repository, MetadataRepository)
    assert isinstance(transaction, BaselineCheckpointRepository)
    assert isinstance(transaction, BaselineCheckpointReader)
    assert isinstance(transaction, BaselineCheckpointWriter)
    assert isinstance(StructuralComponentValidator(), ComponentGenerationValidator)
    assert not hasattr(transaction, "read_source_bytes")
    assert not hasattr(transaction, "stream_source_bytes")


def test_reader_without_writes_is_not_a_transaction() -> None:
    class ReaderOnly:
        get_corpus = StructuralTransaction.get_corpus
        list_corpora = StructuralTransaction.list_corpora
        get_active_corpus_version = StructuralTransaction.get_active_corpus_version
        get_corpus_version = StructuralTransaction.get_corpus_version
        list_corpus_versions = StructuralTransaction.list_corpus_versions
        get_corpus_update = StructuralTransaction.get_corpus_update
        list_non_terminal_corpus_updates = (
            StructuralTransaction.list_non_terminal_corpus_updates
        )
        list_corpus_update_document_memberships = (
            StructuralTransaction.list_corpus_update_document_memberships
        )
        list_corpus_update_component_generations = (
            StructuralTransaction.list_corpus_update_component_generations
        )
        get_corpus_document_version_membership = (
            StructuralTransaction.get_corpus_document_version_membership
        )
        list_corpus_document_version_memberships = (
            StructuralTransaction.list_corpus_document_version_memberships
        )
        get_document_family = StructuralTransaction.get_document_family
        get_document = StructuralTransaction.get_document
        get_document_version = StructuralTransaction.get_document_version
        list_document_versions = StructuralTransaction.list_document_versions
        get_source_artefact = StructuralTransaction.get_source_artefact
        get_parse_outcome = StructuralTransaction.get_parse_outcome
        get_parsed_representation = StructuralTransaction.get_parsed_representation
        list_parsed_representations = StructuralTransaction.list_parsed_representations
        resolve_source_span = StructuralTransaction.resolve_source_span
        list_known_source_spans = StructuralTransaction.list_known_source_spans
        find_document_versions_by_content_hash = (
            StructuralTransaction.find_document_versions_by_content_hash
        )
        find_document_versions_by_external_identifiers = (
            StructuralTransaction.find_document_versions_by_external_identifiers
        )
        get_source_observation = StructuralTransaction.get_source_observation
        list_source_observation_history = (
            StructuralTransaction.list_source_observation_history
        )
        find_source_observations_by_document_version = (
            StructuralTransaction.find_source_observations_by_document_version
        )
        get_source_locator = StructuralTransaction.get_source_locator
        list_source_locator_history = StructuralTransaction.list_source_locator_history
        find_source_locator_by_path = StructuralTransaction.find_source_locator_by_path
        find_source_locators_by_identity_hint = (
            StructuralTransaction.find_source_locators_by_identity_hint
        )
        list_external_identifier_assertions = (
            StructuralTransaction.list_external_identifier_assertions
        )
        list_duplicate_relationships = (
            StructuralTransaction.list_duplicate_relationships
        )
        list_family_memberships_by_family = (
            StructuralTransaction.list_family_memberships_by_family
        )
        list_family_memberships_by_document = (
            StructuralTransaction.list_family_memberships_by_document
        )
        list_family_memberships_by_document_version = (
            StructuralTransaction.list_family_memberships_by_document_version
        )

    assert not isinstance(ReaderOnly(), MetadataTransaction)


def test_public_storage_api_contains_only_contract_boundary() -> None:
    expected_exports = {
        "CORPUS_MANIFEST_CONTRACT_VERSION",
        "CURRENT_CHUNK_PAYLOAD_SCHEMA_VERSION",
        "CURRENT_SOURCE_RECORD_SCHEMA_VERSION",
        "SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS",
        "AbortCorpusUpdateOutcome",
        "BaselineCheckpointRecord",
        "BaselineCheckpointRecordStage",
        "BaselineCheckpointRecordStatus",
        "BaselineCheckpointReader",
        "BaselineCheckpointRepository",
        "BaselineCheckpointWriter",
        "BaselineManifestRecord",
        "BaselineManifestRecordState",
        "BeginBaselineManifestOutcome",
        "BaselineCheckpointIdentityRecord",
        "BaselineCheckpointRecord",
        "BaselineCheckpointRecordFailureCategory",
        "BaselineCheckpointRecordIdentityKind",
        "BaselineCheckpointRecordStage",
        "BaselineCheckpointRecordStatus",
        "BaselineManifestRecord",
        "BaselineManifestRecordState",
        "BeginCorpusUpdateOutcome",
        "BoundingBox",
        "ChunkerVersionRecord",
        "ChunkingPolicyRecord",
        "ChunkMinimumExceptionRecord",
        "ChunkParserVersionRecord",
        "ChunkSplitReasonRecord",
        "ChunkSourceReferenceRecord",
        "ChunkSetReader",
        "ChunkSetSelectionRecord",
        "ChunkSetWriteRequest",
        "ChunkSetWriter",
        "ChunkTokeniserIdentityRecord",
        "ChunkOversizeUnitPolicy",
        "ChunkStructuralBoundaryPolicy",
        "CollectionAvailability",
        "CommitCorpusUpdateOutcome",
        "ComponentGenerationReference",
        "ComponentGenerationState",
        "ComponentGenerationValidator",
        "ComponentKind",
        "ContentOrigin",
        "CoordinateOrigin",
        "CoordinateUnit",
        "CorpusSnapshot",
        "CorpusUpdate",
        "CorpusUpdateState",
        "CorpusVersion",
        "CorpusVersionManifest",
        "CorruptStorageError",
        "IncompatibleSchemaError",
        "InvalidCorpusVersionContractError",
        "MetadataReader",
        "MetadataRepository",
        "MetadataTransaction",
        "MetadataWriter",
        "MigrationRequiredError",
        "NoCorpusChangeOutcome",
        "PageLayoutDescriptor",
        "PageRotation",
        "ParseFailureCategory",
        "ParseFailureRecord",
        "ParseFailureSeverity",
        "ParseFailureStage",
        "ParseFailureUnitKind",
        "ParseOutcome",
        "ParseStatus",
        "ParsedRepresentation",
        "ProducerIdentity",
        "RepositoryClosedError",
        "RepositoryUnavailableError",
        "ResolvedSourceSpan",
        "RollbackCorpusVersionOutcome",
        "SealCorpusUpdateOutcome",
        "SourceArtefact",
        "SourceFileIdentityHint",
        "SourceFingerprint",
        "SourceLocation",
        "SourceLocatorRevision",
        "StorageConcurrencyError",
        "StorageConflictError",
        "StorageError",
        "StorageErrorCategory",
        "StorageMigrationError",
        "StorageTransactionError",
        "StructuralBlock",
        "StructuralBlockKind",
        "TransitionBaselineManifestOutcome",
        "WriteBaselineCheckpointOutcome",
        "WriteChunkSetOutcome",
        "TransitionBaselineManifestOutcome",
        "WriteBaselineCheckpointOutcome",
        "build_parsed_representation_output_digest",
        "build_source_blob_storage_key",
        "open_metadata_repository",
        "serialise_parse_outcome",
        "serialise_parsed_representation",
        "serialise_source_artefact",
        "validate_source_record_schema_version",
    }

    assert set(storage.__all__) == expected_exports
    assert all(hasattr(storage, name) for name in expected_exports)
    assert not any("sqlite" in name.lower() for name in storage.__all__)
