"""Tests for metadata storage protocols."""

from types import TracebackType
from typing import Self

import inference_lab.storage as storage
from inference_lab.provenance import (
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
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage import (
    AbortCorpusUpdateOutcome,
    BeginCorpusUpdateOutcome,
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
    SealCorpusUpdateOutcome,
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

    def close(self) -> None:
        pass


def test_storage_protocols_are_structural() -> None:
    transaction = StructuralTransaction()
    repository = StructuralRepository()

    assert isinstance(transaction, MetadataTransaction)
    assert isinstance(repository, MetadataRepository)
    assert isinstance(StructuralComponentValidator(), ComponentGenerationValidator)


def test_reader_without_writes_is_not_a_transaction() -> None:
    class ReaderOnly:
        get_corpus = StructuralTransaction.get_corpus
        list_corpora = StructuralTransaction.list_corpora
        get_active_corpus_version = StructuralTransaction.get_active_corpus_version
        get_corpus_version = StructuralTransaction.get_corpus_version
        list_corpus_versions = StructuralTransaction.list_corpus_versions
        get_corpus_update = StructuralTransaction.get_corpus_update
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
        "AbortCorpusUpdateOutcome",
        "BeginCorpusUpdateOutcome",
        "CommitCorpusUpdateOutcome",
        "ComponentGenerationReference",
        "ComponentGenerationState",
        "ComponentGenerationValidator",
        "ComponentKind",
        "CorruptStorageError",
        "CorpusSnapshot",
        "CorpusUpdate",
        "CorpusUpdateState",
        "CorpusVersion",
        "CorpusVersionConflictKind",
        "CorpusVersionConflictOutcome",
        "CorpusVersionManifest",
        "IncompatibleSchemaError",
        "InvalidCorpusVersionContractError",
        "MetadataReader",
        "MetadataRepository",
        "MetadataTransaction",
        "MetadataWriter",
        "MigrationRequiredError",
        "NoCorpusChangeOutcome",
        "RepositoryClosedError",
        "RepositoryUnavailableError",
        "RollbackCorpusVersionOutcome",
        "SealCorpusUpdateOutcome",
        "StorageConcurrencyError",
        "StorageConflictError",
        "StorageError",
        "StorageErrorCategory",
        "StorageMigrationError",
        "StorageTransactionError",
        "open_metadata_repository",
    }

    assert set(storage.__all__) == expected_exports
    assert all(hasattr(storage, name) for name in expected_exports)
    assert not any("sqlite" in name.lower() for name in storage.__all__)
