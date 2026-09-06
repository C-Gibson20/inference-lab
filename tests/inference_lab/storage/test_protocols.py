"""Tests for metadata storage protocols."""

from collections.abc import Iterable, Iterator
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
    KnownSourceSpan,
    RecordId,
    SourceObservationId,
    SourceObservationState,
    SourceSpanId,
)
from inference_lab.storage import (
    AbortCorpusUpdateOutcome,
    BeginCorpusUpdateOutcome,
    ComponentKind,
    CorpusUpdate,
    MetadataRepository,
    MetadataTransaction,
    ParsedRepresentation,
    ParseOutcome,
    ResolvedSourceSpan,
    SourceArtefact,
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


class StructuralRepository(StructuralTransaction):
    """Complete structural repository without protocol inheritance."""

    is_closed = False

    def transaction(self) -> MetadataTransaction:
        return StructuralTransaction()

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


def test_reader_without_writes_is_not_a_transaction() -> None:
    class ReaderOnly:
        get_corpus = StructuralTransaction.get_corpus
        list_corpora = StructuralTransaction.list_corpora
        get_corpus_update = StructuralTransaction.get_corpus_update
        list_corpus_update_document_memberships = (
            StructuralTransaction.list_corpus_update_document_memberships
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
        read_source_bytes = StructuralTransaction.read_source_bytes
        stream_source_bytes = StructuralTransaction.stream_source_bytes
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
        "CURRENT_SOURCE_RECORD_SCHEMA_VERSION",
        "SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS",
        "AbortCorpusUpdateOutcome",
        "BeginCorpusUpdateOutcome",
        "BoundingBox",
        "CollectionAvailability",
        "CommitCorpusUpdateOutcome",
        "ComponentGenerationReference",
        "ComponentGenerationState",
        "ComponentKind",
        "ContentOrigin",
        "CoordinateOrigin",
        "CoordinateUnit",
        "CorpusSnapshot",
        "CorpusUpdate",
        "CorpusUpdateState",
        "CorpusVersion",
        "CorpusVersionConflictKind",
        "CorpusVersionConflictOutcome",
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
        "SourceLocation",
        "StorageConcurrencyError",
        "StorageConflictError",
        "StorageError",
        "StorageErrorCategory",
        "StorageMigrationError",
        "StorageTransactionError",
        "StructuralBlock",
        "StructuralBlockKind",
        "build_parsed_representation_output_digest",
        "open_metadata_repository",
        "serialise_parse_outcome",
        "serialise_parsed_representation",
        "serialise_source_artefact",
        "validate_source_record_schema_version",
    }

    assert set(storage.__all__) == expected_exports
    assert all(hasattr(storage, name) for name in expected_exports)
    assert not any("sqlite" in name.lower() for name in storage.__all__)
