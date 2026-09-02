"""Storage-neutral contracts for persistent corpus identity metadata."""

from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from inference_lab.provenance import (
    ContentHash,
    CorpusId,
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


class MetadataReader(Protocol):
    """Read immutable corpus identity metadata using provenance values."""

    def get_corpus(self, corpus_id: CorpusId) -> CorpusId | None:
        """Return a persistent corpus identity, or ``None`` when absent."""
        ...

    def get_document_family(
        self, family_id: DocumentFamilyId
    ) -> DocumentFamilyId | None:
        """Return a document-family identity, or ``None`` when absent."""
        ...

    def get_document(self, document_id: DocumentId) -> DocumentIdentity | None:
        """Return a logical document, or ``None`` when absent."""
        ...

    def get_document_version(
        self, document_version_id: DocumentVersionId
    ) -> DocumentVersion | None:
        """Return an immutable document version, or ``None`` when absent."""
        ...

    def list_document_versions(
        self, document_id: DocumentId
    ) -> tuple[DocumentVersion, ...]:
        """Return a document's versions in canonical identifier order."""
        ...

    def find_document_versions_by_content_hash(
        self, content_hash: ContentHash
    ) -> tuple[DocumentVersion, ...]:
        """Return exact-hash candidates in canonical identifier order."""
        ...

    def find_document_versions_by_external_identifiers(
        self, external_identifiers: tuple[ExternalIdentifier, ...]
    ) -> tuple[DocumentVersion, ...]:
        """Return unique candidates matching any supplied external identifier."""
        ...

    def get_source_observation(
        self, source_observation_id: SourceObservationId
    ) -> SourceObservationState | None:
        """Return the current state for an observed source, or ``None``."""
        ...

    def list_source_observation_history(
        self, source_observation_id: SourceObservationId
    ) -> tuple[SourceObservationState, ...]:
        """Return an observed source's append-only state history."""
        ...

    def list_external_identifier_assertions(
        self, document_version_id: DocumentVersionId
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        """Return append-only identifier assertions for one source version."""
        ...

    def list_duplicate_relationships(
        self, document_version_id: DocumentVersionId
    ) -> tuple[DuplicateRelationship, ...]:
        """Return relationships involving a version in canonical key order."""
        ...

    def list_family_memberships_by_family(
        self, family_id: DocumentFamilyId
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a family's memberships in canonical membership-key order."""
        ...

    def list_family_memberships_by_document(
        self, document_id: DocumentId
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a document's memberships in canonical membership-key order."""
        ...

    def list_family_memberships_by_document_version(
        self, document_version_id: DocumentVersionId
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a version's memberships in canonical membership-key order."""
        ...


class MetadataWriter(Protocol):
    """Write identity metadata within an owned transaction."""

    def add_corpus(self, corpus_id: CorpusId) -> None:
        """Create a corpus identity, treating exact replay as a no-op."""
        ...

    def add_document_family(self, family_id: DocumentFamilyId) -> None:
        """Create a family identity, treating exact replay as a no-op."""
        ...

    def add_document(self, document: DocumentIdentity) -> None:
        """Create a logical document, treating exact replay as a no-op."""
        ...

    def add_document_version(self, document_version: DocumentVersion) -> None:
        """Create a version and its ordered external identifiers atomically."""
        ...

    def add_source_observation(self, state: SourceObservationState) -> None:
        """Create an observed source at revision one or accept exact replay."""
        ...

    def advance_source_observation(
        self,
        source_observation_id: SourceObservationId,
        *,
        expected_version_id: DocumentVersionId,
        new_version_id: DocumentVersionId,
    ) -> SourceObservationState:
        """Atomically advance an observed source from its expected version."""
        ...

    def add_external_identifier_assertions(
        self, assertions: tuple[ExternalIdentifierAssertion, ...]
    ) -> None:
        """Append one canonical identifier-evidence batch or accept replay."""
        ...

    def add_duplicate_relationship(self, relationship: DuplicateRelationship) -> None:
        """Create a canonical duplicate relationship or accept exact replay."""
        ...

    def add_family_membership(self, membership: DocumentFamilyMembership) -> None:
        """Create a family membership or accept exact replay."""
        ...


@runtime_checkable
class MetadataTransaction(MetadataReader, MetadataWriter, Protocol):
    """Context-managed atomic unit of metadata reads and writes.

    Successful context exit commits the complete unit. Exceptional exit rolls
    it back. Implementations reject accidental nesting and use after exit.
    """

    def __enter__(self) -> Self:
        """Start the transaction and return its storage-neutral operations."""
        ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Commit on success or roll back while preserving primary failures."""
        ...


@runtime_checkable
class MetadataRepository(MetadataReader, Protocol):
    """Opened, restart-safe metadata repository with explicit lifecycle."""

    @property
    def is_closed(self) -> bool:
        """Return whether this repository has been permanently closed."""
        ...

    def transaction(self) -> MetadataTransaction:
        """Return a new transaction that starts on context entry."""
        ...

    def close(self) -> None:
        """Release resources idempotently and reject subsequent operations."""
        ...
