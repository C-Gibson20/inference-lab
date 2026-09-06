"""Storage-neutral contracts for persistent corpus identity metadata."""

from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

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
from inference_lab.storage.outcomes import (
    AbortCorpusUpdateOutcome,
    BeginCorpusUpdateOutcome,
    CommitCorpusUpdateOutcome,
    NoCorpusChangeOutcome,
    RollbackCorpusVersionOutcome,
    SealCorpusUpdateOutcome,
)
from inference_lab.storage.parse_records import ParseOutcome
from inference_lab.storage.parsed_records import ParsedRepresentation
from inference_lab.storage.snapshots import CorpusSnapshot
from inference_lab.storage.source_locations import ResolvedSourceSpan
from inference_lab.storage.source_records import SourceArtefact
from inference_lab.storage.versioning import (
    ComponentGenerationReference,
    ComponentKind,
    CorpusUpdate,
    CorpusVersion,
)


@runtime_checkable
class ComponentGenerationValidator(Protocol):
    """Validate one prepared immutable generation against proposed membership."""

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        """Return whether the generation is durable, compatible, and complete."""
        ...


class MetadataReader(Protocol):
    """Read immutable corpus identity metadata using provenance values."""

    def get_corpus(self, corpus_id: CorpusId) -> CorpusMetadata | None:
        """Return a named corpus, or ``None`` when absent."""
        ...

    def list_corpora(self) -> tuple[CorpusMetadata, ...]:
        """Return all named corpora in canonical identifier order."""
        ...

    def get_active_corpus_version(
        self,
        corpus_id: CorpusId,
    ) -> CorpusVersion | None:
        """Return a corpus's active committed version, or ``None``."""
        ...

    def get_corpus_version(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CorpusVersion | None:
        """Return one committed version owned by a corpus, or ``None``."""
        ...

    def list_corpus_versions(
        self,
        corpus_id: CorpusId,
    ) -> tuple[CorpusVersion, ...]:
        """Return committed history in ascending sequence order."""
        ...

    def get_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CorpusUpdate | None:
        """Return one owned durable corpus update, or ``None``."""
        ...

    def list_non_terminal_corpus_updates(self) -> tuple[CorpusUpdate, ...]:
        """Return durable preparing and ready work for explicit recovery."""
        ...

    def list_corpus_update_document_memberships(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        """Return deterministic proposed membership for one update."""
        ...

    def list_corpus_update_component_generations(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> tuple[ComponentGenerationReference, ...]:
        """Return deterministic ready generations staged for one update."""
        ...

    def get_corpus_document_version_membership(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> CorpusDocumentVersionMembership | None:
        """Return one corpus catalogue membership, or ``None``."""
        ...

    def list_corpus_document_version_memberships(
        self,
        corpus_id: CorpusId,
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        """Return a corpus catalogue in canonical document/version order."""
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

    def get_source_artefact(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        """Return corpus-visible managed-source metadata, or ``None``."""
        ...

    def get_parse_outcome(
        self,
        corpus_id: CorpusId,
        parse_id: RecordId,
    ) -> ParseOutcome | None:
        """Return a complete corpus-visible parser outcome, or ``None``."""
        ...

    def get_parsed_representation(
        self,
        corpus_id: CorpusId,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        """Return a complete corpus-visible parsed record, or ``None``."""
        ...

    def list_parsed_representations(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[ParsedRepresentation, ...]:
        """Return a version's parsed records in canonical identity order."""
        ...

    def resolve_source_span(
        self,
        corpus_id: CorpusId,
        source_span_id: SourceSpanId,
    ) -> ResolvedSourceSpan | None:
        """Return a corpus-visible span and its coordinate descriptor."""
        ...

    def list_known_source_spans(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[KnownSourceSpan, ...]:
        """Return a version's validated PROV-03 source-span catalogue."""
        ...

    def find_document_versions_by_content_hash(
        self, corpus_id: CorpusId, content_hash: ContentHash
    ) -> tuple[DocumentVersion, ...]:
        """Return exact-hash candidates in canonical identifier order."""
        ...

    def find_document_versions_by_external_identifiers(
        self,
        corpus_id: CorpusId,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        """Return unique candidates matching any supplied external identifier."""
        ...

    def get_source_observation(
        self, corpus_id: CorpusId, source_observation_id: SourceObservationId
    ) -> SourceObservationState | None:
        """Return the current state for an observed source, or ``None``."""
        ...

    def list_source_observation_history(
        self, corpus_id: CorpusId, source_observation_id: SourceObservationId
    ) -> tuple[SourceObservationState, ...]:
        """Return an observed source's append-only state history."""
        ...

    def list_external_identifier_assertions(
        self, corpus_id: CorpusId, document_version_id: DocumentVersionId
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        """Return append-only identifier assertions for one source version."""
        ...

    def list_duplicate_relationships(
        self, corpus_id: CorpusId, document_version_id: DocumentVersionId
    ) -> tuple[DuplicateRelationship, ...]:
        """Return relationships involving a version in canonical key order."""
        ...

    def list_family_memberships_by_family(
        self, corpus_id: CorpusId, family_id: DocumentFamilyId
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a family's memberships in canonical membership-key order."""
        ...

    def list_family_memberships_by_document(
        self, corpus_id: CorpusId, document_id: DocumentId
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a document's memberships in canonical membership-key order."""
        ...

    def list_family_memberships_by_document_version(
        self, corpus_id: CorpusId, document_version_id: DocumentVersionId
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a version's memberships in canonical membership-key order."""
        ...


class MetadataWriter(Protocol):
    """Write identity metadata within an owned transaction."""

    def add_corpus(self, corpus: CorpusMetadata) -> None:
        """Create a named corpus, treating exact replay as a no-op."""
        ...

    def set_corpus_display_name(
        self,
        corpus_id: CorpusId,
        display_name: str,
    ) -> CorpusMetadata:
        """Replace a corpus display name without changing its identity."""
        ...

    def begin_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        *,
        expected_corpus_version: CorpusVersionId | None,
        required_component_kinds: tuple[ComponentKind, ...] = (),
        restores_corpus_version: CorpusVersionId | None = None,
    ) -> BeginCorpusUpdateOutcome:
        """Create and seed an update, deriving rollback requirements from its target."""
        ...

    def abort_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> AbortCorpusUpdateOutcome:
        """Abort preparing or ready work or return its exact replay."""
        ...

    def register_corpus_update_component_generation(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        generation: ComponentGenerationReference,
        *,
        validator: ComponentGenerationValidator,
    ) -> ComponentGenerationReference:
        """Validate and register one required immutable generation."""
        ...

    def replace_corpus_update_component_generation(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        expected_generation: ComponentGenerationReference,
        replacement_generation: ComponentGenerationReference,
        *,
        validator: ComponentGenerationValidator,
    ) -> ComponentGenerationReference:
        """Validate and replace one explicitly expected generation."""
        ...

    def seal_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> SealCorpusUpdateOutcome | NoCorpusChangeOutcome:
        """Seal a complete manifest or terminate an unchanged proposal."""
        ...

    def commit_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CommitCorpusUpdateOutcome:
        """Atomically compare, advance, and publish one ready update."""
        ...

    def commit_corpus_rollback(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> RollbackCorpusVersionOutcome:
        """Publish one ready rollback as a new monotonic corpus version."""
        ...

    def add_corpus_update_document_membership(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        membership: CorpusDocumentVersionMembership,
    ) -> None:
        """Add a catalogue member to a preparing update or accept replay."""
        ...

    def replace_corpus_update_document_membership(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        expected_membership: CorpusDocumentVersionMembership,
        replacement_membership: CorpusDocumentVersionMembership,
    ) -> None:
        """Replace one explicitly agreeing logical-document membership."""
        ...

    def remove_corpus_update_document_membership(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        membership: CorpusDocumentVersionMembership,
    ) -> None:
        """Remove one exact proposed membership or accept replay."""
        ...

    def add_corpus_document_version_membership(
        self,
        membership: CorpusDocumentVersionMembership,
    ) -> None:
        """Add an explicit catalogue membership or accept exact replay."""
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

    def add_source_artefact(
        self,
        corpus_id: CorpusId,
        artefact: SourceArtefact,
    ) -> None:
        """Add managed-source metadata after proving corpus membership."""
        ...

    def add_parsed_representation(
        self,
        corpus_id: CorpusId,
        representation: ParsedRepresentation,
    ) -> None:
        """Add one complete parsed aggregate after proving corpus membership."""
        ...

    def add_parse_outcome(
        self,
        corpus_id: CorpusId,
        outcome: ParseOutcome,
    ) -> None:
        """Add one complete parser outcome after proving corpus membership."""
        ...

    def add_source_observation(self, state: SourceObservationState) -> None:
        """Create an observed source at revision one or accept exact replay."""
        ...

    def advance_source_observation(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
        *,
        expected_version_id: DocumentVersionId,
        new_version_id: DocumentVersionId,
    ) -> SourceObservationState:
        """Atomically advance an observed source from its expected version."""
        ...

    def add_external_identifier_assertions(
        self,
        corpus_id: CorpusId,
        assertions: tuple[ExternalIdentifierAssertion, ...],
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

    def open_active_corpus_snapshot(
        self,
        corpus_id: CorpusId,
    ) -> CorpusSnapshot:
        """Return an unopened reader that will pin the active version."""
        ...

    def open_corpus_snapshot(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CorpusSnapshot:
        """Return an unopened reader for one committed historical version."""
        ...

    def read_source_bytes(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> bytes | None:
        """Return verified corpus-visible source bytes, or ``None``."""
        ...

    def stream_source_bytes(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> Iterator[bytes] | None:
        """Return fully verified, detached source chunks, or ``None``."""
        ...

    def reconcile_source_storage(self, *, max_entries: int) -> None:
        """Clean abandoned staging and report a bounded final-blob inventory."""
        ...

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
        """Publish bytes, source metadata, and optional parse records safely."""
        ...

    def close(self) -> None:
        """Release resources idempotently and reject subsequent operations."""
        ...
