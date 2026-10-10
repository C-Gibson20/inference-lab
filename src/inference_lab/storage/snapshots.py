"""Storage-neutral lifecycle contract for version-pinned corpus readers."""

from collections.abc import Iterator
from dataclasses import replace
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from inference_lab.provenance import (
    ConfigurationIdentity,
    ContentHash,
    CorpusId,
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
    RecordId,
    SourceObservationId,
    SourceObservationState,
    SourceSpanId,
)
from inference_lab.storage.chunk_records import ChunkSetSelectionRecord
from inference_lab.storage.parse_records import ParseOutcome
from inference_lab.storage.parsed_records import ParsedRepresentation
from inference_lab.storage.source_locations import ResolvedSourceSpan
from inference_lab.storage.source_records import SourceArtefact
from inference_lab.storage.versioning import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusVersion,
    CorpusVersionManifest,
)


@runtime_checkable
class CorpusSnapshot(Protocol):
    """Context-managed reader pinned to one complete committed corpus version."""

    @property
    def corpus_id(self) -> CorpusId:
        """Return the selected corpus identity."""
        ...

    @property
    def version(self) -> CorpusVersion:
        """Return the immutable committed version selected on entry."""
        ...

    @property
    def manifest(self) -> CorpusVersionManifest:
        """Return the complete materialised manifest selected on entry."""
        ...

    def get_document(self, document_id: DocumentId) -> DocumentIdentity | None:
        """Return a document visible in this version, or ``None``."""
        ...

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        """Return a document version visible in this version, or ``None``."""
        ...

    def list_document_versions(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentVersion, ...]:
        """Return only versions of a document visible in this version."""
        ...

    def get_source_artefact(
        self,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        """Return source metadata for a version visible in this snapshot."""
        ...

    def get_chunk_set_selection(
        self,
        document_version_id: DocumentVersionId,
        configuration_identity: ConfigurationIdentity,
    ) -> ChunkSetSelectionRecord | None:
        """Return a compatible chunk set for a visible version, or ``None``."""
        ...

    def read_source_bytes(
        self,
        document_version_id: DocumentVersionId,
    ) -> bytes | None:
        """Return verified source bytes for a version visible in this snapshot."""
        ...

    def stream_source_bytes(
        self,
        document_version_id: DocumentVersionId,
    ) -> Iterator[bytes] | None:
        """Return detached source chunks for a version visible in this snapshot."""
        ...

    def get_parse_outcome(self, parse_id: RecordId) -> ParseOutcome | None:
        """Return an outcome whose source version is visible in this snapshot."""
        ...

    def get_parsed_representation(
        self,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        """Return a parsed record whose source version is visible here."""
        ...

    def resolve_source_span(
        self,
        source_span_id: SourceSpanId,
    ) -> ResolvedSourceSpan | None:
        """Resolve a span whose source version is visible in this snapshot."""
        ...

    def find_document_versions_by_content_hash(
        self,
        content_hash: ContentHash,
    ) -> tuple[DocumentVersion, ...]:
        """Return matching document versions visible in this version."""
        ...

    def find_document_versions_by_external_identifiers(
        self,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        """Return identifier matches visible in this version."""
        ...

    def get_source_observation(
        self,
        source_observation_id: SourceObservationId,
    ) -> SourceObservationState | None:
        """Return the latest state visible in this version, or ``None``."""
        ...

    def list_source_observation_history(
        self,
        source_observation_id: SourceObservationId,
    ) -> tuple[SourceObservationState, ...]:
        """Return only observation states visible in this version."""
        ...

    def list_external_identifier_assertions(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        """Return assertions only for a visible document version."""
        ...

    def list_duplicate_relationships(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[DuplicateRelationship, ...]:
        """Return relationships whose two endpoints are visible."""
        ...

    def get_document_family(
        self,
        family_id: DocumentFamilyId,
    ) -> DocumentFamilyId | None:
        """Return a family related to visible membership, or ``None``."""
        ...

    def list_family_memberships_by_family(
        self,
        family_id: DocumentFamilyId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a family's visible memberships."""
        ...

    def list_family_memberships_by_document(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return visible family memberships for a visible document."""
        ...

    def list_family_memberships_by_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return visible family memberships for a visible version."""
        ...

    @property
    def is_closed(self) -> bool:
        """Return whether the snapshot has permanently released its reader."""
        ...

    def __enter__(self) -> Self:
        """Open and return this snapshot, rejecting nested reuse."""
        ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Close the snapshot while preserving any primary exception."""
        ...

    def close(self) -> None:
        """Release the owned reader idempotently and prevent further use."""
        ...


def generation_matches_reference(
    reference: ComponentGenerationReference,
    candidate: ComponentGenerationReference,
) -> bool:
    """Compare two component references on every field except lifecycle state.

    Callers build ``candidate`` from a durable manifest to confirm that the
    artefact they opened is the one a committed version named. Comparing whole
    references rather than selected fields means a new identity field is
    covered without revisiting each call site.
    """
    if not isinstance(reference, ComponentGenerationReference):
        raise TypeError("reference must be a ComponentGenerationReference.")
    if not isinstance(candidate, ComponentGenerationReference):
        raise TypeError("candidate must be a ComponentGenerationReference.")
    return replace(reference, state=candidate.state) == candidate


def select_ready_component_generation(
    manifest: CorpusVersionManifest,
    component_kind: ComponentKind,
    *,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> ComponentGenerationReference | None:
    """Return the one ready generation of a kind owned by the pinned version.

    Returns ``None`` when the manifest does not name exactly one ready
    generation of that kind for the corpus and version the caller pinned, so
    each caller maps the absence onto its own failure contract.
    """
    if not isinstance(component_kind, ComponentKind):
        raise TypeError("component_kind must be a ComponentKind.")
    references = tuple(
        reference
        for reference in manifest.component_generations
        if reference.component_kind == component_kind
    )
    if len(references) != 1:
        return None
    reference = references[0]
    if (
        reference.state is not ComponentGenerationState.READY
        or reference.corpus_id != corpus_id
        or reference.corpus_version != corpus_version
    ):
        return None
    return reference
