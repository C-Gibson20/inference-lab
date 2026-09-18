"""Tests for the structural corpus snapshot lifecycle protocol."""

from collections.abc import Iterator
from datetime import UTC, datetime
from types import TracebackType
from typing import Self

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
from inference_lab.storage import (
    ChunkSetSelectionRecord,
    CorpusSnapshot,
    CorpusVersion,
    CorpusVersionManifest,
    ParsedRepresentation,
    ParseOutcome,
    ResolvedSourceSpan,
    SourceArtefact,
)

CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002")
MANIFEST = CorpusVersionManifest(CORPUS_ID, VERSION_ID)
VERSION = CorpusVersion(
    CORPUS_ID,
    VERSION_ID,
    1,
    None,
    MANIFEST.manifest_digest,
    datetime(2026, 9, 5, 15, 0, tzinfo=UTC),
)


class _SnapshotWithoutClose:
    """Complete snapshot projection lacking only its ownership release."""

    corpus_id = CORPUS_ID
    version = VERSION
    manifest = MANIFEST
    is_closed = False

    def get_document(self, document_id: DocumentId) -> DocumentIdentity | None:
        raise NotImplementedError

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        raise NotImplementedError

    def list_document_versions(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentVersion, ...]:
        raise NotImplementedError

    def get_source_artefact(
        self,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        raise NotImplementedError

    def get_chunk_set_selection(
        self,
        document_version_id: DocumentVersionId,
        configuration_identity: ConfigurationIdentity,
    ) -> ChunkSetSelectionRecord | None:
        raise NotImplementedError

    def read_source_bytes(
        self,
        document_version_id: DocumentVersionId,
    ) -> bytes | None:
        raise NotImplementedError

    def stream_source_bytes(
        self,
        document_version_id: DocumentVersionId,
    ) -> Iterator[bytes] | None:
        raise NotImplementedError

    def get_parse_outcome(self, parse_id: RecordId) -> ParseOutcome | None:
        raise NotImplementedError

    def get_parsed_representation(
        self,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        raise NotImplementedError

    def resolve_source_span(
        self,
        source_span_id: SourceSpanId,
    ) -> ResolvedSourceSpan | None:
        raise NotImplementedError

    def find_document_versions_by_content_hash(
        self,
        content_hash: ContentHash,
    ) -> tuple[DocumentVersion, ...]:
        raise NotImplementedError

    def find_document_versions_by_external_identifiers(
        self,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        raise NotImplementedError

    def get_source_observation(
        self,
        source_observation_id: SourceObservationId,
    ) -> SourceObservationState | None:
        raise NotImplementedError

    def list_source_observation_history(
        self,
        source_observation_id: SourceObservationId,
    ) -> tuple[SourceObservationState, ...]:
        raise NotImplementedError

    def list_external_identifier_assertions(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        raise NotImplementedError

    def list_duplicate_relationships(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[DuplicateRelationship, ...]:
        raise NotImplementedError

    def get_document_family(
        self,
        family_id: DocumentFamilyId,
    ) -> DocumentFamilyId | None:
        raise NotImplementedError

    def list_family_memberships_by_family(
        self,
        family_id: DocumentFamilyId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        raise NotImplementedError

    def list_family_memberships_by_document(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        raise NotImplementedError

    def list_family_memberships_by_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False


class StructuralSnapshot(_SnapshotWithoutClose):
    """Complete structural snapshot independent of concrete storage adapters."""

    def close(self) -> None:
        pass


def test_snapshot_protocol_is_structural_and_runtime_checkable() -> None:
    assert isinstance(StructuralSnapshot(), CorpusSnapshot)


def test_object_without_owned_close_lifecycle_is_not_snapshot() -> None:
    assert not isinstance(_SnapshotWithoutClose(), CorpusSnapshot)
