"""Reusable deterministic fixtures for SQLite storage tests."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusMetadata,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentIdentity,
    DocumentVersion,
    DuplicateRelationship,
)
from inference_lab.storage import MetadataTransaction
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

APPLIED_AT = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)


def fixed_applied_at() -> datetime:
    """Return the shared deterministic migration timestamp."""
    return APPLIED_AT


def open_repository(path: Path) -> SQLiteMetadataRepository:
    """Open a migrated repository with the deterministic test clock."""
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=fixed_applied_at,
    )


@dataclass(frozen=True, slots=True)
class CompleteMetadataGraph:
    """Values comprising one complete metadata graph fixture."""

    corpus: CorpusMetadata
    family_id: DocumentFamilyId
    versions: tuple[DocumentVersion, ...]
    duplicate_relationships: tuple[DuplicateRelationship, ...]
    family_memberships: tuple[DocumentFamilyMembership, ...]


def add_complete_graph(
    transaction: MetadataTransaction,
    graph: CompleteMetadataGraph,
) -> None:
    """Add a complete metadata graph through the public transaction contract."""
    transaction.add_corpus(graph.corpus)
    transaction.add_document_family(graph.family_id)

    document_ids = tuple(
        dict.fromkeys(version.document_id for version in graph.versions)
    )
    for document_id in document_ids:
        transaction.add_document(DocumentIdentity(document_id))
    for version in graph.versions:
        transaction.add_document_version(version)
    for version in graph.versions:
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                graph.corpus.corpus_id,
                version.document_id,
                version.document_version_id,
            )
        )
    for relationship in graph.duplicate_relationships:
        transaction.add_duplicate_relationship(relationship)
    for membership in graph.family_memberships:
        transaction.add_family_membership(membership)
