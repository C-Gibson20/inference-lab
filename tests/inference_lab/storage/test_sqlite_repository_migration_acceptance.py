import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    ExternalIdentifier,
    HashAlgorithm,
)
from inference_lab.storage import MigrationRequiredError, StorageMigrationError
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
_CORPUS_ID = CorpusId("01014415-1989-449f-8824-cb13e418d92f")
_FAMILY_ID = DocumentFamilyId("6a5a72ce-e82f-4168-adb4-36c090dd19f0")
_FIRST_DOCUMENT_ID = DocumentId("056b503e-e15d-4ed2-87db-d352ff97dcf2")
_SECOND_DOCUMENT_ID = DocumentId("277f6718-4465-4323-9cf0-c09007e8d366")
_FIRST_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_SECOND_VERSION_ID = DocumentVersionId("22222222-2222-4222-8222-222222222222")
_FIRST_VERSION = DocumentVersion(
    _FIRST_VERSION_ID,
    _FIRST_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
    (ExternalIdentifier("doi", "10.1000/private-migration"),),
)
_SECOND_VERSION = DocumentVersion(
    _SECOND_VERSION_ID,
    _SECOND_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "b" * 64),
)
_DUPLICATE = DuplicateRelationship(
    _FIRST_VERSION_ID,
    _SECOND_VERSION_ID,
    DuplicateClass.BYTE_IDENTICAL,
)
_MEMBERSHIP = DocumentFamilyMembership(
    _FAMILY_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
)


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _revision_two() -> MigrationDefinition:
    return MigrationDefinition(
        revision=2,
        migration_id="test_add_corpus_note",
        statements=("ALTER TABLE corpora ADD COLUMN migration_note TEXT",),
    )


def _failing_revision_three() -> MigrationDefinition:
    return MigrationDefinition(
        revision=3,
        migration_id="test_failing_revision_three",
        statements=(
            "CREATE TABLE revision_three_marker (value TEXT) STRICT",
            "CREATE TABLE documents (conflicting_value TEXT)",
        ),
    )


def _create_complete_revision_one(path: Path) -> None:
    repository = SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS_ID)
        transaction.add_document_family(_FAMILY_ID)
        transaction.add_document(DocumentIdentity(_FIRST_DOCUMENT_ID))
        transaction.add_document(DocumentIdentity(_SECOND_DOCUMENT_ID))
        transaction.add_document_version(_FIRST_VERSION)
        transaction.add_document_version(_SECOND_VERSION)
        transaction.add_duplicate_relationship(_DUPLICATE)
        transaction.add_family_membership(_MEMBERSHIP)
    repository.close()


def _database_snapshot(path: Path) -> tuple[object, ...]:
    connection = sqlite3.connect(path)
    try:
        user_version = connection.execute("PRAGMA user_version").fetchone()
        schema_version = connection.execute("PRAGMA schema_version").fetchone()
        history = connection.execute(
            """SELECT revision, migration_id, checksum, applied_at_utc
            FROM migration_history ORDER BY revision"""
        ).fetchall()
        counts = tuple(
            connection.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
            for table_name in (
                "corpora",
                "documents",
                "document_versions",
                "external_identifiers",
                "duplicate_relationships",
                "document_families",
                "document_family_memberships",
            )
        )
        return user_version, schema_version, tuple(history), counts
    finally:
        connection.close()


def _assert_complete_graph(repository: SQLiteMetadataRepository) -> None:
    assert repository.get_corpus(_CORPUS_ID) == _CORPUS_ID
    assert repository.get_document_family(_FAMILY_ID) == _FAMILY_ID
    assert repository.get_document(_FIRST_DOCUMENT_ID) == DocumentIdentity(
        _FIRST_DOCUMENT_ID
    )
    assert repository.get_document_version(_FIRST_VERSION_ID) == _FIRST_VERSION
    assert repository.get_document_version(_SECOND_VERSION_ID) == _SECOND_VERSION
    assert repository.list_duplicate_relationships(_FIRST_VERSION_ID) == (_DUPLICATE,)
    assert repository.list_family_memberships_by_family(_FAMILY_ID) == (_MEMBERSHIP,)


def test_non_empty_repository_requires_explicit_upgrade_without_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_revision_one(database_path)
    migrations = (MIGRATIONS[0], _revision_two())
    before = _database_snapshot(database_path)

    with pytest.raises(MigrationRequiredError):
        SQLiteMetadataRepository.open(database_path, migrations=migrations)

    assert _database_snapshot(database_path) == before


def test_supported_upgrade_preserves_complete_graph_and_reopens_idempotently(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_revision_one(database_path)
    migrations = (MIGRATIONS[0], _revision_two())

    migrated = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        migrations=migrations,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert migrated.schema_inspection.current_revision == 2
        _assert_complete_graph(migrated)
    finally:
        migrated.close()

    reopened = SQLiteMetadataRepository.open(
        database_path,
        migrations=migrations,
    )
    try:
        assert reopened.schema_inspection.current_revision == 2
        _assert_complete_graph(reopened)
    finally:
        reopened.close()

    before_idempotent_reopen = _database_snapshot(database_path)
    idempotent = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        migrations=migrations,
        applied_at_factory=_fixed_applied_at,
    )
    idempotent.close()

    assert _database_snapshot(database_path) == before_idempotent_reopen


def test_failed_later_migration_preserves_last_checkpoint_and_domain_data(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_revision_one(database_path)
    revision_two = _revision_two()

    with pytest.raises(StorageMigrationError):
        SQLiteMetadataRepository.open(
            database_path,
            apply_migrations=True,
            migrations=(MIGRATIONS[0], revision_two, _failing_revision_three()),
            applied_at_factory=_fixed_applied_at,
        )

    recovered = SQLiteMetadataRepository.open(
        database_path,
        migrations=(MIGRATIONS[0], revision_two),
    )
    try:
        assert recovered.schema_inspection.current_revision == 2
        _assert_complete_graph(recovered)
    finally:
        recovered.close()

    connection = sqlite3.connect(database_path)
    try:
        assert (
            connection.execute(
                """SELECT 1 FROM sqlite_schema
            WHERE name = 'revision_three_marker'"""
            ).fetchone()
            is None
        )
        assert connection.execute(
            "SELECT count(*) FROM migration_history"
        ).fetchone() == (2,)
    finally:
        connection.close()
