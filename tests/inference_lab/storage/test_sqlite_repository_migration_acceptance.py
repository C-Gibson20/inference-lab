"""Acceptance tests for SQLite repository migration."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
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
from inference_lab.storage import (
    MetadataRepository,
    MigrationRequiredError,
    ParseOutcome,
    ParseStatus,
    SourceArtefact,
    StorageMigrationError,
    open_metadata_repository,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import fixed_applied_at as _fixed_applied_at  # noqa: TID252
from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    PARSE_ID,
    PARSED_RECORD_ID,
    PARSER,
    SOURCE_BYTES,
    SOURCE_HASH,
    parsed_representation,
)

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
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
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
_SOURCE_VERSION = DocumentVersion(DOCUMENT_VERSION_ID, DOCUMENT_ID, SOURCE_HASH)
_SOURCE_MEMBERSHIP = CorpusDocumentVersionMembership(
    _CORPUS_ID,
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
)
_SOURCE_ARTEFACT = SourceArtefact(
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    content_hash=SOURCE_HASH,
    byte_count=len(SOURCE_BYTES),
    media_type="application/pdf",
    storage_key=f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}",
)
_SOURCE_OUTCOME = ParseOutcome(
    parse_id=PARSE_ID,
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    status=ParseStatus.SUCCEEDED,
    parser=PARSER,
    normalisation_version="1.0.0",
    parsed_record_id=PARSED_RECORD_ID,
    failures=(),
)


def _failing_source_persistence_migration() -> MigrationDefinition:
    return MigrationDefinition(
        revision=3,
        migration_id="test_failing_source_persistence",
        statements=(
            *MIGRATIONS[2].statements,
            "CREATE TABLE documents (conflicting_value TEXT)",
        ),
    )


def _failing_revision_eight() -> MigrationDefinition:
    return MigrationDefinition(
        revision=8,
        migration_id="test_failing_revision_eight",
        statements=(
            "CREATE TABLE revision_seven_marker (value TEXT) STRICT",
            "CREATE TABLE documents (conflicting_value TEXT)",
        ),
    )


def _create_complete_baseline(
    path: Path,
    *,
    migrations: tuple[MigrationDefinition, ...] = (MIGRATIONS[0],),
) -> None:
    repository = SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        migrations=migrations,
        applied_at_factory=_fixed_applied_at,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Primary corpus"))
        transaction.add_document_family(_FAMILY_ID)
        transaction.add_document(DocumentIdentity(_FIRST_DOCUMENT_ID))
        transaction.add_document(DocumentIdentity(_SECOND_DOCUMENT_ID))
        transaction.add_document_version(_FIRST_VERSION)
        transaction.add_document_version(_SECOND_VERSION)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS_ID,
                _FIRST_DOCUMENT_ID,
                _FIRST_VERSION_ID,
            )
        )
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS_ID,
                _SECOND_DOCUMENT_ID,
                _SECOND_VERSION_ID,
            )
        )
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


def _assert_complete_graph(
    repository: MetadataRepository,
    *,
    additional_memberships: tuple[CorpusDocumentVersionMembership, ...] = (),
) -> None:
    assert repository.get_document_family(_FAMILY_ID) == _FAMILY_ID
    assert repository.get_document(_FIRST_DOCUMENT_ID) == DocumentIdentity(
        _FIRST_DOCUMENT_ID
    )
    assert repository.get_document_version(_FIRST_VERSION_ID) == _FIRST_VERSION
    assert repository.get_document_version(_SECOND_VERSION_ID) == _SECOND_VERSION
    expected_memberships = (
        CorpusDocumentVersionMembership(
            _CORPUS_ID, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID
        ),
        CorpusDocumentVersionMembership(
            _CORPUS_ID, _SECOND_DOCUMENT_ID, _SECOND_VERSION_ID
        ),
        *additional_memberships,
    )
    assert repository.list_corpus_document_version_memberships(_CORPUS_ID) == tuple(
        sorted(
            expected_memberships,
            key=lambda membership: membership.document_version_id.as_str(),
        )
    )
    assert repository.list_duplicate_relationships(_CORPUS_ID, _FIRST_VERSION_ID) == (
        _DUPLICATE,
    )
    assert repository.list_family_memberships_by_family(_CORPUS_ID, _FAMILY_ID) == (
        _MEMBERSHIP,
    )


def test_corpus_version_upgrade_preserves_complete_stor_01_graph(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_baseline(database_path, migrations=MIGRATIONS[:2])

    reopened = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert reopened.schema_inspection.current_revision == 7
        assert reopened.get_corpus(_CORPUS_ID) == CorpusMetadata(
            _CORPUS_ID, "Primary corpus"
        )
        assert reopened.list_corpus_document_version_memberships(_CORPUS_ID) == (
            CorpusDocumentVersionMembership(
                _CORPUS_ID, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID
            ),
            CorpusDocumentVersionMembership(
                _CORPUS_ID, _SECOND_DOCUMENT_ID, _SECOND_VERSION_ID
            ),
        )
        assert reopened.list_duplicate_relationships(_CORPUS_ID, _FIRST_VERSION_ID) == (
            _DUPLICATE,
        )
        assert reopened.list_family_memberships_by_family(_CORPUS_ID, _FAMILY_ID) == (
            _MEMBERSHIP,
        )
        _assert_complete_graph(reopened)
    finally:
        reopened.close()

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            """SELECT corpus_id, corpus_version
            FROM corpus_active_versions"""
        ).fetchall() == [(_CORPUS_ID.as_str(), None)]
        assert connection.execute(
            """SELECT revision, migration_id, checksum, applied_at_utc
            FROM migration_history
            WHERE revision = 4"""
        ).fetchone() == (
            4,
            "corpus_version_publication",
            MIGRATIONS[3].checksum,
            "2026-09-01T12:30:00.000000+00:00",
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()

    reopened_without_permission = SQLiteMetadataRepository.open(database_path)
    try:
        assert reopened_without_permission.schema_inspection.current_revision == 7
        _assert_complete_graph(reopened_without_permission)
    finally:
        reopened_without_permission.close()


@pytest.mark.parametrize("legacy_revision", range(1, 7))
def test_each_supported_legacy_revision_upgrades_to_chunk_schema(
    tmp_path: Path,
    legacy_revision: int,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_baseline(
        database_path,
        migrations=MIGRATIONS[:legacy_revision],
    )

    migrated = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert migrated.schema_inspection.current_revision == 7
        _assert_complete_graph(migrated)
    finally:
        migrated.close()

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'chunk_sets'"
        ).fetchone() == ("chunk_sets",)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'chunk_sets_selection_idx'"
        ).fetchone() == ("chunk_sets_selection_idx",)
        assert connection.execute(
            "SELECT count(*) FROM migration_history WHERE revision = 7"
        ).fetchone() == (1,)
    finally:
        connection.close()


def test_non_empty_repository_requires_explicit_upgrade_without_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_baseline(database_path)
    before = _database_snapshot(database_path)

    with pytest.raises(MigrationRequiredError):
        SQLiteMetadataRepository.open(database_path)

    assert _database_snapshot(database_path) == before


def test_failed_corpus_version_migration_rolls_back_all_schema_and_data_changes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_baseline(database_path, migrations=MIGRATIONS[:3])
    before = _database_snapshot(database_path)
    failing_migration = MigrationDefinition(
        revision=4,
        migration_id="test_failing_corpus_version_publication",
        statements=(
            *MIGRATIONS[3].statements[:-1],
            "CREATE TABLE documents (conflicting_value TEXT)",
        ),
    )

    with pytest.raises(StorageMigrationError):
        SQLiteMetadataRepository.open(
            database_path,
            apply_migrations=True,
            migrations=(*MIGRATIONS[:3], failing_migration),
            applied_at_factory=_fixed_applied_at,
        )

    assert _database_snapshot(database_path) == before
    connection = sqlite3.connect(database_path)
    try:
        assert (
            connection.execute(
                """SELECT name
            FROM sqlite_schema
            WHERE name IN (
                'corpus_updates',
                'corpus_versions',
                'corpus_active_versions'
            )"""
            ).fetchall()
            == []
        )
    finally:
        connection.close()

    recovered = SQLiteMetadataRepository.open(
        database_path,
        migrations=MIGRATIONS[:3],
    )
    try:
        assert recovered.schema_inspection.current_revision == 3
        _assert_complete_graph(recovered)
    finally:
        recovered.close()


def test_supported_upgrade_preserves_complete_graph_and_reopens_idempotently(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_baseline(database_path)

    migrated = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert migrated.schema_inspection.current_revision == 7
        _assert_complete_graph(migrated)
    finally:
        migrated.close()

    assert _database_snapshot(database_path)[2] == (
        (
            1,
            "multi_corpus_metadata_baseline",
            MIGRATIONS[0].checksum,
            "2026-09-01T12:30:00.000000+00:00",
        ),
        (
            2,
            "source_locator_continuity",
            MIGRATIONS[1].checksum,
            "2026-09-01T12:30:00.000000+00:00",
        ),
        (
            3,
            "source_persistence",
            MIGRATIONS[2].checksum,
            "2026-09-01T12:30:00.000000+00:00",
        ),
        (
            4,
            "corpus_version_publication",
            MIGRATIONS[3].checksum,
            "2026-09-01T12:30:00.000000+00:00",
        ),
        (
            5,
            "baseline_checkpoint_persistence",
            MIGRATIONS[4].checksum,
            "2026-09-01T12:30:00.000000+00:00",
        ),
        (
            6,
            "baseline_checkpoint_records",
            MIGRATIONS[5].checksum,
            "2026-09-01T12:30:00.000000+00:00",
        ),
        (
            7,
            "canonical_chunk_sets",
            MIGRATIONS[6].checksum,
            "2026-09-01T12:30:00.000000+00:00",
        ),
    )

    reopened = SQLiteMetadataRepository.open(
        database_path,
    )
    try:
        assert reopened.schema_inspection.current_revision == 7
        _assert_complete_graph(reopened)
    finally:
        reopened.close()

    before_idempotent_reopen = _database_snapshot(database_path)
    idempotent = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    idempotent.close()

    assert _database_snapshot(database_path) == before_idempotent_reopen


def test_failed_later_migration_preserves_last_checkpoint_and_domain_data(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_baseline(database_path)

    with pytest.raises(StorageMigrationError):
        SQLiteMetadataRepository.open(
            database_path,
            apply_migrations=True,
            migrations=(*MIGRATIONS, _failing_revision_eight()),
            applied_at_factory=_fixed_applied_at,
        )

    recovered = SQLiteMetadataRepository.open(
        database_path,
    )
    try:
        assert recovered.schema_inspection.current_revision == 7
        _assert_complete_graph(recovered)
    finally:
        recovered.close()

    connection = sqlite3.connect(database_path)
    try:
        assert (
            connection.execute(
                """SELECT 1 FROM sqlite_schema
                WHERE name = 'revision_seven_marker'"""
            ).fetchone()
            is None
        )
        assert connection.execute(
            "SELECT count(*) FROM migration_history"
        ).fetchone() == (7,)
    finally:
        connection.close()


def test_failed_source_migration_rolls_back_the_entire_revision(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_complete_baseline(database_path, migrations=MIGRATIONS[:2])
    before = _database_snapshot(database_path)

    with pytest.raises(StorageMigrationError):
        SQLiteMetadataRepository.open(
            database_path,
            apply_migrations=True,
            migrations=(*MIGRATIONS[:2], _failing_source_persistence_migration()),
            applied_at_factory=_fixed_applied_at,
        )

    assert _database_snapshot(database_path) == before
    connection = sqlite3.connect(database_path)
    try:
        assert (
            connection.execute(
                """SELECT 1 FROM sqlite_schema
            WHERE name = 'source_artefacts'"""
            ).fetchone()
            is None
        )
        assert connection.execute(
            "SELECT revision FROM migration_history ORDER BY revision"
        ).fetchall() == [(1,), (2,)]
    finally:
        connection.close()

    recovered = SQLiteMetadataRepository.open(
        database_path,
        migrations=MIGRATIONS[:2],
    )
    try:
        _assert_complete_graph(recovered)
    finally:
        recovered.close()


def test_public_factory_upgrades_failed_revision_two_without_losing_prior_data(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-metadata" / "metadata.sqlite3"
    source_dir = tmp_path / "private-source-data"
    _create_complete_baseline(database_path, migrations=MIGRATIONS[:2])
    revision_two_snapshot = _database_snapshot(database_path)

    with pytest.raises(MigrationRequiredError) as migration_required:
        open_metadata_repository(database_path, source_dir=source_dir)

    assert _database_snapshot(database_path) == revision_two_snapshot
    assert str(database_path) not in str(migration_required.value)
    assert str(source_dir) not in str(migration_required.value)

    with pytest.raises(StorageMigrationError):
        SQLiteMetadataRepository.open(
            database_path,
            apply_migrations=True,
            migrations=(*MIGRATIONS[:2], _failing_source_persistence_migration()),
            applied_at_factory=_fixed_applied_at,
        )

    assert _database_snapshot(database_path) == revision_two_snapshot

    migrated: MetadataRepository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    try:
        _assert_complete_graph(migrated)
        with migrated.transaction() as transaction:
            transaction.add_document(DocumentIdentity(DOCUMENT_ID))
            transaction.add_document_version(_SOURCE_VERSION)
            transaction.add_corpus_document_version_membership(_SOURCE_MEMBERSHIP)
        migrated.persist_source_artefact(
            _CORPUS_ID,
            _SOURCE_ARTEFACT,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
            parse_outcome=_SOURCE_OUTCOME,
            parsed_representation=parsed_representation(),
        )
    finally:
        migrated.close()

    reopened: MetadataRepository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
    )
    try:
        _assert_complete_graph(
            reopened,
            additional_memberships=(_SOURCE_MEMBERSHIP,),
        )
        assert (
            reopened.get_source_artefact(_CORPUS_ID, DOCUMENT_VERSION_ID)
            == _SOURCE_ARTEFACT
        )
        assert (
            reopened.read_source_bytes(_CORPUS_ID, DOCUMENT_VERSION_ID) == SOURCE_BYTES
        )
        source_stream = reopened.stream_source_bytes(
            _CORPUS_ID,
            DOCUMENT_VERSION_ID,
        )
        assert source_stream is not None
        assert b"".join(source_stream) == SOURCE_BYTES
        assert (
            reopened.get_parsed_representation(_CORPUS_ID, PARSED_RECORD_ID)
            == parsed_representation()
        )
        assert reopened.get_parse_outcome(_CORPUS_ID, PARSE_ID) == _SOURCE_OUTCOME
    finally:
        reopened.close()
