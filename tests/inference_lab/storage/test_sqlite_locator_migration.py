"""Tests for the source-locator continuity schema migration."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.storage import MigrationRequiredError
from inference_lab.storage._sqlite_connection import open_sqlite_connection
from inference_lab.storage._sqlite_initialisation import initialise_empty_database
from inference_lab.storage._sqlite_migrations import MIGRATIONS
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import fixed_applied_at as _fixed_applied_at  # noqa: TID252

_CORPUS_ID = "11111111-1111-4111-8111-111111111111"
_OTHER_CORPUS_ID = "22222222-2222-4222-8222-222222222222"
_DOCUMENT_ID = "33333333-3333-4333-8333-333333333333"
_VERSION_ID = "44444444-4444-4444-8444-444444444444"
_SOURCE_ID = "55555555-5555-4555-8555-555555555555"
_OTHER_SOURCE_ID = "66666666-6666-4666-8666-666666666666"
_THIRD_SOURCE_ID = "77777777-7777-4777-8777-777777777777"
_DIGEST = "a" * 64


def _create_revision_one_database(path: Path) -> None:
    connection = open_sqlite_connection(path)
    initialise_empty_database(
        connection,
        migrations=(MIGRATIONS[0],),
        applied_at_factory=_fixed_applied_at,
    )
    try:
        connection.execute(
            "INSERT INTO corpora (corpus_id, display_name) VALUES (?, ?)",
            (_CORPUS_ID, "Private corpus"),
        )
        connection.execute(
            "INSERT INTO documents (document_id) VALUES (?)",
            (_DOCUMENT_ID,),
        )
        connection.execute(
            """INSERT INTO document_versions (
                document_version_id,
                document_id,
                content_hash_algorithm,
                content_hash_digest
            ) VALUES (?, ?, 'sha256', ?)""",
            (_VERSION_ID, _DOCUMENT_ID, _DIGEST),
        )
        connection.execute(
            """INSERT INTO corpus_document_version_memberships (
                corpus_id,
                document_id,
                document_version_id
            ) VALUES (?, ?, ?)""",
            (_CORPUS_ID, _DOCUMENT_ID, _VERSION_ID),
        )
        connection.execute(
            """INSERT INTO source_observations (
                corpus_id,
                source_observation_id,
                document_id,
                current_document_version_id,
                current_revision
            ) VALUES (?, ?, ?, ?, 1)""",
            (_CORPUS_ID, _SOURCE_ID, _DOCUMENT_ID, _VERSION_ID),
        )
        connection.execute(
            """INSERT INTO source_observation_history (
                corpus_id,
                source_observation_id,
                observation_revision,
                document_id,
                document_version_id
            ) VALUES (?, ?, 1, ?, ?)""",
            (_CORPUS_ID, _SOURCE_ID, _DOCUMENT_ID, _VERSION_ID),
        )
        connection.commit()
    finally:
        connection.close()


def _insert_source_observation(
    connection: sqlite3.Connection,
    *,
    corpus_id: str,
    source_id: str,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO corpora (corpus_id, display_name) VALUES (?, 'Corpus')",
        (corpus_id,),
    )
    connection.execute(
        """INSERT OR IGNORE INTO corpus_document_version_memberships (
            corpus_id, document_id, document_version_id
        ) VALUES (?, ?, ?)""",
        (corpus_id, _DOCUMENT_ID, _VERSION_ID),
    )
    connection.execute(
        """INSERT INTO source_observations (
            corpus_id,
            source_observation_id,
            document_id,
            current_document_version_id,
            current_revision
        ) VALUES (?, ?, ?, ?, 1)""",
        (corpus_id, source_id, _DOCUMENT_ID, _VERSION_ID),
    )


def _insert_revision(
    connection: sqlite3.Connection,
    *,
    corpus_id: str = _CORPUS_ID,
    source_id: str = _SOURCE_ID,
    revision: int = 1,
    path: str = "/private/source.txt",
    device_id: int | None = 7,
    file_id: int | None = 11,
) -> None:
    connection.execute(
        """INSERT INTO source_locator_revisions (
            corpus_id,
            source_observation_id,
            locator_revision,
            locator_path,
            device_id,
            file_id,
            content_hash_algorithm,
            content_hash_digest,
            byte_count,
            modified_time_ns
        ) VALUES (?, ?, ?, ?, ?, ?, 'sha256', ?, 42, 123456789)""",
        (corpus_id, source_id, revision, path, device_id, file_id, _DIGEST),
    )


def test_fresh_database_initialises_with_locator_continuity_schema(
    tmp_path: Path,
) -> None:
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert repository.schema_inspection.current_revision == 6
        connection = repository._connection_or_raise()
        tables = connection.execute(
            """SELECT name FROM sqlite_schema
            WHERE type = 'table' AND name LIKE '%source_locator%'
            ORDER BY name"""
        ).fetchall()
        assert [tuple(row) for row in tables] == [
            ("current_source_locators",),
            ("source_locator_revisions",),
        ]
    finally:
        repository.close()


def test_standard_upgrade_requires_permission_then_preserves_existing_records(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_revision_one_database(database_path)

    with pytest.raises(MigrationRequiredError):
        SQLiteMetadataRepository.open(database_path)

    migrated = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert migrated.schema_inspection.current_revision == 6
        connection = migrated._connection_or_raise()
        assert connection.execute("SELECT count(*) FROM corpora").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
        assert (
            connection.execute("SELECT count(*) FROM document_versions").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT count(*) FROM source_observations").fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM source_observation_history"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM source_locator_revisions"
            ).fetchone()[0]
            == 0
        )
    finally:
        migrated.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert reopened.schema_inspection.current_revision == 6
    finally:
        reopened.close()


def test_locator_schema_enforces_ownership_revision_and_complete_hints(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_revision_one_database(database_path)
    repository = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        connection = repository._connection_or_raise()

        with pytest.raises(sqlite3.IntegrityError):
            _insert_revision(connection, corpus_id=_OTHER_CORPUS_ID)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_revision(connection, revision=0)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_revision(connection, revision=2)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_revision(connection, device_id=7, file_id=None)

        _insert_revision(connection)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_revision(connection)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO current_source_locators (
                    corpus_id, source_observation_id, locator_revision, locator_path
                ) VALUES (?, ?, 2, '/private/missing.txt')""",
                (_CORPUS_ID, _SOURCE_ID),
            )
    finally:
        repository.close()


def test_locator_revision_history_is_append_only(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_revision_one_database(database_path)
    repository = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        connection = repository._connection_or_raise()
        _insert_revision(connection)

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE source_locator_revisions SET byte_count = 43")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM source_locator_revisions")

        row = connection.execute(
            """SELECT locator_revision, byte_count
            FROM source_locator_revisions"""
        ).fetchone()
        assert tuple(row) == (1, 42)
    finally:
        repository.close()


def test_current_locator_is_unique_within_a_corpus_but_not_across_corpora(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_revision_one_database(database_path)
    repository = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        connection = repository._connection_or_raise()
        _insert_revision(connection)
        connection.execute(
            """INSERT INTO current_source_locators (
                corpus_id, source_observation_id, locator_revision, locator_path
            ) VALUES (?, ?, 1, '/private/source.txt')""",
            (_CORPUS_ID, _SOURCE_ID),
        )

        _insert_source_observation(
            connection,
            corpus_id=_CORPUS_ID,
            source_id=_OTHER_SOURCE_ID,
        )
        _insert_revision(
            connection,
            source_id=_OTHER_SOURCE_ID,
            path="/private/source.txt",
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO current_source_locators (
                    corpus_id, source_observation_id, locator_revision, locator_path
                ) VALUES (?, ?, 1, '/private/source.txt')""",
                (_CORPUS_ID, _OTHER_SOURCE_ID),
            )

        _insert_source_observation(
            connection,
            corpus_id=_OTHER_CORPUS_ID,
            source_id=_THIRD_SOURCE_ID,
        )
        _insert_revision(
            connection,
            corpus_id=_OTHER_CORPUS_ID,
            source_id=_THIRD_SOURCE_ID,
            path="/private/source.txt",
        )
        connection.execute(
            """INSERT INTO current_source_locators (
                corpus_id, source_observation_id, locator_revision, locator_path
            ) VALUES (?, ?, 1, '/private/source.txt')""",
            (_OTHER_CORPUS_ID, _THIRD_SOURCE_ID),
        )
    finally:
        repository.close()
