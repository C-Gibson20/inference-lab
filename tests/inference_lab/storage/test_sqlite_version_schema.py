"""Tests for the durable corpus-version SQLite schema."""

import sqlite3
from collections.abc import Generator

import pytest

from inference_lab.storage._sqlite_migrations import MIGRATIONS

_CORPUS_A = "00000000-0000-0000-0000-000000000001"
_CORPUS_B = "00000000-0000-0000-0000-000000000002"
_VERSION_A1 = "00000000-0000-0000-0000-000000000011"
_VERSION_A2 = "00000000-0000-0000-0000-000000000012"
_VERSION_B1 = "00000000-0000-0000-0000-000000000021"
_DOCUMENT = "00000000-0000-0000-0000-000000000031"
_DOCUMENT_VERSION = "00000000-0000-0000-0000-000000000032"
_CREATED_AT = "2026-09-05T12:30:00.000000+00:00"
_DIGEST = "a" * 64


@pytest.fixture
def version_database() -> Generator[sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS:
        for statement in migration.statements:
            connection.execute(statement)
    try:
        yield connection
    finally:
        connection.close()


def _insert_corpus(connection: sqlite3.Connection, corpus_id: str) -> None:
    connection.execute(
        "INSERT INTO corpora (corpus_id, display_name) VALUES (?, 'Test corpus')",
        (corpus_id,),
    )


def _insert_update(
    connection: sqlite3.Connection,
    *,
    corpus_id: str,
    corpus_version: str,
    expected_corpus_version: str | None = None,
) -> None:
    connection.execute(
        """INSERT INTO corpus_updates (
            corpus_id,
            corpus_version,
            expected_corpus_version,
            state,
            manifest_digest,
            created_at,
            restores_corpus_version
        ) VALUES (?, ?, ?, 'preparing', NULL, ?, NULL)""",
        (corpus_id, corpus_version, expected_corpus_version, _CREATED_AT),
    )


def _seal_update(
    connection: sqlite3.Connection,
    *,
    corpus_id: str,
    corpus_version: str,
) -> None:
    connection.execute(
        """UPDATE corpus_updates
        SET state = 'ready', manifest_digest = ?
        WHERE corpus_id = ? AND corpus_version = ?""",
        (_DIGEST, corpus_id, corpus_version),
    )


def _insert_version(
    connection: sqlite3.Connection,
    *,
    corpus_id: str,
    corpus_version: str,
    sequence: int,
    parent_corpus_version: str | None = None,
) -> None:
    connection.execute(
        """INSERT INTO corpus_versions (
            corpus_id,
            corpus_version,
            sequence,
            parent_corpus_version,
            manifest_digest,
            created_at,
            restores_corpus_version
        ) VALUES (?, ?, ?, ?, ?, ?, NULL)""",
        (
            corpus_id,
            corpus_version,
            sequence,
            parent_corpus_version,
            _DIGEST,
            _CREATED_AT,
        ),
    )


def _publish_empty_version(
    connection: sqlite3.Connection,
    *,
    corpus_id: str,
    corpus_version: str,
    sequence: int,
    parent_corpus_version: str | None = None,
) -> None:
    _insert_update(
        connection,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        expected_corpus_version=parent_corpus_version,
    )
    _seal_update(
        connection,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
    )
    _insert_version(
        connection,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        sequence=sequence,
        parent_corpus_version=parent_corpus_version,
    )
    connection.execute(
        """UPDATE corpus_active_versions
        SET corpus_version = ?
        WHERE corpus_id = ?""",
        (corpus_version, corpus_id),
    )
    connection.execute(
        """UPDATE corpus_updates
        SET state = 'committed'
        WHERE corpus_id = ? AND corpus_version = ?""",
        (corpus_id, corpus_version),
    )


def test_revision_adds_complete_version_publication_schema(
    version_database: sqlite3.Connection,
) -> None:
    tables = {
        row[0]
        for row in version_database.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table'"
        )
    }
    assert {
        "corpus_active_versions",
        "corpus_update_component_generations",
        "corpus_update_document_memberships",
        "corpus_update_required_components",
        "corpus_updates",
        "corpus_version_document_memberships",
        "corpus_versions",
    } <= tables

    indexes = {
        row[0]
        for row in version_database.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'index'"
        )
    }
    assert {
        "corpus_active_versions_version_idx",
        "corpus_update_component_generations_state_idx",
        "corpus_update_document_memberships_version_idx",
        "corpus_updates_recovery_idx",
        "corpus_version_document_memberships_document_idx",
        "corpus_versions_history_idx",
    } <= indexes

    triggers = {
        row[0]
        for row in version_database.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'trigger'"
        )
    }
    assert {
        "corpus_active_versions_advance",
        "corpus_update_component_generations_update_guard",
        "corpus_update_document_memberships_update_guard",
        "corpus_update_required_components_update_guard",
        "corpus_updates_lifecycle_update",
        "corpus_version_document_memberships_update_immutable",
        "corpus_versions_update_immutable",
    } <= triggers


def test_new_corpus_receives_exactly_one_nullable_active_pointer(
    version_database: sqlite3.Connection,
) -> None:
    _insert_corpus(version_database, _CORPUS_A)

    assert version_database.execute(
        """SELECT corpus_id, corpus_version
        FROM corpus_active_versions
        WHERE corpus_id = ?""",
        (_CORPUS_A,),
    ).fetchone() == (_CORPUS_A, None)
    with pytest.raises(sqlite3.IntegrityError):
        version_database.execute(
            """INSERT INTO corpus_active_versions (corpus_id, corpus_version)
            VALUES (?, NULL)""",
            (_CORPUS_A,),
        )


def test_required_components_must_be_ready_before_sealing(
    version_database: sqlite3.Connection,
) -> None:
    _insert_corpus(version_database, _CORPUS_A)
    _insert_update(
        version_database,
        corpus_id=_CORPUS_A,
        corpus_version=_VERSION_A1,
    )
    version_database.execute(
        """INSERT INTO corpus_update_required_components (
            corpus_id, corpus_version, component_kind
        ) VALUES (?, ?, 'lexical')""",
        (_CORPUS_A, _VERSION_A1),
    )

    with pytest.raises(sqlite3.IntegrityError):
        _seal_update(
            version_database,
            corpus_id=_CORPUS_A,
            corpus_version=_VERSION_A1,
        )

    version_database.execute(
        """INSERT INTO corpus_update_component_generations (
            corpus_id,
            corpus_version,
            component_kind,
            generation_id,
            implementation,
            format_version,
            contract_version,
            configuration_digest,
            source_manifest_digest,
            state
        ) VALUES (?, ?, 'lexical', 'lexical-generation-1', 'fake-index',
                  '1.0.0', '1.0.0', ?, ?, 'ready')""",
        (_CORPUS_A, _VERSION_A1, _DIGEST, _DIGEST),
    )
    _seal_update(
        version_database,
        corpus_id=_CORPUS_A,
        corpus_version=_VERSION_A1,
    )

    for mutation in (
        """UPDATE corpus_update_component_generations
        SET generation_id = 'changed'
        WHERE corpus_version = ?""",
        """DELETE FROM corpus_update_required_components
        WHERE corpus_version = ?""",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            version_database.execute(mutation, (_VERSION_A1,))


def test_document_staging_requires_target_corpus_catalogue_membership(
    version_database: sqlite3.Connection,
) -> None:
    _insert_corpus(version_database, _CORPUS_A)
    _insert_corpus(version_database, _CORPUS_B)
    version_database.execute(
        "INSERT INTO documents (document_id) VALUES (?)",
        (_DOCUMENT,),
    )
    version_database.execute(
        """INSERT INTO document_versions (
            document_version_id,
            document_id,
            content_hash_algorithm,
            content_hash_digest
        ) VALUES (?, ?, 'sha256', ?)""",
        (_DOCUMENT_VERSION, _DOCUMENT, _DIGEST),
    )
    version_database.execute(
        """INSERT INTO corpus_document_version_memberships (
            corpus_id, document_id, document_version_id
        ) VALUES (?, ?, ?)""",
        (_CORPUS_A, _DOCUMENT, _DOCUMENT_VERSION),
    )
    _insert_update(
        version_database,
        corpus_id=_CORPUS_B,
        corpus_version=_VERSION_B1,
    )

    with pytest.raises(sqlite3.IntegrityError):
        version_database.execute(
            """INSERT INTO corpus_update_document_memberships (
                corpus_id, corpus_version, document_id, document_version_id
            ) VALUES (?, ?, ?, ?)""",
            (_CORPUS_B, _VERSION_B1, _DOCUMENT, _DOCUMENT_VERSION),
        )


def test_version_ownership_sequence_and_active_pointer_are_monotonic(
    version_database: sqlite3.Connection,
) -> None:
    _insert_corpus(version_database, _CORPUS_A)
    _insert_corpus(version_database, _CORPUS_B)
    _publish_empty_version(
        version_database,
        corpus_id=_CORPUS_A,
        corpus_version=_VERSION_A1,
        sequence=1,
    )

    with pytest.raises(sqlite3.IntegrityError):
        _insert_update(
            version_database,
            corpus_id=_CORPUS_B,
            corpus_version=_VERSION_B1,
            expected_corpus_version=_VERSION_A1,
        )
    with pytest.raises(sqlite3.IntegrityError):
        _insert_update(
            version_database,
            corpus_id=_CORPUS_B,
            corpus_version=_VERSION_A1,
        )

    _insert_update(
        version_database,
        corpus_id=_CORPUS_A,
        corpus_version=_VERSION_A2,
        expected_corpus_version=_VERSION_A1,
    )
    _seal_update(
        version_database,
        corpus_id=_CORPUS_A,
        corpus_version=_VERSION_A2,
    )
    with pytest.raises(sqlite3.IntegrityError):
        _insert_version(
            version_database,
            corpus_id=_CORPUS_A,
            corpus_version=_VERSION_A2,
            sequence=3,
            parent_corpus_version=_VERSION_A1,
        )
    _insert_version(
        version_database,
        corpus_id=_CORPUS_A,
        corpus_version=_VERSION_A2,
        sequence=2,
        parent_corpus_version=_VERSION_A1,
    )
    version_database.execute(
        "UPDATE corpus_active_versions SET corpus_version = ? WHERE corpus_id = ?",
        (_VERSION_A2, _CORPUS_A),
    )

    with pytest.raises(sqlite3.IntegrityError):
        version_database.execute(
            "UPDATE corpus_active_versions SET corpus_version = ? WHERE corpus_id = ?",
            (_VERSION_A1, _CORPUS_A),
        )


def test_committed_versions_and_memberships_are_immutable(
    version_database: sqlite3.Connection,
) -> None:
    _insert_corpus(version_database, _CORPUS_A)
    version_database.execute(
        "INSERT INTO documents (document_id) VALUES (?)",
        (_DOCUMENT,),
    )
    version_database.execute(
        """INSERT INTO document_versions (
            document_version_id,
            document_id,
            content_hash_algorithm,
            content_hash_digest
        ) VALUES (?, ?, 'sha256', ?)""",
        (_DOCUMENT_VERSION, _DOCUMENT, _DIGEST),
    )
    version_database.execute(
        """INSERT INTO corpus_document_version_memberships (
            corpus_id, document_id, document_version_id
        ) VALUES (?, ?, ?)""",
        (_CORPUS_A, _DOCUMENT, _DOCUMENT_VERSION),
    )
    _insert_update(
        version_database,
        corpus_id=_CORPUS_A,
        corpus_version=_VERSION_A1,
    )
    version_database.execute(
        """INSERT INTO corpus_update_document_memberships (
            corpus_id, corpus_version, document_id, document_version_id
        ) VALUES (?, ?, ?, ?)""",
        (_CORPUS_A, _VERSION_A1, _DOCUMENT, _DOCUMENT_VERSION),
    )
    _seal_update(
        version_database,
        corpus_id=_CORPUS_A,
        corpus_version=_VERSION_A1,
    )
    _insert_version(
        version_database,
        corpus_id=_CORPUS_A,
        corpus_version=_VERSION_A1,
        sequence=1,
    )
    version_database.execute(
        """INSERT INTO corpus_version_document_memberships (
            corpus_id, corpus_version, document_id, document_version_id
        ) VALUES (?, ?, ?, ?)""",
        (_CORPUS_A, _VERSION_A1, _DOCUMENT, _DOCUMENT_VERSION),
    )
    version_database.execute(
        "UPDATE corpus_active_versions SET corpus_version = ? WHERE corpus_id = ?",
        (_VERSION_A1, _CORPUS_A),
    )
    version_database.execute(
        """UPDATE corpus_updates SET state = 'committed'
        WHERE corpus_id = ? AND corpus_version = ?""",
        (_CORPUS_A, _VERSION_A1),
    )

    mutations = (
        (
            "UPDATE corpus_versions SET manifest_digest = ? WHERE corpus_version = ?",
            ("b" * 64, _VERSION_A1),
        ),
        (
            "DELETE FROM corpus_version_document_memberships WHERE corpus_version = ?",
            (_VERSION_A1,),
        ),
        (
            "UPDATE corpus_updates SET state = 'aborted' WHERE corpus_version = ?",
            (_VERSION_A1,),
        ),
    )
    for statement, parameters in mutations:
        with pytest.raises(sqlite3.IntegrityError):
            version_database.execute(statement, parameters)

    assert version_database.execute(
        """SELECT version_record.sequence, active.corpus_version, update_record.state
        FROM corpus_versions AS version_record
        JOIN corpus_active_versions AS active
            ON active.corpus_id = version_record.corpus_id
        JOIN corpus_updates AS update_record
            ON update_record.corpus_version = version_record.corpus_version
        WHERE version_record.corpus_version = ?""",
        (_VERSION_A1,),
    ).fetchone() == (1, _VERSION_A1, "committed")
