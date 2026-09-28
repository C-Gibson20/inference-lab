"""Focused registry contract for safe, contiguous schema migrations."""

import sqlite3

import pytest

from inference_lab.storage._sqlite_migrations import (
    MIGRATIONS,
    MigrationDefinition,
    validate_migration_registry,
)

_DOCUMENT_ID = "00000000-0000-0000-0000-000000000001"
_OTHER_DOCUMENT_ID = "00000000-0000-0000-0000-000000000002"
_VERSION_ID = "00000000-0000-0000-0000-000000000003"
_OTHER_VERSION_ID = "00000000-0000-0000-0000-000000000004"
_FAMILY_ID = "00000000-0000-0000-0000-000000000005"
_DIGEST = "a" * 64


def _current_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS:
        for statement in migration.statements:
            connection.execute(statement)
    return connection


def _insert_document(connection: sqlite3.Connection, document_id: str) -> None:
    connection.execute(
        "INSERT INTO documents (document_id) VALUES (?)",
        (document_id,),
    )


def _insert_version(
    connection: sqlite3.Connection,
    *,
    version_id: str,
    document_id: str,
    digest: str = _DIGEST,
) -> None:
    connection.execute(
        """INSERT INTO document_versions (
            document_version_id, document_id,
            content_hash_algorithm, content_hash_digest
        ) VALUES (?, ?, 'sha256', ?)""",
        (version_id, document_id, digest),
    )


def test_external_identifiers_are_ordered_and_unique_per_version() -> None:
    connection = _current_database()
    try:
        _insert_document(connection, _DOCUMENT_ID)
        _insert_version(
            connection,
            version_id=_VERSION_ID,
            document_id=_DOCUMENT_ID,
        )
        statement = """INSERT INTO external_identifiers (
            document_version_id, ordinal, namespace, identifier_value
        ) VALUES (?, ?, ?, ?)"""
        connection.execute(statement, (_VERSION_ID, 0, "doi", "10.1/example"))
        connection.execute(statement, (_VERSION_ID, 1, "pmid", "12345678"))

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(statement, (_VERSION_ID, 0, "patent", "GB1"))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(statement, (_VERSION_ID, 2, "doi", "10.1/example"))

        assert connection.execute(
            """SELECT ordinal, namespace FROM external_identifiers
            WHERE document_version_id = ? ORDER BY ordinal""",
            (_VERSION_ID,),
        ).fetchall() == [(0, "doi"), (1, "pmid")]
    finally:
        connection.close()


def test_duplicate_and_family_constraints_reject_invalid_graph_writes() -> None:
    connection = _current_database()
    try:
        _insert_document(connection, _DOCUMENT_ID)
        _insert_document(connection, _OTHER_DOCUMENT_ID)
        _insert_version(
            connection,
            version_id=_VERSION_ID,
            document_id=_DOCUMENT_ID,
        )
        _insert_version(
            connection,
            version_id=_OTHER_VERSION_ID,
            document_id=_DOCUMENT_ID,
            digest="b" * 64,
        )
        _insert_version(
            connection,
            version_id="00000000-0000-0000-0000-000000000006",
            document_id=_OTHER_DOCUMENT_ID,
            digest="b" * 64,
        )
        duplicate = """INSERT INTO duplicate_relationships (
            duplicate_class, first_version_id, second_version_id
        ) VALUES ('byte_identical', ?, ?)"""

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(duplicate, (_VERSION_ID, _VERSION_ID))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(duplicate, (_OTHER_VERSION_ID, _VERSION_ID))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                duplicate,
                (_VERSION_ID, "00000000-0000-0000-0000-000000000006"),
            )

        connection.execute(
            "INSERT INTO document_families (family_id) VALUES (?)",
            (_FAMILY_ID,),
        )
        family_membership = """INSERT INTO document_family_memberships (
            family_id, document_id, document_version_id
        ) VALUES (?, ?, ?)"""
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                family_membership,
                (_FAMILY_ID, _OTHER_DOCUMENT_ID, _VERSION_ID),
            )

        document_membership = """INSERT INTO document_family_memberships (
            family_id, document_id, document_version_id
        ) VALUES (?, ?, NULL)"""
        connection.execute(document_membership, (_FAMILY_ID, _DOCUMENT_ID))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(document_membership, (_FAMILY_ID, _DOCUMENT_ID))
    finally:
        connection.close()


def test_migration_registry_rejects_unsafe_gapped_and_empty_definitions() -> None:
    with pytest.raises(ValueError, match="transaction control"):
        MigrationDefinition(
            revision=2,
            migration_id="unsafe_migration",
            statements=("BEGIN",),
        )

    gapped = MigrationDefinition(
        revision=MIGRATIONS[-1].revision + 2,
        migration_id="gapped_migration",
        statements=("SELECT 1",),
    )
    with pytest.raises(ValueError, match="contiguous"):
        validate_migration_registry((*MIGRATIONS, gapped))

    with pytest.raises(ValueError, match="must not be empty"):
        validate_migration_registry(())
