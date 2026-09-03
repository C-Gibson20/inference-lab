"""Tests for the SQLite migration registry and schema."""

import sqlite3
from collections.abc import Generator

import pytest

from inference_lab.storage._sqlite_migrations import (
    MIGRATIONS,
    SQLITE_APPLICATION_ID,
    MigrationDefinition,
    validate_migration_registry,
)

_DOCUMENT_ID = "00000000-0000-0000-0000-000000000001"
_OTHER_DOCUMENT_ID = "00000000-0000-0000-0000-000000000002"
_VERSION_ID = "00000000-0000-0000-0000-000000000003"
_OTHER_VERSION_ID = "00000000-0000-0000-0000-000000000004"
_FAMILY_ID = "00000000-0000-0000-0000-000000000005"
_DIGEST = "a" * 64


@pytest.fixture
def baseline_database() -> Generator[sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for statement in MIGRATIONS[0].statements:
        connection.execute(statement)
    try:
        yield connection
    finally:
        connection.close()


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
            document_version_id,
            document_id,
            content_hash_algorithm,
            content_hash_digest
        ) VALUES (?, ?, 'sha256', ?)""",
        (version_id, document_id, digest),
    )


def test_multi_corpus_baseline_is_stable_and_registry_is_valid() -> None:
    assert SQLITE_APPLICATION_ID == 0x494C4142
    assert validate_migration_registry(MIGRATIONS) is MIGRATIONS
    assert len(MIGRATIONS) == 1
    assert MIGRATIONS[0].revision == 1
    assert MIGRATIONS[0].migration_id == "multi_corpus_metadata_baseline"
    assert MIGRATIONS[0].checksum == (
        "7516eeb4654186443550d64d7c61f5eb541e944c24c3f3d978017442d651d0bc"
    )


def test_baseline_creates_the_complete_current_metadata_scope(
    baseline_database: sqlite3.Connection,
) -> None:
    rows = baseline_database.execute(
        """SELECT name
        FROM sqlite_schema
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name"""
    ).fetchall()

    assert rows == [
        ("corpora",),
        ("corpus_document_version_memberships",),
        ("document_families",),
        ("document_family_memberships",),
        ("document_versions",),
        ("documents",),
        ("duplicate_relationships",),
        ("external_identifier_assertions",),
        ("external_identifiers",),
        ("metadata_schema",),
        ("migration_history",),
        ("source_observation_history",),
        ("source_observations",),
    ]
    assert baseline_database.execute(
        "SELECT schema_name, current_revision FROM metadata_schema"
    ).fetchone() == ("inference_lab_metadata", 0)


@pytest.mark.parametrize(
    "invalid_id",
    [
        "00000000-0000-0000-0000-00000000000A",
        "000000000000-0000-0000-000000000001",
        "0000000--0000-0000-0000-000000000001",
        "not-a-document-identifier",
    ],
)
def test_identity_columns_reject_noncanonical_values(
    baseline_database: sqlite3.Connection,
    invalid_id: str,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert_document(baseline_database, invalid_id)


def test_document_versions_require_an_existing_document(
    baseline_database: sqlite3.Connection,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert_version(
            baseline_database,
            version_id=_VERSION_ID,
            document_id=_DOCUMENT_ID,
        )


def test_content_hashes_are_indexed_but_not_unique(
    baseline_database: sqlite3.Connection,
) -> None:
    _insert_document(baseline_database, _DOCUMENT_ID)
    _insert_version(
        baseline_database,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    _insert_version(
        baseline_database,
        version_id=_OTHER_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )

    rows = baseline_database.execute(
        """SELECT document_version_id
        FROM document_versions
        WHERE content_hash_algorithm = 'sha256' AND content_hash_digest = ?
        ORDER BY document_version_id""",
        (_DIGEST,),
    ).fetchall()
    assert rows == [(_VERSION_ID,), (_OTHER_VERSION_ID,)]


def test_external_identifiers_enforce_order_and_per_version_uniqueness(
    baseline_database: sqlite3.Connection,
) -> None:
    _insert_document(baseline_database, _DOCUMENT_ID)
    _insert_version(
        baseline_database,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    baseline_database.execute(
        """INSERT INTO external_identifiers (
            document_version_id, ordinal, namespace, identifier_value
        ) VALUES (?, 0, 'doi', '10.1/example')""",
        (_VERSION_ID,),
    )

    with pytest.raises(sqlite3.IntegrityError):
        baseline_database.execute(
            """INSERT INTO external_identifiers (
                document_version_id, ordinal, namespace, identifier_value
            ) VALUES (?, 0, 'patent', 'GB1')""",
            (_VERSION_ID,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        baseline_database.execute(
            """INSERT INTO external_identifiers (
                document_version_id, ordinal, namespace, identifier_value
            ) VALUES (?, 1, 'doi', '10.1/example')""",
            (_VERSION_ID,),
        )


def test_duplicate_relationships_require_canonical_distinct_endpoints(
    baseline_database: sqlite3.Connection,
) -> None:
    _insert_document(baseline_database, _DOCUMENT_ID)
    _insert_version(
        baseline_database,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    _insert_version(
        baseline_database,
        version_id=_OTHER_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )

    with pytest.raises(sqlite3.IntegrityError):
        baseline_database.execute(
            """INSERT INTO duplicate_relationships (
                duplicate_class, first_version_id, second_version_id
            ) VALUES ('byte_identical', ?, ?)""",
            (_VERSION_ID, _VERSION_ID),
        )
    with pytest.raises(sqlite3.IntegrityError):
        baseline_database.execute(
            """INSERT INTO duplicate_relationships (
                duplicate_class, first_version_id, second_version_id
            ) VALUES ('byte_identical', ?, ?)""",
            (_OTHER_VERSION_ID, _VERSION_ID),
        )


def test_duplicate_hash_triggers_reject_mismatched_writes() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS:
        for statement in migration.statements:
            connection.execute(statement)
    _insert_document(connection, _DOCUMENT_ID)
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

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """INSERT INTO duplicate_relationships (
                duplicate_class, first_version_id, second_version_id
            ) VALUES ('byte_identical', ?, ?)""",
            (_VERSION_ID, _OTHER_VERSION_ID),
        )

    connection.execute(
        """UPDATE document_versions
        SET content_hash_digest = ?
        WHERE document_version_id = ?""",
        (_DIGEST, _OTHER_VERSION_ID),
    )
    connection.execute(
        """INSERT INTO duplicate_relationships (
            duplicate_class, first_version_id, second_version_id
        ) VALUES ('byte_identical', ?, ?)""",
        (_VERSION_ID, _OTHER_VERSION_ID),
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """UPDATE document_versions
            SET content_hash_digest = ?
            WHERE document_version_id = ?""",
            ("b" * 64, _OTHER_VERSION_ID),
        )

    assert connection.execute(
        """SELECT content_hash_digest
        FROM document_versions
        WHERE document_version_id = ?""",
        (_OTHER_VERSION_ID,),
    ).fetchone() == (_DIGEST,)
    connection.close()


def test_version_membership_must_belong_to_the_same_document(
    baseline_database: sqlite3.Connection,
) -> None:
    _insert_document(baseline_database, _DOCUMENT_ID)
    _insert_document(baseline_database, _OTHER_DOCUMENT_ID)
    _insert_version(
        baseline_database,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    baseline_database.execute(
        "INSERT INTO document_families (family_id) VALUES (?)",
        (_FAMILY_ID,),
    )

    with pytest.raises(sqlite3.IntegrityError):
        baseline_database.execute(
            """INSERT INTO document_family_memberships (
                family_id, document_id, document_version_id
            ) VALUES (?, ?, ?)""",
            (_FAMILY_ID, _OTHER_DOCUMENT_ID, _VERSION_ID),
        )


def test_document_level_membership_key_is_unique_when_version_is_null(
    baseline_database: sqlite3.Connection,
) -> None:
    _insert_document(baseline_database, _DOCUMENT_ID)
    baseline_database.execute(
        "INSERT INTO document_families (family_id) VALUES (?)",
        (_FAMILY_ID,),
    )
    statement = """INSERT INTO document_family_memberships (
        family_id, document_id, document_version_id
    ) VALUES (?, ?, NULL)"""
    baseline_database.execute(statement, (_FAMILY_ID, _DOCUMENT_ID))

    with pytest.raises(sqlite3.IntegrityError):
        baseline_database.execute(statement, (_FAMILY_ID, _DOCUMENT_ID))


@pytest.mark.parametrize(
    "bad_statements",
    [
        (),
        (" BEGIN",),
        ("COMMIT",),
    ],
)
def test_migration_definitions_reject_unsafe_statements(
    bad_statements: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        MigrationDefinition(
            revision=2,
            migration_id="unsafe_migration",
            statements=bad_statements,
        )


def test_registry_rejects_revision_gaps() -> None:
    second = MigrationDefinition(
        revision=3,
        migration_id="future_migration",
        statements=("SELECT 1",),
    )

    with pytest.raises(ValueError, match="contiguous"):
        validate_migration_registry((MIGRATIONS[0], second))


def test_registry_must_contain_an_initial_migration() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_migration_registry(())
