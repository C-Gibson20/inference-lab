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
_CORPUS_ID = "00000000-0000-0000-0000-000000000006"
_OTHER_CORPUS_ID = "00000000-0000-0000-0000-000000000007"
_SOURCE_OBSERVATION_ID = "00000000-0000-0000-0000-000000000008"
_DIGEST = "a" * 64


@pytest.fixture
def revision_one_database() -> Generator[sqlite3.Connection]:
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


def test_revision_one_is_stable_and_registry_is_valid() -> None:
    assert SQLITE_APPLICATION_ID == 0x494C4142
    assert validate_migration_registry(MIGRATIONS) is MIGRATIONS
    assert len(MIGRATIONS) == 5
    assert MIGRATIONS[0].revision == 1
    assert MIGRATIONS[0].migration_id == "initial_metadata"
    assert MIGRATIONS[0].checksum == (
        "bd5e54ddf8a3964ae426d97c904f1d00f9a23f2307642b9538aa0cc7335016d1"
    )
    assert MIGRATIONS[1].revision == 2
    assert MIGRATIONS[1].migration_id == "source_observation_identity"
    assert MIGRATIONS[1].checksum == (
        "2cedb3bd9dc35c6f679f500926ba37774bfb7a0b4a80420c194b76745045e2fc"
    )
    assert MIGRATIONS[2].revision == 3
    assert MIGRATIONS[2].migration_id == "duplicate_hash_integrity"
    assert MIGRATIONS[2].checksum == (
        "5341fdcb642e8868be679df0930f8fe0604b9eca67e71333847eaba94f63b95b"
    )
    assert MIGRATIONS[3].revision == 4
    assert MIGRATIONS[3].migration_id == "corpus_catalogue_membership"
    assert MIGRATIONS[3].checksum == (
        "aa9e6a65702f728b0d5f67af695417681732a2fe20542f0d7e16e135d9bbd0db"
    )
    assert MIGRATIONS[4].revision == 5
    assert MIGRATIONS[4].migration_id == "corpus_owned_source_observations"
    assert MIGRATIONS[4].checksum == (
        "2081ebb00c9f7f903a190dda256369c56d9d19b0322d13fa4ca3c912c3a060ca"
    )


def test_revision_four_backfills_unambiguous_legacy_catalogue() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS[:3]:
        for statement in migration.statements:
            connection.execute(statement)
    connection.execute("INSERT INTO corpora (corpus_id) VALUES (?)", (_CORPUS_ID,))
    _insert_document(connection, _DOCUMENT_ID)
    _insert_version(
        connection,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )

    for statement in MIGRATIONS[3].statements:
        connection.execute(statement)

    assert connection.execute(
        "SELECT corpus_id, display_name FROM corpora"
    ).fetchall() == [(_CORPUS_ID, _CORPUS_ID)]
    assert connection.execute(
        """SELECT corpus_id, document_id, document_version_id
        FROM corpus_document_version_memberships"""
    ).fetchall() == [(_CORPUS_ID, _DOCUMENT_ID, _VERSION_ID)]
    connection.close()


def test_revision_four_rejects_ambiguous_legacy_catalogue_atomically() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS[:3]:
        for statement in migration.statements:
            connection.execute(statement)
    connection.executemany(
        "INSERT INTO corpora (corpus_id) VALUES (?)",
        ((_CORPUS_ID,), (_OTHER_CORPUS_ID,)),
    )
    _insert_document(connection, _DOCUMENT_ID)
    _insert_version(
        connection,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    connection.commit()

    connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.IntegrityError):
        for statement in MIGRATIONS[3].statements:
            connection.execute(statement)
    connection.rollback()

    columns = connection.execute("PRAGMA table_info(corpora)").fetchall()
    assert [column[1] for column in columns] == ["storage_id", "corpus_id"]
    assert (
        connection.execute(
            """SELECT 1 FROM sqlite_schema
        WHERE type = 'table' AND name = 'corpus_document_version_memberships'"""
        ).fetchone()
        is None
    )
    connection.close()


def test_revision_four_allows_multiple_empty_legacy_corpora() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS[:3]:
        for statement in migration.statements:
            connection.execute(statement)
    connection.executemany(
        "INSERT INTO corpora (corpus_id) VALUES (?)",
        ((_CORPUS_ID,), (_OTHER_CORPUS_ID,)),
    )

    for statement in MIGRATIONS[3].statements:
        connection.execute(statement)

    assert connection.execute(
        "SELECT corpus_id, display_name FROM corpora ORDER BY corpus_id"
    ).fetchall() == [
        (_CORPUS_ID, _CORPUS_ID),
        (_OTHER_CORPUS_ID, _OTHER_CORPUS_ID),
    ]
    connection.close()


def test_revision_five_backfills_unambiguous_source_ownership() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS[:3]:
        for statement in migration.statements:
            connection.execute(statement)
    connection.execute("INSERT INTO corpora (corpus_id) VALUES (?)", (_CORPUS_ID,))
    _insert_document(connection, _DOCUMENT_ID)
    _insert_version(
        connection,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    connection.execute(
        """INSERT INTO source_observations (
            source_observation_id, document_id,
            current_document_version_id, current_revision
        ) VALUES (?, ?, ?, 1)""",
        (_SOURCE_OBSERVATION_ID, _DOCUMENT_ID, _VERSION_ID),
    )
    connection.execute(
        """INSERT INTO source_observation_history (
            source_observation_id, observation_revision,
            document_id, document_version_id
        ) VALUES (?, 1, ?, ?)""",
        (_SOURCE_OBSERVATION_ID, _DOCUMENT_ID, _VERSION_ID),
    )
    connection.execute(
        """INSERT INTO external_identifier_assertions (
            source_observation_id, document_version_id, document_id,
            namespace, identifier_value
        ) VALUES (?, ?, ?, 'doi', '10.1000/migrated')""",
        (_SOURCE_OBSERVATION_ID, _VERSION_ID, _DOCUMENT_ID),
    )
    for migration in MIGRATIONS[3:]:
        for statement in migration.statements:
            connection.execute(statement)

    assert connection.execute(
        """SELECT corpus_id, source_observation_id,
            document_id, current_document_version_id
        FROM source_observations"""
    ).fetchall() == [(_CORPUS_ID, _SOURCE_OBSERVATION_ID, _DOCUMENT_ID, _VERSION_ID)]
    assert connection.execute(
        """SELECT corpus_id, source_observation_id,
            document_id, document_version_id
        FROM source_observation_history"""
    ).fetchall() == [(_CORPUS_ID, _SOURCE_OBSERVATION_ID, _DOCUMENT_ID, _VERSION_ID)]
    assert connection.execute(
        """SELECT corpus_id, source_observation_id,
            document_version_id, namespace, identifier_value
        FROM external_identifier_assertions"""
    ).fetchall() == [
        (
            _CORPUS_ID,
            _SOURCE_OBSERVATION_ID,
            _VERSION_ID,
            "doi",
            "10.1000/migrated",
        )
    ]
    connection.close()


def test_revision_five_rejects_ambiguous_source_ownership_atomically() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS[:4]:
        for statement in migration.statements:
            connection.execute(statement)
    connection.executemany(
        "INSERT INTO corpora (corpus_id, display_name) VALUES (?, ?)",
        ((_CORPUS_ID, "First"), (_OTHER_CORPUS_ID, "Second")),
    )
    _insert_document(connection, _DOCUMENT_ID)
    _insert_version(
        connection,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    connection.execute(
        """INSERT INTO source_observations (
            source_observation_id, document_id,
            current_document_version_id, current_revision
        ) VALUES (?, ?, ?, 1)""",
        (_SOURCE_OBSERVATION_ID, _DOCUMENT_ID, _VERSION_ID),
    )
    connection.execute(
        """INSERT INTO source_observation_history (
            source_observation_id, observation_revision,
            document_id, document_version_id
        ) VALUES (?, 1, ?, ?)""",
        (_SOURCE_OBSERVATION_ID, _DOCUMENT_ID, _VERSION_ID),
    )
    connection.commit()

    connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.IntegrityError):
        for statement in MIGRATIONS[4].statements:
            connection.execute(statement)
    connection.rollback()

    columns = connection.execute("PRAGMA table_info(source_observations)").fetchall()
    assert "corpus_id" not in [column[1] for column in columns]
    assert connection.execute(
        "SELECT count(*) FROM source_observations"
    ).fetchone() == (1,)
    connection.close()


def test_revision_one_creates_only_the_initial_metadata_scope(
    revision_one_database: sqlite3.Connection,
) -> None:
    rows = revision_one_database.execute(
        """SELECT name
        FROM sqlite_schema
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name"""
    ).fetchall()

    assert rows == [
        ("corpora",),
        ("document_families",),
        ("document_family_memberships",),
        ("document_versions",),
        ("documents",),
        ("duplicate_relationships",),
        ("external_identifiers",),
        ("metadata_schema",),
        ("migration_history",),
    ]
    assert revision_one_database.execute(
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
    revision_one_database: sqlite3.Connection,
    invalid_id: str,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert_document(revision_one_database, invalid_id)


def test_document_versions_require_an_existing_document(
    revision_one_database: sqlite3.Connection,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert_version(
            revision_one_database,
            version_id=_VERSION_ID,
            document_id=_DOCUMENT_ID,
        )


def test_content_hashes_are_indexed_but_not_unique(
    revision_one_database: sqlite3.Connection,
) -> None:
    _insert_document(revision_one_database, _DOCUMENT_ID)
    _insert_version(
        revision_one_database,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    _insert_version(
        revision_one_database,
        version_id=_OTHER_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )

    rows = revision_one_database.execute(
        """SELECT document_version_id
        FROM document_versions
        WHERE content_hash_algorithm = 'sha256' AND content_hash_digest = ?
        ORDER BY document_version_id""",
        (_DIGEST,),
    ).fetchall()
    assert rows == [(_VERSION_ID,), (_OTHER_VERSION_ID,)]


def test_external_identifiers_enforce_order_and_per_version_uniqueness(
    revision_one_database: sqlite3.Connection,
) -> None:
    _insert_document(revision_one_database, _DOCUMENT_ID)
    _insert_version(
        revision_one_database,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    revision_one_database.execute(
        """INSERT INTO external_identifiers (
            document_version_id, ordinal, namespace, identifier_value
        ) VALUES (?, 0, 'doi', '10.1/example')""",
        (_VERSION_ID,),
    )

    with pytest.raises(sqlite3.IntegrityError):
        revision_one_database.execute(
            """INSERT INTO external_identifiers (
                document_version_id, ordinal, namespace, identifier_value
            ) VALUES (?, 0, 'patent', 'GB1')""",
            (_VERSION_ID,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        revision_one_database.execute(
            """INSERT INTO external_identifiers (
                document_version_id, ordinal, namespace, identifier_value
            ) VALUES (?, 1, 'doi', '10.1/example')""",
            (_VERSION_ID,),
        )


def test_duplicate_relationships_require_canonical_distinct_endpoints(
    revision_one_database: sqlite3.Connection,
) -> None:
    _insert_document(revision_one_database, _DOCUMENT_ID)
    _insert_version(
        revision_one_database,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    _insert_version(
        revision_one_database,
        version_id=_OTHER_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )

    with pytest.raises(sqlite3.IntegrityError):
        revision_one_database.execute(
            """INSERT INTO duplicate_relationships (
                duplicate_class, first_version_id, second_version_id
            ) VALUES ('byte_identical', ?, ?)""",
            (_VERSION_ID, _VERSION_ID),
        )
    with pytest.raises(sqlite3.IntegrityError):
        revision_one_database.execute(
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


def test_revision_three_rejects_preexisting_mismatched_relationships() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS[:2]:
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
    connection.execute(
        """INSERT INTO duplicate_relationships (
            duplicate_class, first_version_id, second_version_id
        ) VALUES ('byte_identical', ?, ?)""",
        (_VERSION_ID, _OTHER_VERSION_ID),
    )
    connection.commit()

    connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.IntegrityError):
        for statement in MIGRATIONS[2].statements:
            connection.execute(statement)
    connection.rollback()

    assert (
        connection.execute(
            """SELECT name
        FROM sqlite_schema
        WHERE type = 'trigger' AND name LIKE '%duplicate_hash%'"""
        ).fetchall()
        == []
    )
    connection.close()


def test_version_membership_must_belong_to_the_same_document(
    revision_one_database: sqlite3.Connection,
) -> None:
    _insert_document(revision_one_database, _DOCUMENT_ID)
    _insert_document(revision_one_database, _OTHER_DOCUMENT_ID)
    _insert_version(
        revision_one_database,
        version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
    )
    revision_one_database.execute(
        "INSERT INTO document_families (family_id) VALUES (?)",
        (_FAMILY_ID,),
    )

    with pytest.raises(sqlite3.IntegrityError):
        revision_one_database.execute(
            """INSERT INTO document_family_memberships (
                family_id, document_id, document_version_id
            ) VALUES (?, ?, ?)""",
            (_FAMILY_ID, _OTHER_DOCUMENT_ID, _VERSION_ID),
        )


def test_document_level_membership_key_is_unique_when_version_is_null(
    revision_one_database: sqlite3.Connection,
) -> None:
    _insert_document(revision_one_database, _DOCUMENT_ID)
    revision_one_database.execute(
        "INSERT INTO document_families (family_id) VALUES (?)",
        (_FAMILY_ID,),
    )
    statement = """INSERT INTO document_family_memberships (
        family_id, document_id, document_version_id
    ) VALUES (?, ?, NULL)"""
    revision_one_database.execute(statement, (_FAMILY_ID, _DOCUMENT_ID))

    with pytest.raises(sqlite3.IntegrityError):
        revision_one_database.execute(statement, (_FAMILY_ID, _DOCUMENT_ID))


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
