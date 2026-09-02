import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from inference_lab.provenance import (
    ContentHash,
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
from inference_lab.storage import CorruptStorageError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
_FAMILY_ID = DocumentFamilyId("ae06e397-6433-481d-bbe4-1d80250e8891")
_FIRST_DOCUMENT_ID = DocumentId("488759cb-869f-4790-bf9d-1504f9d30fb8")
_SECOND_DOCUMENT_ID = DocumentId("9bc9d26c-0408-4c8b-ab88-a93718847a16")
_FIRST_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_SECOND_VERSION_ID = DocumentVersionId("22222222-2222-4222-8222-222222222222")
_FIRST_VERSION = DocumentVersion(
    _FIRST_VERSION_ID,
    _FIRST_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
    (
        ExternalIdentifier("doi", "10.1000/first"),
        ExternalIdentifier("pmid", "12345678"),
    ),
)
_SECOND_VERSION = DocumentVersion(
    _SECOND_VERSION_ID,
    _SECOND_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "b" * 64),
)


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _create_populated_database(path: Path) -> None:
    repository = SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    with repository.transaction() as transaction:
        transaction.add_document_family(_FAMILY_ID)
        transaction.add_document(DocumentIdentity(_FIRST_DOCUMENT_ID))
        transaction.add_document(DocumentIdentity(_SECOND_DOCUMENT_ID))
        transaction.add_document_version(_FIRST_VERSION)
        transaction.add_document_version(_SECOND_VERSION)
        transaction.add_duplicate_relationship(
            DuplicateRelationship(
                _FIRST_VERSION_ID,
                _SECOND_VERSION_ID,
                DuplicateClass.BYTE_IDENTICAL,
            )
        )
        transaction.add_family_membership(
            DocumentFamilyMembership(
                _FAMILY_ID,
                _FIRST_DOCUMENT_ID,
                _FIRST_VERSION_ID,
            )
        )
    repository.close()


def _mutate_database(path: Path, statements: tuple[str, ...]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def test_reopen_rejects_dangling_foreign_keys_without_repairing_them(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_populated_database(database_path)
    _mutate_database(
        database_path,
        (
            """DELETE FROM documents
            WHERE document_id = '488759cb-869f-4790-bf9d-1504f9d30fb8'""",
        ),
    )

    with pytest.raises(CorruptStorageError):
        SQLiteMetadataRepository.open(database_path)

    connection = sqlite3.connect(database_path)
    try:
        assert (
            connection.execute(
                "SELECT 1 FROM documents WHERE document_id = ?",
                (_FIRST_DOCUMENT_ID.as_str(),),
            ).fetchone()
            is None
        )
        assert connection.execute(
            "SELECT 1 FROM document_versions WHERE document_version_id = ?",
            (_FIRST_VERSION_ID.as_str(),),
        ).fetchone() == (1,)
    finally:
        connection.close()


def test_reopen_rejects_family_version_ownership_corruption(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_populated_database(database_path)
    _mutate_database(
        database_path,
        (
            """UPDATE document_family_memberships
            SET document_id = '9bc9d26c-0408-4c8b-ab88-a93718847a16'""",
        ),
    )

    with pytest.raises(CorruptStorageError):
        SQLiteMetadataRepository.open(database_path)


def test_reopen_rejects_rows_violating_schema_checks(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_populated_database(database_path)
    _mutate_database(
        database_path,
        (
            "PRAGMA ignore_check_constraints = ON",
            """UPDATE duplicate_relationships
            SET first_version_id = '22222222-2222-4222-8222-222222222222',
                second_version_id = '11111111-1111-4111-8111-111111111111'""",
        ),
    )

    with pytest.raises(CorruptStorageError):
        SQLiteMetadataRepository.open(database_path)


def test_noncanonical_stored_identifier_order_fails_on_reconstruction(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_populated_database(database_path)
    _mutate_database(
        database_path,
        (
            "UPDATE external_identifiers SET ordinal = ordinal + 10",
            """UPDATE external_identifiers
            SET ordinal = CASE namespace WHEN 'doi' THEN 1 ELSE 0 END""",
        ),
    )

    repository = SQLiteMetadataRepository.open(database_path)
    try:
        with pytest.raises(CorruptStorageError):
            repository.get_document_version(_FIRST_VERSION_ID)
        with pytest.raises(CorruptStorageError):
            repository.find_document_versions_by_content_hash(
                _FIRST_VERSION.content_hash
            )
    finally:
        repository.close()
