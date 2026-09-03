"""Tests for SQLite document-version persistence."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.provenance import (
    ContentHash,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    ExternalIdentifier,
    HashAlgorithm,
)
from inference_lab.storage import CorruptStorageError, StorageConflictError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import open_repository as _open_repository  # noqa: TID252

_DOCUMENT_ID = DocumentId("37d4f985-0237-4f7c-a37a-360c873e259e")
_FIRST_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_SECOND_VERSION_ID = DocumentVersionId("99999999-9999-4999-8999-999999999999")


def _version(
    version_id: DocumentVersionId,
    *,
    digest: str = "a" * 64,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
) -> DocumentVersion:
    return DocumentVersion(
        document_version_id=version_id,
        document_id=_DOCUMENT_ID,
        content_hash=ContentHash(HashAlgorithm.SHA256, digest),
        external_identifiers=external_identifiers,
    )


def test_document_version_round_trips_exactly_after_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    expected = _version(
        _FIRST_VERSION_ID,
        external_identifiers=(
            ExternalIdentifier("doi", "10.1000/private-value"),
            ExternalIdentifier("pmid", "12345678"),
        ),
    )
    first = _open_repository(database_path)
    with first.transaction() as transaction:
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(expected)
    first.close()

    second = SQLiteMetadataRepository.open(database_path)
    try:
        stored = second.get_document_version(_FIRST_VERSION_ID)

        assert stored == expected
        assert stored is not None
        assert isinstance(stored.document_version_id, DocumentVersionId)
        assert not hasattr(stored, "row_id")
        assert not hasattr(stored, "storage_id")
    finally:
        second.close()


def test_versions_list_in_canonical_id_order_regardless_of_insert_order(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    first = _version(_FIRST_VERSION_ID)
    second = _version(_SECOND_VERSION_ID, digest="b" * 64)
    try:
        with repository.transaction() as transaction:
            transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
            transaction.add_document_version(second)
            transaction.add_document_version(first)

        assert repository.list_document_versions(_DOCUMENT_ID) == (first, second)
        assert (
            repository.list_document_versions(
                DocumentId("0fc12573-b8f9-438c-b915-8e3ac06bc6c0")
            )
            == ()
        )
    finally:
        repository.close()


def test_exact_document_version_replay_is_idempotent(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    expected = _version(
        _FIRST_VERSION_ID,
        external_identifiers=(ExternalIdentifier("doi", "10.1000/value"),),
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
            transaction.add_document_version(expected)
            transaction.add_document_version(expected)

        assert repository.get_document_version(_FIRST_VERSION_ID) == expected
    finally:
        repository.close()


def test_reusing_version_identity_with_different_data_is_a_conflict(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    original = _version(_FIRST_VERSION_ID)
    conflicting = _version(_FIRST_VERSION_ID, digest="b" * 64)
    try:
        with repository.transaction() as transaction:
            transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
            transaction.add_document_version(original)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_document_version(conflicting)

        assert repository.get_document_version(_FIRST_VERSION_ID) == original
    finally:
        repository.close()


def test_version_requires_an_existing_logical_document(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_document_version(_version(_FIRST_VERSION_ID))

        assert repository.get_document_version(_FIRST_VERSION_ID) is None
    finally:
        repository.close()


def test_external_identifier_failure_rolls_back_the_version_row(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    raw_connection = sqlite3.connect(database_path)
    try:
        raw_connection.execute(
            """CREATE TRIGGER reject_second_external_identifier
            BEFORE INSERT ON external_identifiers
            WHEN NEW.ordinal = 1
            BEGIN
                SELECT RAISE(ABORT, 'private-database-payload');
            END"""
        )
        raw_connection.commit()
    finally:
        raw_connection.close()

    expected = _version(
        _FIRST_VERSION_ID,
        external_identifiers=(
            ExternalIdentifier("doi", "10.1000/value"),
            ExternalIdentifier("pmid", "12345678"),
        ),
    )
    try:
        with pytest.raises(StorageConflictError) as raised:
            with repository.transaction() as transaction:
                transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
                transaction.add_document_version(expected)

        assert "private-database-payload" not in str(raised.value)
        assert repository.get_document(_DOCUMENT_ID) is None
        assert repository.get_document_version(_FIRST_VERSION_ID) is None
    finally:
        repository.close()


def test_non_contiguous_stored_identifier_order_fails_closed(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    expected = _version(
        _FIRST_VERSION_ID,
        external_identifiers=(ExternalIdentifier("doi", "10.1000/value"),),
    )
    repository = _open_repository(database_path)
    with repository.transaction() as transaction:
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(expected)
    repository.close()

    raw_connection = sqlite3.connect(database_path)
    try:
        raw_connection.execute("UPDATE external_identifiers SET ordinal = 4")
        raw_connection.commit()
    finally:
        raw_connection.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        with pytest.raises(CorruptStorageError):
            reopened.get_document_version(_FIRST_VERSION_ID)
    finally:
        reopened.close()
