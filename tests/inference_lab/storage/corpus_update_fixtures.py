"""Shared domain fixtures for SQLite corpus-update behavior tests."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
)
from inference_lab.storage import ComponentKind
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CREATED_AT = datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
_CREATED_AT_TEXT = "2026-09-05T12:30:00.000000+00:00"
_CORPUS_A = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"),
    "Private first corpus",
)
_CORPUS_B = CorpusMetadata(
    CorpusId("22222222-2222-4222-8222-222222222222"),
    "Private second corpus",
)
_PARENT_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_PROPOSED_VERSION = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_OTHER_VERSION = CorpusVersionId("55555555-5555-4555-8555-555555555555")
_FIRST_DOCUMENT_ID = DocumentId("66666666-6666-4666-8666-666666666666")
_SECOND_DOCUMENT_ID = DocumentId("77777777-7777-4777-8777-777777777777")
_FIRST_DOCUMENT_VERSION_ID = DocumentVersionId("88888888-8888-4888-8888-888888888888")
_SECOND_DOCUMENT_VERSION_ID = DocumentVersionId("99999999-9999-4999-8999-999999999999")
_REPLACEMENT_VERSION_ID = DocumentVersionId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_FIRST_DOCUMENT_VERSION = DocumentVersion(
    _FIRST_DOCUMENT_VERSION_ID,
    _FIRST_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_SECOND_DOCUMENT_VERSION = DocumentVersion(
    _SECOND_DOCUMENT_VERSION_ID,
    _SECOND_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "b" * 64),
)
_REPLACEMENT_VERSION = DocumentVersion(
    _REPLACEMENT_VERSION_ID,
    _FIRST_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "d" * 64),
)
_FIRST_MEMBERSHIP = CorpusDocumentVersionMembership(
    _CORPUS_A.corpus_id,
    _FIRST_DOCUMENT_ID,
    _FIRST_DOCUMENT_VERSION_ID,
)
_SECOND_MEMBERSHIP = CorpusDocumentVersionMembership(
    _CORPUS_A.corpus_id,
    _SECOND_DOCUMENT_ID,
    _SECOND_DOCUMENT_VERSION_ID,
)
_REPLACEMENT_MEMBERSHIP = CorpusDocumentVersionMembership(
    _CORPUS_A.corpus_id,
    _FIRST_DOCUMENT_ID,
    _REPLACEMENT_VERSION_ID,
)
_DENSE = ComponentKind("dense")
_LEXICAL = ComponentKind("lexical")
_MANIFEST_DIGEST = "c" * 64


def _fixed_created_at() -> datetime:
    return _CREATED_AT


def _open(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        created_at_factory=_fixed_created_at,
    )


def _add_catalogue(repository: SQLiteMetadataRepository) -> None:
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS_A)
        transaction.add_corpus(_CORPUS_B)
        for document_id, version in (
            (_FIRST_DOCUMENT_ID, _FIRST_DOCUMENT_VERSION),
            (_SECOND_DOCUMENT_ID, _SECOND_DOCUMENT_VERSION),
        ):
            transaction.add_document(DocumentIdentity(document_id))
            transaction.add_document_version(version)
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _CORPUS_A.corpus_id,
                    document_id,
                    version.document_version_id,
                )
            )
        transaction.add_document_version(_REPLACEMENT_VERSION)
        transaction.add_corpus_document_version_membership(_REPLACEMENT_MEMBERSHIP)


def _create_committed_parent(path: Path) -> None:
    repository = _open(path)
    _add_catalogue(repository)
    repository.close()

    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.execute(
            """INSERT INTO corpus_updates (
                corpus_id,
                corpus_version,
                expected_corpus_version,
                state,
                manifest_digest,
                created_at,
                restores_corpus_version
            ) VALUES (?, ?, NULL, 'ready', ?, ?, NULL)""",
            (
                _CORPUS_A.corpus_id.as_str(),
                _PARENT_VERSION.as_str(),
                _MANIFEST_DIGEST,
                _CREATED_AT_TEXT,
            ),
        )
        connection.execute(
            """INSERT INTO corpus_versions (
                corpus_id,
                corpus_version,
                sequence,
                parent_corpus_version,
                manifest_digest,
                created_at,
                restores_corpus_version
            ) VALUES (?, ?, 1, NULL, ?, ?, NULL)""",
            (
                _CORPUS_A.corpus_id.as_str(),
                _PARENT_VERSION.as_str(),
                _MANIFEST_DIGEST,
                _CREATED_AT_TEXT,
            ),
        )
        connection.executemany(
            """INSERT INTO corpus_version_document_memberships (
                corpus_id, corpus_version, document_id, document_version_id
            ) VALUES (?, ?, ?, ?)""",
            (
                (
                    membership.corpus_id.as_str(),
                    _PARENT_VERSION.as_str(),
                    membership.document_id.as_str(),
                    membership.document_version_id.as_str(),
                )
                for membership in (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP)
            ),
        )
        connection.execute(
            """UPDATE corpus_active_versions
            SET corpus_version = ?
            WHERE corpus_id = ?""",
            (_PARENT_VERSION.as_str(), _CORPUS_A.corpus_id.as_str()),
        )
        connection.execute(
            """UPDATE corpus_updates SET state = 'committed'
            WHERE corpus_id = ? AND corpus_version = ?""",
            (_CORPUS_A.corpus_id.as_str(), _PARENT_VERSION.as_str()),
        )
        connection.commit()
    finally:
        connection.close()
