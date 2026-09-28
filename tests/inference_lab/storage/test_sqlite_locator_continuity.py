"""Acceptance tests for private SQLite source-locator continuity."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage import (
    CorruptStorageError,
    SourceFileIdentityHint,
    SourceFingerprint,
    SourceLocatorRevision,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import open_repository as _open_repository  # noqa: TID252

_FIRST_CORPUS = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"),
    "First locator corpus",
)
_SECOND_CORPUS = CorpusMetadata(
    CorpusId("22222222-2222-4222-8222-222222222222"),
    "Second locator corpus",
)
_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_VERSION = DocumentVersion(
    DocumentVersionId("44444444-4444-4444-8444-444444444444"),
    _DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_FIRST_SOURCE_ID = SourceObservationId("55555555-5555-4555-8555-555555555555")
_SECOND_SOURCE_ID = SourceObservationId("66666666-6666-4666-8666-666666666666")
_THIRD_SOURCE_ID = SourceObservationId("77777777-7777-4777-8777-777777777777")
_SHARED_HINT = SourceFileIdentityHint(device_id=7, file_id=11)


def _add_catalogue(repository: SQLiteMetadataRepository) -> None:
    """Add two corpus catalogues and three independent observations."""
    with repository.transaction() as transaction:
        transaction.add_corpus(_FIRST_CORPUS)
        transaction.add_corpus(_SECOND_CORPUS)
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(_VERSION)
        for corpus in (_FIRST_CORPUS, _SECOND_CORPUS):
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    corpus.corpus_id,
                    _DOCUMENT_ID,
                    _VERSION.document_version_id,
                )
            )
        for corpus, source_id in (
            (_FIRST_CORPUS, _FIRST_SOURCE_ID),
            (_FIRST_CORPUS, _SECOND_SOURCE_ID),
            (_SECOND_CORPUS, _THIRD_SOURCE_ID),
        ):
            transaction.add_source_observation(
                SourceObservationState(
                    corpus.corpus_id,
                    source_id,
                    1,
                    _DOCUMENT_ID,
                    _VERSION.document_version_id,
                )
            )


def _revision(
    *,
    corpus_id: CorpusId = _FIRST_CORPUS.corpus_id,
    source_id: SourceObservationId = _FIRST_SOURCE_ID,
    locator_revision: int = 1,
    path: Path,
    identity_hint: SourceFileIdentityHint | None = _SHARED_HINT,
    digest_character: str = "a",
    byte_count: int = 42,
    modified_time_ns: int = 123_456_789,
) -> SourceLocatorRevision:
    return SourceLocatorRevision(
        corpus_id=corpus_id,
        source_observation_id=source_id,
        locator_revision=locator_revision,
        path=path,
        identity_hint=identity_hint,
        fingerprint=SourceFingerprint(
            ContentHash(HashAlgorithm.SHA256, digest_character * 64),
            byte_count,
            modified_time_ns,
        ),
    )


def _install_trigger(path: Path, statement: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def _set_current_locator(
    database_path: Path,
    *,
    locator_revision: int,
    locator_path: Path,
) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """UPDATE current_source_locators
            SET locator_revision = ?, locator_path = ?""",
            (locator_revision, str(locator_path)),
        )
        connection.commit()
    finally:
        connection.close()


def test_stale_or_conflicting_locator_writes_roll_back_without_partial_history(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    first_path = (tmp_path / "first.txt").resolve()
    first = _revision(path=first_path)
    conflicting_initial = _revision(path=(tmp_path / "different.txt").resolve())
    skipped_revision = _revision(
        locator_revision=3,
        path=(tmp_path / "skipped.txt").resolve(),
    )
    reused_path = _revision(
        source_id=_SECOND_SOURCE_ID,
        path=first_path,
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_source_locator(first)

        for operation in (
            lambda transaction: transaction.add_source_locator(conflicting_initial),
            lambda transaction: transaction.advance_source_locator(skipped_revision),
            lambda transaction: transaction.add_source_locator(reused_path),
        ):
            with pytest.raises(StorageConflictError):
                with repository.transaction() as transaction:
                    operation(transaction)

        assert repository.list_source_locator_history(
            _FIRST_CORPUS.corpus_id,
            _FIRST_SOURCE_ID,
        ) == (first,)
        assert (
            repository.get_source_locator(
                _FIRST_CORPUS.corpus_id,
                _SECOND_SOURCE_ID,
            )
            is None
        )
        assert (
            repository.list_source_locator_history(
                _FIRST_CORPUS.corpus_id,
                _SECOND_SOURCE_ID,
            )
            == ()
        )
    finally:
        repository.close()


def test_wrong_corpus_or_unknown_observation_cannot_receive_a_locator(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    wrong_corpus = _revision(
        corpus_id=_SECOND_CORPUS.corpus_id,
        source_id=_FIRST_SOURCE_ID,
        path=(tmp_path / "wrong-corpus.txt").resolve(),
    )
    unknown_observation = _revision(
        source_id=SourceObservationId("88888888-8888-4888-8888-888888888888"),
        path=(tmp_path / "unknown.txt").resolve(),
    )
    try:
        for revision in (wrong_corpus, unknown_observation):
            with pytest.raises(StorageConflictError):
                with repository.transaction() as transaction:
                    transaction.add_source_locator(revision)
        assert (
            repository.find_source_locators_by_identity_hint(
                _FIRST_CORPUS.corpus_id,
                _SHARED_HINT,
            )
            == ()
        )
        assert (
            repository.find_source_locators_by_identity_hint(
                _SECOND_CORPUS.corpus_id,
                _SHARED_HINT,
            )
            == ()
        )
    finally:
        repository.close()


def test_stale_current_pointer_is_reported_as_corrupt_storage(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    first = _revision(path=(tmp_path / "first.txt").resolve())
    second = _revision(
        locator_revision=2,
        path=(tmp_path / "second.txt").resolve(),
    )
    with repository.transaction() as transaction:
        transaction.add_source_locator(first)
        transaction.advance_source_locator(second)
    repository.close()
    _set_current_locator(
        database_path,
        locator_revision=1,
        locator_path=first.path,
    )

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        with pytest.raises(CorruptStorageError):
            reopened.get_source_locator(
                _FIRST_CORPUS.corpus_id,
                _FIRST_SOURCE_ID,
            )
        with pytest.raises(CorruptStorageError):
            reopened.find_source_locator_by_path(
                _FIRST_CORPUS.corpus_id,
                first.path,
            )
        with pytest.raises(CorruptStorageError):
            reopened.find_source_locators_by_identity_hint(
                _FIRST_CORPUS.corpus_id,
                _SHARED_HINT,
            )
    finally:
        reopened.close()


def test_missing_current_pointer_is_reported_as_corrupt_storage(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    revision = _revision(path=(tmp_path / "source.txt").resolve())
    with repository.transaction() as transaction:
        transaction.add_source_locator(revision)
    repository.close()

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DELETE FROM current_source_locators")
        connection.commit()
    finally:
        connection.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        with pytest.raises(CorruptStorageError):
            reopened.get_source_locator(
                _FIRST_CORPUS.corpus_id,
                _FIRST_SOURCE_ID,
            )
    finally:
        reopened.close()
