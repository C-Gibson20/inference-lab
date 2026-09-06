"""Tests for SQLite repository concurrency."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import perf_counter

import pytest
from structlog.testing import capture_logs

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
from inference_lab.storage import (
    CommitCorpusUpdateOutcome,
    CorpusUpdate,
    CorpusUpdateState,
    SealCorpusUpdateOutcome,
    StorageConcurrencyError,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import fixed_applied_at as _fixed_applied_at  # noqa: TID252

_BASELINE_CORPUS_ID = CorpusId("08b00c09-b9c0-44b7-a3a4-1acb526162ab")
_BASELINE_CORPUS = CorpusMetadata(_BASELINE_CORPUS_ID, "Baseline corpus")
_UNCOMMITTED_DOCUMENT_ID = DocumentId("ce118577-f665-419f-9b4a-c9f875a59718")
_AFTER_LOCK_CORPUS_ID = CorpusId("c2a62ed4-de89-4b4c-a5e2-a8e6340e2d81")
_AFTER_LOCK_CORPUS = CorpusMetadata(_AFTER_LOCK_CORPUS_ID, "After-lock corpus")
_CREATED_AT = datetime(2026, 9, 6, 16, 0, tzinfo=UTC)
_PARENT_VERSION = CorpusVersionId("11111111-1111-4111-8111-111111111111")
_WINNING_VERSION = CorpusVersionId("22222222-2222-4222-8222-222222222222")
_LOSING_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_OTHER_PARENT_VERSION = CorpusVersionId("14141414-1414-4414-8414-141414141414")
_OTHER_CHILD_VERSION = CorpusVersionId("15151515-1515-4515-8515-151515151515")
_WINNING_DOCUMENT_ID = DocumentId("44444444-4444-4444-8444-444444444444")
_LOSING_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_WINNING_DOCUMENT_VERSION = DocumentVersion(
    DocumentVersionId("66666666-6666-4666-8666-666666666666"),
    _WINNING_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_LOSING_DOCUMENT_VERSION = DocumentVersion(
    DocumentVersionId("77777777-7777-4777-8777-777777777777"),
    _LOSING_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "b" * 64),
)
_WINNING_MEMBERSHIP = CorpusDocumentVersionMembership(
    _BASELINE_CORPUS_ID,
    _WINNING_DOCUMENT_ID,
    _WINNING_DOCUMENT_VERSION.document_version_id,
)
_LOSING_MEMBERSHIP = CorpusDocumentVersionMembership(
    _BASELINE_CORPUS_ID,
    _LOSING_DOCUMENT_ID,
    _LOSING_DOCUMENT_VERSION.document_version_id,
)
_OTHER_MEMBERSHIP = CorpusDocumentVersionMembership(
    _AFTER_LOCK_CORPUS_ID,
    _LOSING_DOCUMENT_ID,
    _LOSING_DOCUMENT_VERSION.document_version_id,
)


def _initialise_repository(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )


def _fixed_created_at() -> datetime:
    return _CREATED_AT


def _prepare_same_parent_race(path: Path) -> CommitCorpusUpdateOutcome:
    repository = SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
        created_at_factory=_fixed_created_at,
    )
    parent: CommitCorpusUpdateOutcome | None = None
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_BASELINE_CORPUS)
            for document_id, version, membership in (
                (
                    _WINNING_DOCUMENT_ID,
                    _WINNING_DOCUMENT_VERSION,
                    _WINNING_MEMBERSHIP,
                ),
                (
                    _LOSING_DOCUMENT_ID,
                    _LOSING_DOCUMENT_VERSION,
                    _LOSING_MEMBERSHIP,
                ),
            ):
                transaction.add_document(DocumentIdentity(document_id))
                transaction.add_document_version(version)
                transaction.add_corpus_document_version_membership(membership)

        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _BASELINE_CORPUS_ID,
                _PARENT_VERSION,
                expected_corpus_version=None,
            )
            sealed = transaction.seal_corpus_update(
                _BASELINE_CORPUS_ID,
                _PARENT_VERSION,
            )
            assert isinstance(sealed, SealCorpusUpdateOutcome)
            parent = transaction.commit_corpus_update(
                _BASELINE_CORPUS_ID,
                _PARENT_VERSION,
            )

        for corpus_version, membership in (
            (_WINNING_VERSION, _WINNING_MEMBERSHIP),
            (_LOSING_VERSION, _LOSING_MEMBERSHIP),
        ):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    _BASELINE_CORPUS_ID,
                    corpus_version,
                    expected_corpus_version=_PARENT_VERSION,
                )
                transaction.add_corpus_update_document_membership(
                    _BASELINE_CORPUS_ID,
                    corpus_version,
                    membership,
                )
                sealed = transaction.seal_corpus_update(
                    _BASELINE_CORPUS_ID,
                    corpus_version,
                )
                assert isinstance(sealed, SealCorpusUpdateOutcome)
    finally:
        repository.close()

    assert parent is not None
    return parent


def _prepare_different_corpus_commits(
    path: Path,
) -> tuple[CommitCorpusUpdateOutcome, CommitCorpusUpdateOutcome]:
    repository = SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
        created_at_factory=_fixed_created_at,
    )
    parents: list[CommitCorpusUpdateOutcome] = []
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_BASELINE_CORPUS)
            transaction.add_corpus(_AFTER_LOCK_CORPUS)
            for document_id, version in (
                (_WINNING_DOCUMENT_ID, _WINNING_DOCUMENT_VERSION),
                (_LOSING_DOCUMENT_ID, _LOSING_DOCUMENT_VERSION),
            ):
                transaction.add_document(DocumentIdentity(document_id))
                transaction.add_document_version(version)
            transaction.add_corpus_document_version_membership(_WINNING_MEMBERSHIP)
            transaction.add_corpus_document_version_membership(_OTHER_MEMBERSHIP)

        for corpus_id, corpus_version in (
            (_BASELINE_CORPUS_ID, _PARENT_VERSION),
            (_AFTER_LOCK_CORPUS_ID, _OTHER_PARENT_VERSION),
        ):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    corpus_id,
                    corpus_version,
                    expected_corpus_version=None,
                )
                sealed = transaction.seal_corpus_update(corpus_id, corpus_version)
                assert isinstance(sealed, SealCorpusUpdateOutcome)
                parents.append(
                    transaction.commit_corpus_update(corpus_id, corpus_version)
                )

        for corpus_id, corpus_version, parent_version, membership in (
            (
                _BASELINE_CORPUS_ID,
                _WINNING_VERSION,
                _PARENT_VERSION,
                _WINNING_MEMBERSHIP,
            ),
            (
                _AFTER_LOCK_CORPUS_ID,
                _OTHER_CHILD_VERSION,
                _OTHER_PARENT_VERSION,
                _OTHER_MEMBERSHIP,
            ),
        ):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    corpus_id,
                    corpus_version,
                    expected_corpus_version=parent_version,
                )
                transaction.add_corpus_update_document_membership(
                    corpus_id,
                    corpus_version,
                    membership,
                )
                sealed = transaction.seal_corpus_update(corpus_id, corpus_version)
                assert isinstance(sealed, SealCorpusUpdateOutcome)
    finally:
        repository.close()

    assert len(parents) == 2
    return parents[0], parents[1]


def _journal_mode(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        return row[0]
    finally:
        connection.close()


def test_repository_enables_durable_wal_across_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _initialise_repository(database_path)
    try:
        assert _journal_mode(database_path) == "wal"
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert _journal_mode(database_path) == "wal"
    finally:
        reopened.close()


def test_reader_continues_with_committed_state_during_write(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    writer = _initialise_repository(database_path)
    with writer.transaction() as transaction:
        transaction.add_corpus(_BASELINE_CORPUS)
    reader = SQLiteMetadataRepository.open(database_path)
    try:
        with writer.transaction() as transaction:
            transaction.add_document(DocumentIdentity(_UNCOMMITTED_DOCUMENT_ID))

            started_at = perf_counter()
            assert reader.get_corpus(_BASELINE_CORPUS_ID) == _BASELINE_CORPUS
            assert reader.get_document(_UNCOMMITTED_DOCUMENT_ID) is None
            assert perf_counter() - started_at < 0.5

        assert reader.get_document(_UNCOMMITTED_DOCUMENT_ID) == DocumentIdentity(
            _UNCOMMITTED_DOCUMENT_ID
        )
    finally:
        reader.close()
        writer.close()


def test_second_writer_times_out_safely_and_recovers_after_commit(
    tmp_path: Path,
) -> None:
    private_marker = "private-lock-path"
    database_path = tmp_path / private_marker / "metadata.sqlite3"
    writer = _initialise_repository(database_path)
    contender = SQLiteMetadataRepository.open(database_path, busy_timeout_ms=25)
    try:
        with writer.transaction() as transaction:
            transaction.add_corpus(_BASELINE_CORPUS)

            with capture_logs() as logs:
                started_at = perf_counter()
                with pytest.raises(StorageConcurrencyError) as raised:
                    with contender.transaction():
                        pass
                elapsed = perf_counter() - started_at

            assert elapsed < 1.0
            assert raised.value.retryable is True
            assert private_marker not in str(raised.value)
            assert "locked" not in str(raised.value).lower()
            assert logs == [
                {
                    "event": "transaction_failed",
                    "pipeline_stage": "storage",
                    "error_category": "concurrency",
                    "retryable": True,
                    "log_level": "error",
                }
            ]
            assert private_marker not in repr(logs)

        with contender.transaction() as transaction:
            transaction.add_corpus(_AFTER_LOCK_CORPUS)

        assert contender.get_corpus(_BASELINE_CORPUS_ID) == _BASELINE_CORPUS
        assert contender.get_corpus(_AFTER_LOCK_CORPUS_ID) == _AFTER_LOCK_CORPUS
    finally:
        contender.close()
        writer.close()


def test_same_parent_writers_publish_exactly_one_complete_version(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-race" / "metadata.sqlite3"
    parent = _prepare_same_parent_race(database_path)
    winner = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=_fixed_created_at,
    )
    contender_ready = Event()
    start_contender = Event()
    first_attempt_finished = Event()
    winner_committed = Event()

    def contend() -> tuple[
        StorageConcurrencyError,
        StorageConflictError,
        CorpusUpdate,
    ]:
        repository = SQLiteMetadataRepository.open(
            database_path,
            busy_timeout_ms=25,
            created_at_factory=_fixed_created_at,
        )
        lock_error: StorageConcurrencyError | None = None
        stale_error: StorageConflictError | None = None
        losing_update: CorpusUpdate | None = None
        try:
            contender_ready.set()
            if not start_contender.wait(timeout=5):
                raise AssertionError("winner did not start the contender")
            try:
                with repository.transaction() as transaction:
                    transaction.commit_corpus_update(
                        _BASELINE_CORPUS_ID,
                        _LOSING_VERSION,
                    )
            except StorageConcurrencyError as error:
                lock_error = error
            finally:
                first_attempt_finished.set()

            if not winner_committed.wait(timeout=5):
                raise AssertionError("winner did not finish its commit")
            try:
                with repository.transaction() as transaction:
                    transaction.commit_corpus_update(
                        _BASELINE_CORPUS_ID,
                        _LOSING_VERSION,
                    )
            except StorageConflictError as error:
                stale_error = error
            losing_update = repository.get_corpus_update(
                _BASELINE_CORPUS_ID,
                _LOSING_VERSION,
            )
        finally:
            repository.close()

        assert lock_error is not None
        assert stale_error is not None
        assert losing_update is not None
        return lock_error, stale_error, losing_update

    winning_publication: CommitCorpusUpdateOutcome | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(contend)
            assert contender_ready.wait(timeout=5)
            try:
                with winner.transaction() as transaction:
                    start_contender.set()
                    assert first_attempt_finished.wait(timeout=5)
                    winning_publication = transaction.commit_corpus_update(
                        _BASELINE_CORPUS_ID,
                        _WINNING_VERSION,
                    )
            finally:
                winner_committed.set()
            lock_error, stale_error, losing_update = future.result(timeout=5)

        assert winning_publication is not None
        assert winning_publication.version.sequence == 2
        assert winning_publication.version.parent_corpus_version == _PARENT_VERSION
        assert winning_publication.manifest.document_memberships == (
            _WINNING_MEMBERSHIP,
        )
        assert lock_error.retryable is True
        assert stale_error.retryable is False
        assert losing_update.state is CorpusUpdateState.READY
        assert winner.get_active_corpus_version(_BASELINE_CORPUS_ID) == (
            winning_publication.version
        )
        assert winner.list_corpus_versions(_BASELINE_CORPUS_ID) == (
            parent.version,
            winning_publication.version,
        )
        assert (
            winner.get_corpus_version(
                _BASELINE_CORPUS_ID,
                _LOSING_VERSION,
            )
            is None
        )
        assert winner.list_corpus_update_document_memberships(
            _BASELINE_CORPUS_ID,
            _LOSING_VERSION,
        ) == (_LOSING_MEMBERSHIP,)
    finally:
        winner.close()

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            """SELECT sequence, corpus_version FROM corpus_versions
            WHERE corpus_id = ? ORDER BY sequence""",
            (_BASELINE_CORPUS_ID.as_str(),),
        ).fetchall() == [
            (1, _PARENT_VERSION.as_str()),
            (2, _WINNING_VERSION.as_str()),
        ]
        assert connection.execute(
            """SELECT count(*) FROM corpus_version_document_memberships
            WHERE corpus_id = ? AND corpus_version = ?""",
            (_BASELINE_CORPUS_ID.as_str(), _LOSING_VERSION.as_str()),
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_different_corpus_writers_both_publish_after_bounded_lock_retry(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-independent" / "metadata.sqlite3"
    first_parent, second_parent = _prepare_different_corpus_commits(database_path)
    first_writer = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=_fixed_created_at,
    )
    contender_ready = Event()
    start_contender = Event()
    first_attempt_finished = Event()
    first_writer_committed = Event()

    def commit_second_corpus() -> tuple[
        StorageConcurrencyError,
        CommitCorpusUpdateOutcome,
    ]:
        repository = SQLiteMetadataRepository.open(
            database_path,
            busy_timeout_ms=25,
            created_at_factory=_fixed_created_at,
        )
        lock_error: StorageConcurrencyError | None = None
        publication: CommitCorpusUpdateOutcome | None = None
        try:
            contender_ready.set()
            if not start_contender.wait(timeout=5):
                raise AssertionError("first writer did not start the contender")
            try:
                with repository.transaction() as transaction:
                    transaction.commit_corpus_update(
                        _AFTER_LOCK_CORPUS_ID,
                        _OTHER_CHILD_VERSION,
                    )
            except StorageConcurrencyError as error:
                lock_error = error
            finally:
                first_attempt_finished.set()

            if not first_writer_committed.wait(timeout=5):
                raise AssertionError("first writer did not finish its commit")
            with repository.transaction() as transaction:
                publication = transaction.commit_corpus_update(
                    _AFTER_LOCK_CORPUS_ID,
                    _OTHER_CHILD_VERSION,
                )
        finally:
            repository.close()

        assert lock_error is not None
        assert publication is not None
        return lock_error, publication

    first_publication: CommitCorpusUpdateOutcome | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(commit_second_corpus)
            assert contender_ready.wait(timeout=5)
            try:
                with first_writer.transaction() as transaction:
                    start_contender.set()
                    assert first_attempt_finished.wait(timeout=5)
                    first_publication = transaction.commit_corpus_update(
                        _BASELINE_CORPUS_ID,
                        _WINNING_VERSION,
                    )
            finally:
                first_writer_committed.set()
            lock_error, second_publication = future.result(timeout=5)

        assert first_publication is not None
        assert lock_error.retryable is True
        assert first_publication.version.sequence == 2
        assert first_publication.version.parent_corpus_version == _PARENT_VERSION
        assert first_publication.manifest.document_memberships == (_WINNING_MEMBERSHIP,)
        assert second_publication.replayed is False
        assert second_publication.version.sequence == 2
        assert second_publication.version.parent_corpus_version == (
            _OTHER_PARENT_VERSION
        )
        assert second_publication.manifest.document_memberships == (_OTHER_MEMBERSHIP,)

        assert first_writer.get_active_corpus_version(_BASELINE_CORPUS_ID) == (
            first_publication.version
        )
        assert first_writer.get_active_corpus_version(_AFTER_LOCK_CORPUS_ID) == (
            second_publication.version
        )
        assert first_writer.list_corpus_versions(_BASELINE_CORPUS_ID) == (
            first_parent.version,
            first_publication.version,
        )
        assert first_writer.list_corpus_versions(_AFTER_LOCK_CORPUS_ID) == (
            second_parent.version,
            second_publication.version,
        )
    finally:
        first_writer.close()

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """SELECT corpus_id, corpus_version, document_id
            FROM corpus_version_document_memberships
            WHERE corpus_version IN (?, ?)
            ORDER BY corpus_id""",
            (_WINNING_VERSION.as_str(), _OTHER_CHILD_VERSION.as_str()),
        ).fetchall()
        assert rows == [
            (
                _BASELINE_CORPUS_ID.as_str(),
                _WINNING_VERSION.as_str(),
                _WINNING_DOCUMENT_ID.as_str(),
            ),
            (
                _AFTER_LOCK_CORPUS_ID.as_str(),
                _OTHER_CHILD_VERSION.as_str(),
                _LOSING_DOCUMENT_ID.as_str(),
            ),
        ]
    finally:
        connection.close()
