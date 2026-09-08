"""Tests for aborting durable SQLite corpus updates."""

import sqlite3
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage import (
    AbortCorpusUpdateOutcome,
    CorpusUpdateState,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.storage.corpus_update_fixtures import (
    _CORPUS_A,
    _CORPUS_B,
    _FIRST_DOCUMENT_ID,
    _FIRST_MEMBERSHIP,
    _OTHER_VERSION,
    _PARENT_VERSION,
    _PROPOSED_VERSION,
    _SECOND_MEMBERSHIP,
    _add_catalogue,
    _create_committed_parent,
    _open,
)


def test_abort_preparing_update_is_atomic_private_and_leaves_active_unchanged(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
    aborted: AbortCorpusUpdateOutcome | None = None
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=_PARENT_VERSION,
            )
            transaction.remove_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _SECOND_MEMBERSHIP,
            )

        with capture_logs() as logs:
            with repository.transaction() as transaction:
                aborted = transaction.abort_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                )
                assert (
                    transaction.get_corpus_update(
                        _CORPUS_A.corpus_id,
                        _PROPOSED_VERSION,
                    )
                    == aborted.update
                )

        assert aborted is not None
        assert aborted.replayed is False
        assert aborted.update.state is CorpusUpdateState.ABORTED
        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP,)
        assert logs == [
            {
                "event": "corpus_update_aborted",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS_A.corpus_id.as_str(),
                "corpus_version": _PROPOSED_VERSION.as_str(),
                "update_state": "aborted",
                "replayed": False,
                "log_level": "info",
            },
            {
                "event": "transaction_completed",
                "pipeline_stage": "storage",
                "log_level": "info",
            },
        ]
        rendered = repr(logs)
        assert _CORPUS_A.display_name not in rendered
        assert _FIRST_DOCUMENT_ID.as_str() not in rendered
        assert str(database_path) not in rendered

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    _SECOND_MEMBERSHIP,
                )
    finally:
        repository.close()

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            "SELECT corpus_version FROM corpus_active_versions WHERE corpus_id = ?",
            (_CORPUS_A.corpus_id.as_str(),),
        ).fetchone() == (_PARENT_VERSION.as_str(),)
        assert connection.execute(
            "SELECT corpus_version FROM corpus_versions WHERE corpus_id = ?",
            (_CORPUS_A.corpus_id.as_str(),),
        ).fetchall() == [(_PARENT_VERSION.as_str(),)]
    finally:
        connection.close()


def test_exact_abort_replay_reopens_after_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    aborted: AbortCorpusUpdateOutcome | None = None
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            expected_corpus_version=None,
        )
        aborted = transaction.abort_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        )
    repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    replayed: AbortCorpusUpdateOutcome | None = None
    try:
        with reopened.transaction() as transaction:
            replayed = transaction.abort_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )

        assert aborted is not None
        assert replayed is not None
        assert replayed.replayed is True
        assert replayed.update == aborted.update
        assert (
            reopened.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == aborted.update
        )
    finally:
        reopened.close()


def test_rolled_back_abort_remains_preparing_and_has_no_success_log(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=None,
            )

        with capture_logs() as logs:
            with pytest.raises(RuntimeError, match="private-primary"):
                with repository.transaction() as transaction:
                    transaction.abort_corpus_update(
                        _CORPUS_A.corpus_id,
                        _PROPOSED_VERSION,
                    )
                    raise RuntimeError("private-primary")

        update = repository.get_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        )
        assert update is not None
        assert update.state is CorpusUpdateState.PREPARING
        assert [log["event"] for log in logs] == ["transaction_rolled_back"]
        assert "private-primary" not in repr(logs)
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("corpus_id", "corpus_version"),
    [
        (_CORPUS_B.corpus_id, _PROPOSED_VERSION),
        (_CORPUS_A.corpus_id, _OTHER_VERSION),
    ],
    ids=("cross-corpus", "unknown-update"),
)
def test_abort_rejects_unowned_update_identity(
    tmp_path: Path,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=None,
            )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.abort_corpus_update(corpus_id, corpus_version)

        update = repository.get_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        )
        assert update is not None
        assert update.state is CorpusUpdateState.PREPARING
    finally:
        repository.close()


def test_abort_rejects_a_committed_update(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.abort_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PARENT_VERSION,
                )

        update = repository.get_corpus_update(
            _CORPUS_A.corpus_id,
            _PARENT_VERSION,
        )
        assert update is not None
        assert update.state is CorpusUpdateState.COMMITTED
    finally:
        repository.close()
