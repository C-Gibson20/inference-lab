"""Tests for SQLite corpus-publication conflicts."""

import sqlite3
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import (
    CorpusUpdateState,
    StorageConflictError,
)
from tests.inference_lab.storage.corpus_publication_fixtures import (
    _CORPUS_A,
    _CORPUS_B,
    _FIRST_MEMBERSHIP,
    _SECOND_MEMBERSHIP,
    _STALE_VERSION,
    _VERSION_A1,
    _VERSION_A2,
    _add_catalogue,
    _commit,
    _open,
    _prepare_update,
)


def test_stale_ready_writer_fails_without_consuming_sequence_or_state(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    _prepare_update(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
    )
    _commit(repository, _VERSION_A1)
    _prepare_update(
        repository,
        _VERSION_A2,
        expected_corpus_version=_VERSION_A1,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    _prepare_update(
        repository,
        _STALE_VERSION,
        expected_corpus_version=_VERSION_A1,
        memberships=(_SECOND_MEMBERSHIP,),
    )
    _commit(repository, _VERSION_A2)
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError):
                with repository.transaction() as transaction:
                    transaction.commit_corpus_update(
                        _CORPUS_A.corpus_id,
                        _STALE_VERSION,
                    )

        stale = repository.get_corpus_update(
            _CORPUS_A.corpus_id,
            _STALE_VERSION,
        )
        assert stale is not None
        assert stale.state is CorpusUpdateState.READY
        assert [log["event"] for log in logs] == [
            "corpus_version_conflict",
            "transaction_failed",
        ]
        assert logs[0]["conflict_kind"] == "expected_active_mismatch"
    finally:
        repository.close()

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            """SELECT corpus_version FROM corpus_active_versions
            WHERE corpus_id = ?""",
            (_CORPUS_A.corpus_id.as_str(),),
        ).fetchone() == (_VERSION_A2.as_str(),)
        assert connection.execute(
            """SELECT sequence, corpus_version FROM corpus_versions
            WHERE corpus_id = ? ORDER BY sequence""",
            (_CORPUS_A.corpus_id.as_str(),),
        ).fetchall() == [(1, _VERSION_A1.as_str()), (2, _VERSION_A2.as_str())]
    finally:
        connection.close()


@pytest.mark.parametrize("state", ["preparing", "aborted"])
def test_commit_rejects_non_ready_update(tmp_path: Path, state: str) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            _VERSION_A1,
            expected_corpus_version=None,
        )
        if state == "aborted":
            transaction.abort_corpus_update(_CORPUS_A.corpus_id, _VERSION_A1)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.commit_corpus_update(
                    _CORPUS_A.corpus_id,
                    _VERSION_A1,
                )
    finally:
        repository.close()


def test_committed_version_id_cannot_be_reused_by_another_corpus(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    _prepare_update(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    committed = _commit(repository, _VERSION_A1)
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError):
                with repository.transaction() as transaction:
                    transaction.begin_corpus_update(
                        _CORPUS_B.corpus_id,
                        _VERSION_A1,
                        expected_corpus_version=None,
                    )

        assert [log["event"] for log in logs] == [
            "corpus_version_conflict",
            "transaction_failed",
        ]
        assert logs[0]["conflict_kind"] == "version_id_reused"
        assert repository.get_active_corpus_version(_CORPUS_A.corpus_id) == (
            committed.version
        )
        assert repository.list_corpus_versions(_CORPUS_A.corpus_id) == (
            committed.version,
        )
        assert repository.get_active_corpus_version(_CORPUS_B.corpus_id) is None
        assert repository.list_corpus_versions(_CORPUS_B.corpus_id) == ()
    finally:
        repository.close()
