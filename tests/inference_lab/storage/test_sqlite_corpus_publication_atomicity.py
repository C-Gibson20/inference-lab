"""Tests for SQLite corpus-publication atomicity and failure recovery."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import (
    RepositoryUnavailableError,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.storage.corpus_publication_fixtures import (
    _CORPUS_A,
    _FIRST_MEMBERSHIP,
    _LEXICAL_GENERATION,
    _PRIVATE_FAILURE,
    _VERSION_A1,
    _add_catalogue,
    _commit,
    _CommitFailingConnection,
    _execute_schema_statement,
    _fixed_created_at,
    _open,
    _prepare_update,
)


def test_caller_rollback_after_commit_operation_publishes_nothing(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    sealed = _prepare_update(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(RuntimeError, match="private-primary"):
                with repository.transaction() as transaction:
                    transaction.commit_corpus_update(
                        _CORPUS_A.corpus_id,
                        _VERSION_A1,
                    )
                    raise RuntimeError("private-primary")

        update = repository.get_corpus_update(
            _CORPUS_A.corpus_id,
            _VERSION_A1,
        )
        assert update == sealed.update
        assert [log["event"] for log in logs] == ["transaction_rolled_back"]
        assert "private-primary" not in repr(logs)
    finally:
        repository.close()

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            "SELECT count(*) FROM corpus_versions"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT corpus_version FROM corpus_active_versions WHERE corpus_id = ?",
            (_CORPUS_A.corpus_id.as_str(),),
        ).fetchone() == (None,)
        assert connection.execute(
            "SELECT count(*) FROM corpus_version_document_memberships"
        ).fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "trigger_timing",
    [
        "BEFORE INSERT ON corpus_versions",
        "BEFORE INSERT ON corpus_version_document_memberships",
        "BEFORE UPDATE OF corpus_version ON corpus_active_versions",
        "BEFORE UPDATE OF state ON corpus_updates WHEN NEW.state = 'committed'",
    ],
    ids=(
        "immutable-version",
        "materialised-membership",
        "active-pointer",
        "committed-lifecycle",
    ),
)
def test_each_publication_statement_failure_rolls_back_every_change(
    tmp_path: Path,
    trigger_timing: str,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    sealed = _prepare_update(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
        component=_LEXICAL_GENERATION,
    )
    _execute_schema_statement(
        database_path,
        f"""CREATE TRIGGER reject_publication
        {trigger_timing}
        BEGIN
            SELECT RAISE(ABORT, '{_PRIVATE_FAILURE}');
        END""",
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError) as raised:
                with repository.transaction() as transaction:
                    transaction.commit_corpus_update(
                        _CORPUS_A.corpus_id,
                        _VERSION_A1,
                    )

        assert _PRIVATE_FAILURE not in str(raised.value)
        assert logs == [
            {
                "event": "transaction_failed",
                "pipeline_stage": "storage",
                "error_category": "conflict",
                "retryable": False,
                "log_level": "error",
            }
        ]
        assert _PRIVATE_FAILURE not in repr(logs)
        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _VERSION_A1,
            )
            == sealed.update
        )
        assert (
            repository.list_corpus_update_document_memberships(
                _CORPUS_A.corpus_id,
                _VERSION_A1,
            )
            == sealed.manifest.document_memberships
        )
        assert (
            repository.list_corpus_update_component_generations(
                _CORPUS_A.corpus_id,
                _VERSION_A1,
            )
            == sealed.manifest.component_generations
        )
        assert repository.get_active_corpus_version(_CORPUS_A.corpus_id) is None
        assert (
            repository.get_corpus_version(
                _CORPUS_A.corpus_id,
                _VERSION_A1,
            )
            is None
        )
        assert repository.list_corpus_versions(_CORPUS_A.corpus_id) == ()

        connection = sqlite3.connect(database_path)
        try:
            assert connection.execute(
                "SELECT count(*) FROM corpus_versions"
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT count(*) FROM corpus_version_document_memberships"
            ).fetchone() == (0,)
        finally:
            connection.close()

        _execute_schema_statement(database_path, "DROP TRIGGER reject_publication")
        committed = _commit(repository, _VERSION_A1)
        assert committed.version.sequence == 1
        assert committed.manifest == sealed.manifest
    finally:
        repository.close()


def test_transaction_commit_failure_rolls_back_publication_and_success_log(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    initial = _open(database_path)
    _add_catalogue(initial)
    sealed = _prepare_update(
        initial,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    initial.close()

    connection = sqlite3.connect(
        database_path,
        isolation_level=None,
        factory=_CommitFailingConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    repository = SQLiteMetadataRepository(
        connection,
        MIGRATIONS,
        created_at_factory=_fixed_created_at,
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageTransactionError) as raised:
                with repository.transaction() as transaction:
                    transaction.commit_corpus_update(
                        _CORPUS_A.corpus_id,
                        _VERSION_A1,
                    )

        assert _PRIVATE_FAILURE not in str(raised.value)
        assert logs == [
            {
                "event": "transaction_failed",
                "pipeline_stage": "storage",
                "error_category": "transaction",
                "retryable": False,
                "log_level": "error",
            }
        ]
        assert _PRIVATE_FAILURE not in repr(logs)
        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _VERSION_A1,
            )
            == sealed.update
        )
        assert repository.get_active_corpus_version(_CORPUS_A.corpus_id) is None
        assert repository.list_corpus_versions(_CORPUS_A.corpus_id) == ()
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=_fixed_created_at,
    )
    try:
        committed = _commit(reopened, _VERSION_A1)
        assert committed.version.sequence == 1
        assert committed.manifest == sealed.manifest
    finally:
        reopened.close()


def test_publication_clock_failure_preserves_ready_proposal_for_retry(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    initial = _open(database_path)
    _add_catalogue(initial)
    sealed = _prepare_update(
        initial,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    initial.close()

    def failing_clock() -> datetime:
        raise RuntimeError(_PRIVATE_FAILURE)

    repository = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=failing_clock,
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(RepositoryUnavailableError) as raised:
                with repository.transaction() as transaction:
                    transaction.commit_corpus_update(
                        _CORPUS_A.corpus_id,
                        _VERSION_A1,
                    )

        assert _PRIVATE_FAILURE not in str(raised.value)
        assert _PRIVATE_FAILURE not in repr(logs)
        assert [log["event"] for log in logs] == ["transaction_failed"]
        assert logs[0]["error_category"] == "repository_unavailable"
        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _VERSION_A1,
            )
            == sealed.update
        )
        assert repository.get_active_corpus_version(_CORPUS_A.corpus_id) is None
        assert repository.list_corpus_versions(_CORPUS_A.corpus_id) == ()
    finally:
        repository.close()

    reopened = _open(database_path)
    try:
        committed = _commit(reopened, _VERSION_A1)
        assert committed.version.sequence == 1
        assert committed.manifest == sealed.manifest
    finally:
        reopened.close()
