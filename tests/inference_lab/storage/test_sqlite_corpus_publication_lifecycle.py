"""Tests for successful SQLite corpus publication and replay."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import (
    CommitCorpusUpdateOutcome,
    ComponentGenerationState,
    CorpusUpdateState,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.storage.corpus_publication_fixtures import (
    _CORPUS_A,
    _CORPUS_B,
    _CREATED_AT,
    _FIRST_DOCUMENT_ID,
    _FIRST_DOCUMENT_VERSION_ID,
    _FIRST_MEMBERSHIP,
    _LEXICAL,
    _LEXICAL_GENERATION,
    _PRIVATE_FAILURE,
    _SECOND_MEMBERSHIP,
    _UNKNOWN_CORPUS_ID,
    _VERSION_A1,
    _VERSION_A2,
    _add_catalogue,
    _commit,
    _fixed_created_at,
    _open,
    _prepare_update,
    _UncertainCommitConnection,
)


def test_first_publication_is_atomic_materialised_and_logged_after_commit(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    sealed = _prepare_update(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
        component=_LEXICAL_GENERATION,
    )
    outcome: CommitCorpusUpdateOutcome | None = None
    try:
        with capture_logs() as logs:
            with repository.transaction() as transaction:
                outcome = transaction.commit_corpus_update(
                    _CORPUS_A.corpus_id,
                    _VERSION_A1,
                )
                assert outcome.update.state is CorpusUpdateState.COMMITTED
                assert (
                    transaction.get_active_corpus_version(_CORPUS_A.corpus_id)
                    == outcome.version
                )
                assert (
                    transaction.get_corpus_version(
                        _CORPUS_A.corpus_id,
                        _VERSION_A1,
                    )
                    == outcome.version
                )
                assert transaction.list_corpus_versions(_CORPUS_A.corpus_id) == (
                    outcome.version,
                )

        assert outcome is not None
        assert outcome.replayed is False
        assert outcome.manifest == sealed.manifest
        assert outcome.version.corpus_id == _CORPUS_A.corpus_id
        assert outcome.version.corpus_version == _VERSION_A1
        assert outcome.version.sequence == 1
        assert outcome.version.parent_corpus_version is None
        assert outcome.version.manifest_digest == sealed.manifest.manifest_digest
        assert outcome.version.created_at == _CREATED_AT
        assert len(outcome.manifest.component_generations) == 1
        assert outcome.manifest.component_generations[0].generation_id == (
            _LEXICAL_GENERATION.generation_id
        )
        assert (
            outcome.manifest.component_generations[0].state
            is ComponentGenerationState.READY
        )
        assert logs == [
            {
                "event": "corpus_version_committed",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS_A.corpus_id.as_str(),
                "corpus_version": _VERSION_A1.as_str(),
                "sequence": 1,
                "document_count": 1,
                "component_count": 1,
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
    finally:
        repository.close()

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            "SELECT corpus_version FROM corpus_active_versions WHERE corpus_id = ?",
            (_CORPUS_A.corpus_id.as_str(),),
        ).fetchone() == (_VERSION_A1.as_str(),)
        assert connection.execute(
            """SELECT sequence, parent_corpus_version, manifest_digest
            FROM corpus_versions
            WHERE corpus_id = ? AND corpus_version = ?""",
            (_CORPUS_A.corpus_id.as_str(), _VERSION_A1.as_str()),
        ).fetchone() == (1, None, sealed.manifest.manifest_digest)
        assert connection.execute(
            """SELECT document_id, document_version_id
            FROM corpus_version_document_memberships
            WHERE corpus_id = ? AND corpus_version = ?""",
            (_CORPUS_A.corpus_id.as_str(), _VERSION_A1.as_str()),
        ).fetchall() == [
            (_FIRST_DOCUMENT_ID.as_str(), _FIRST_DOCUMENT_VERSION_ID.as_str())
        ]
        assert connection.execute(
            """SELECT component_kind, generation_id, state
            FROM corpus_update_component_generations
            WHERE corpus_id = ? AND corpus_version = ?""",
            (_CORPUS_A.corpus_id.as_str(), _VERSION_A1.as_str()),
        ).fetchone() == (
            _LEXICAL.as_str(),
            _LEXICAL_GENERATION.generation_id,
            "ready",
        )
    finally:
        connection.close()


def test_exact_commit_replay_survives_restart_without_calling_clock(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    _prepare_update(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    committed = _commit(repository, _VERSION_A1)
    repository.close()

    def unexpected_clock_call() -> datetime:
        raise AssertionError("an exact commit replay must retain persisted time")

    reopened = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=unexpected_clock_call,
    )
    try:
        replayed = _commit(reopened, _VERSION_A1)

        assert replayed.replayed is True
        assert replayed.update == committed.update
        assert replayed.version == committed.version
        assert replayed.manifest == committed.manifest
    finally:
        reopened.close()


def test_exact_retry_recovers_after_uncertain_commit_response(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
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
        factory=_UncertainCommitConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    uncertain = SQLiteMetadataRepository(
        connection,
        MIGRATIONS,
        created_at_factory=_fixed_created_at,
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageTransactionError) as raised:
                with uncertain.transaction() as transaction:
                    transaction.commit_corpus_update(
                        _CORPUS_A.corpus_id,
                        _VERSION_A1,
                    )

        committed = uncertain.get_active_corpus_version(_CORPUS_A.corpus_id)
        assert committed is not None
        assert committed.corpus_version == _VERSION_A1
        assert committed.sequence == 1
        assert uncertain.list_corpus_versions(_CORPUS_A.corpus_id) == (committed,)
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
        assert _CORPUS_A.display_name not in repr(logs)
        assert str(database_path) not in repr(logs)
    finally:
        uncertain.close()

    def unexpected_clock_call() -> datetime:
        raise AssertionError("an exact retry must retain the committed timestamp")

    reopened = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=unexpected_clock_call,
    )
    try:
        with capture_logs() as retry_logs:
            replayed = _commit(reopened, _VERSION_A1)

        assert replayed.replayed is True
        assert replayed.version == committed
        assert replayed.manifest == sealed.manifest
        assert reopened.list_corpus_versions(_CORPUS_A.corpus_id) == (committed,)
        assert retry_logs == [
            {
                "event": "corpus_version_committed",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS_A.corpus_id.as_str(),
                "corpus_version": _VERSION_A1.as_str(),
                "sequence": 1,
                "document_count": 1,
                "component_count": 0,
                "replayed": True,
                "log_level": "info",
            },
            {
                "event": "transaction_completed",
                "pipeline_stage": "storage",
                "log_level": "info",
            },
        ]
    finally:
        reopened.close()


def test_ordinary_advance_uses_parent_and_next_per_corpus_sequence(
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
    first = _commit(repository, _VERSION_A1)
    _prepare_update(
        repository,
        _VERSION_A2,
        expected_corpus_version=_VERSION_A1,
        memberships=(_SECOND_MEMBERSHIP,),
    )
    try:
        second = _commit(repository, _VERSION_A2)

        assert first.version.sequence == 1
        assert second.version.sequence == 2
        assert second.version.parent_corpus_version == _VERSION_A1
        assert second.manifest.document_memberships == (
            _FIRST_MEMBERSHIP,
            _SECOND_MEMBERSHIP,
        )
    finally:
        repository.close()


def test_active_and_ordered_history_reads_survive_restart_and_enforce_ownership(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    assert repository.get_active_corpus_version(_CORPUS_A.corpus_id) is None
    assert repository.get_active_corpus_version(_UNKNOWN_CORPUS_ID) is None
    assert repository.list_corpus_versions(_CORPUS_A.corpus_id) == ()
    assert repository.get_corpus_version(_CORPUS_A.corpus_id, _VERSION_A1) is None
    _prepare_update(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    first = _commit(repository, _VERSION_A1)
    _prepare_update(
        repository,
        _VERSION_A2,
        expected_corpus_version=_VERSION_A1,
        memberships=(_SECOND_MEMBERSHIP,),
    )
    second = _commit(repository, _VERSION_A2)
    repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert reopened.get_active_corpus_version(_CORPUS_A.corpus_id) == (
            second.version
        )
        assert (
            reopened.get_corpus_version(
                _CORPUS_A.corpus_id,
                _VERSION_A1,
            )
            == first.version
        )
        assert (
            reopened.get_corpus_version(
                _CORPUS_A.corpus_id,
                _VERSION_A2,
            )
            == second.version
        )
        assert reopened.list_corpus_versions(_CORPUS_A.corpus_id) == (
            first.version,
            second.version,
        )

        assert reopened.get_corpus_version(_CORPUS_B.corpus_id, _VERSION_A1) is None
        assert reopened.get_active_corpus_version(_CORPUS_B.corpus_id) is None
        assert reopened.list_corpus_versions(_CORPUS_B.corpus_id) == ()
    finally:
        reopened.close()
