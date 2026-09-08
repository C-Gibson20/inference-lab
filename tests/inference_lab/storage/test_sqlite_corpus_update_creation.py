"""Tests for durable SQLite corpus-update creation and replay."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
)
from inference_lab.storage import (
    BeginCorpusUpdateOutcome,
    ComponentKind,
    CorpusUpdate,
    CorpusUpdateState,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.storage.corpus_update_fixtures import (
    _CORPUS_A,
    _CORPUS_B,
    _CREATED_AT,
    _DENSE,
    _FIRST_DOCUMENT_ID,
    _FIRST_DOCUMENT_VERSION_ID,
    _LEXICAL,
    _OTHER_VERSION,
    _PARENT_VERSION,
    _PROPOSED_VERSION,
    _SECOND_DOCUMENT_ID,
    _SECOND_DOCUMENT_VERSION_ID,
    _add_catalogue,
    _create_committed_parent,
    _open,
)


def test_begin_first_update_persists_canonical_identity_and_logs_after_commit(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    repository = _open(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_CORPUS_A)

        outcome: BeginCorpusUpdateOutcome | None = None
        with capture_logs() as logs:
            with repository.transaction() as transaction:
                outcome = transaction.begin_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    expected_corpus_version=None,
                    required_component_kinds=(_LEXICAL, _DENSE),
                )
                assert (
                    transaction.get_corpus_update(
                        _CORPUS_A.corpus_id,
                        _PROPOSED_VERSION,
                    )
                    == outcome.update
                )

        assert outcome is not None
        assert outcome.replayed is False
        assert outcome.update == CorpusUpdate(
            corpus_id=_CORPUS_A.corpus_id,
            corpus_version=_PROPOSED_VERSION,
            expected_corpus_version=None,
            required_component_kinds=(_DENSE, _LEXICAL),
            state=CorpusUpdateState.PREPARING,
            created_at=_CREATED_AT,
        )
        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == outcome.update
        )
        assert (
            repository.get_corpus_update(
                _CORPUS_B.corpus_id,
                _PROPOSED_VERSION,
            )
            is None
        )
        assert logs == [
            {
                "event": "corpus_update_started",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS_A.corpus_id.as_str(),
                "corpus_version": _PROPOSED_VERSION.as_str(),
                "update_state": "preparing",
                "required_component_count": 2,
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
            """SELECT component_kind
            FROM corpus_update_required_components
            WHERE corpus_version = ?
            ORDER BY component_kind""",
            (_PROPOSED_VERSION.as_str(),),
        ).fetchall() == [("dense",), ("lexical",)]
        assert connection.execute(
            """SELECT count(*) FROM corpus_update_document_memberships
            WHERE corpus_version = ?""",
            (_PROPOSED_VERSION.as_str(),),
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_begin_from_active_parent_materialises_complete_membership(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
    outcome: BeginCorpusUpdateOutcome | None = None
    try:
        with repository.transaction() as transaction:
            outcome = transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=_PARENT_VERSION,
            )

        assert outcome is not None
        assert outcome.update.expected_corpus_version == _PARENT_VERSION
    finally:
        repository.close()

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            """SELECT document_id, document_version_id
            FROM corpus_update_document_memberships
            WHERE corpus_id = ? AND corpus_version = ?
            ORDER BY document_id, document_version_id""",
            (_CORPUS_A.corpus_id.as_str(), _PROPOSED_VERSION.as_str()),
        ).fetchall() == [
            (_FIRST_DOCUMENT_ID.as_str(), _FIRST_DOCUMENT_VERSION_ID.as_str()),
            (_SECOND_DOCUMENT_ID.as_str(), _SECOND_DOCUMENT_VERSION_ID.as_str()),
        ]
    finally:
        connection.close()


def test_exact_begin_replay_survives_restart_without_replacing_creation_time(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    created: BeginCorpusUpdateOutcome | None = None
    with repository.transaction() as transaction:
        created = transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            expected_corpus_version=None,
            required_component_kinds=(_DENSE, _LEXICAL),
        )
    repository.close()

    def unexpected_clock_call() -> datetime:
        raise AssertionError("exact replay must retain the persisted timestamp")

    reopened = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=unexpected_clock_call,
    )
    replayed: BeginCorpusUpdateOutcome | None = None
    try:
        with reopened.transaction() as transaction:
            replayed = transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=None,
                required_component_kinds=(_LEXICAL, _DENSE),
            )

        assert created is not None
        assert replayed is not None
        assert replayed.replayed is True
        assert replayed.update == created.update
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("corpus_id", "expected_corpus_version", "required_component_kinds"),
    [
        (_CORPUS_A.corpus_id, None, (_DENSE,)),
        (_CORPUS_A.corpus_id, _OTHER_VERSION, (_DENSE, _LEXICAL)),
        (_CORPUS_B.corpus_id, None, (_DENSE, _LEXICAL)),
    ],
    ids=("changed-components", "changed-parent", "reused-version-id"),
)
def test_begin_replay_rejects_changed_identity_payload(
    tmp_path: Path,
    corpus_id: CorpusId,
    expected_corpus_version: CorpusVersionId | None,
    required_component_kinds: tuple[ComponentKind, ...],
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    original: BeginCorpusUpdateOutcome | None = None
    try:
        with repository.transaction() as transaction:
            original = transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=None,
                required_component_kinds=(_DENSE, _LEXICAL),
            )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    corpus_id,
                    _PROPOSED_VERSION,
                    expected_corpus_version=expected_corpus_version,
                    required_component_kinds=required_component_kinds,
                )

        assert original is not None
        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == original.update
        )
    finally:
        repository.close()


def test_begin_requires_exact_active_parent_and_logs_bounded_conflict(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError):
                with repository.transaction() as transaction:
                    transaction.begin_corpus_update(
                        _CORPUS_A.corpus_id,
                        _PROPOSED_VERSION,
                        expected_corpus_version=None,
                    )

        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            is None
        )
        assert [log["event"] for log in logs] == [
            "corpus_version_conflict",
            "transaction_failed",
        ]
        assert logs[0]["conflict_kind"] == "expected_active_mismatch"
        assert logs[0]["corpus_id"] == _CORPUS_A.corpus_id.as_str()
        assert logs[0]["corpus_version"] == _PROPOSED_VERSION.as_str()
        assert logs[0]["error_category"] == "conflict"
        assert _CORPUS_A.display_name not in repr(logs)
        assert str(database_path) not in repr(logs)
    finally:
        repository.close()


def test_begin_rejects_an_unknown_target_corpus(tmp_path: Path) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    expected_corpus_version=None,
                )

        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            is None
        )
    finally:
        repository.close()


def test_rolled_back_begin_is_neither_durable_nor_logged_as_started(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    try:
        with capture_logs() as logs:
            with pytest.raises(RuntimeError, match="private-primary"):
                with repository.transaction() as transaction:
                    transaction.begin_corpus_update(
                        _CORPUS_A.corpus_id,
                        _PROPOSED_VERSION,
                        expected_corpus_version=None,
                    )
                    raise RuntimeError("private-primary")

        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            is None
        )
        assert [log["event"] for log in logs] == ["transaction_rolled_back"]
        assert "private-primary" not in repr(logs)
    finally:
        repository.close()


@pytest.mark.parametrize(
    "created_at_factory",
    [None, lambda: "not-a-datetime"],
    ids=("not-callable", "invalid-return"),
)
def test_created_at_factory_failures_do_not_publish_an_update(
    tmp_path: Path,
    created_at_factory: object,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    if created_at_factory is None:
        with pytest.raises(TypeError, match="created_at_factory"):
            SQLiteMetadataRepository.open(
                database_path,
                apply_migrations=True,
                created_at_factory=created_at_factory,  # type: ignore[arg-type]
            )
        assert not database_path.exists()
        return

    repository = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        created_at_factory=created_at_factory,  # type: ignore[arg-type]
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_CORPUS_A)
        with pytest.raises(RepositoryUnavailableError):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    expected_corpus_version=None,
                )
        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            is None
        )
    finally:
        repository.close()
