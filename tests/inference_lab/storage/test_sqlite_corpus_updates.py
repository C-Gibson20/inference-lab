"""Tests for durable SQLite corpus-update creation and reopening."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

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
    AbortCorpusUpdateOutcome,
    BeginCorpusUpdateOutcome,
    ComponentKind,
    CorpusUpdate,
    CorpusUpdateState,
    RepositoryUnavailableError,
    StorageConflictError,
)
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


def test_add_membership_is_ordered_idempotent_and_not_an_implicit_replace(
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
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _SECOND_MEMBERSHIP,
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
            )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    _REPLACEMENT_MEMBERSHIP,
                )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP)
    finally:
        repository.close()


def test_add_requires_explicit_target_corpus_catalogue_membership(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_B.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=None,
            )

        for membership in (
            _FIRST_MEMBERSHIP,
            CorpusDocumentVersionMembership(
                _CORPUS_B.corpus_id,
                _FIRST_DOCUMENT_ID,
                _FIRST_DOCUMENT_VERSION_ID,
            ),
        ):
            with pytest.raises(StorageConflictError):
                with repository.transaction() as transaction:
                    transaction.add_corpus_update_document_membership(
                        _CORPUS_B.corpus_id,
                        _PROPOSED_VERSION,
                        membership,
                    )

        assert (
            repository.list_corpus_update_document_memberships(
                _CORPUS_B.corpus_id,
                _PROPOSED_VERSION,
            )
            == ()
        )
    finally:
        repository.close()


def test_replace_consumes_explicit_logical_document_agreement_and_replays(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=_PARENT_VERSION,
            )
            transaction.replace_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
                _REPLACEMENT_MEMBERSHIP,
            )
            transaction.replace_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
                _REPLACEMENT_MEMBERSHIP,
            )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_REPLACEMENT_MEMBERSHIP, _SECOND_MEMBERSHIP)
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert reopened.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_REPLACEMENT_MEMBERSHIP, _SECOND_MEMBERSHIP)
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("expected_membership", "replacement_membership"),
    [
        (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP),
        (_REPLACEMENT_MEMBERSHIP, _REPLACEMENT_MEMBERSHIP),
    ],
    ids=("different-logical-document", "stale-expected-membership"),
)
def test_replace_rejects_identity_disagreement_without_mutation(
    tmp_path: Path,
    expected_membership: CorpusDocumentVersionMembership,
    replacement_membership: CorpusDocumentVersionMembership,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=_PARENT_VERSION,
            )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.replace_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    expected_membership,
                    replacement_membership,
                )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP)
    finally:
        repository.close()


def test_remove_is_idempotent_and_preserves_global_catalogue_membership(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
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
            transaction.remove_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _SECOND_MEMBERSHIP,
            )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP,)
        assert (
            repository.get_corpus_document_version_membership(
                _CORPUS_A.corpus_id,
                _SECOND_DOCUMENT_VERSION_ID,
            )
            == _SECOND_MEMBERSHIP
        )
    finally:
        repository.close()


def test_staging_mutation_rolls_back_without_losing_the_update(
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

        with pytest.raises(RuntimeError, match="private-primary"):
            with repository.transaction() as transaction:
                transaction.add_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    _FIRST_MEMBERSHIP,
                )
                raise RuntimeError("private-primary")

        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            is not None
        )
        assert (
            repository.list_corpus_update_document_memberships(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == ()
        )
    finally:
        repository.close()


def test_sealed_membership_allows_exact_replay_but_rejects_changes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            expected_corpus_version=None,
        )
        transaction.add_corpus_update_document_membership(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            _FIRST_MEMBERSHIP,
        )
    repository.close()

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """UPDATE corpus_updates
            SET state = 'ready', manifest_digest = ?
            WHERE corpus_id = ? AND corpus_version = ?""",
            (
                _MANIFEST_DIGEST,
                _CORPUS_A.corpus_id.as_str(),
                _PROPOSED_VERSION.as_str(),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        with reopened.transaction() as transaction:
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
            )
            transaction.remove_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _SECOND_MEMBERSHIP,
            )

        with pytest.raises(StorageConflictError):
            with reopened.transaction() as transaction:
                transaction.add_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    _SECOND_MEMBERSHIP,
                )

        assert reopened.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP,)
    finally:
        reopened.close()


def test_inspection_rejects_unknown_or_cross_corpus_update_identity(
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

        for corpus_id, corpus_version in (
            (_CORPUS_B.corpus_id, _PROPOSED_VERSION),
            (_CORPUS_A.corpus_id, _OTHER_VERSION),
        ):
            with pytest.raises(StorageConflictError):
                repository.list_corpus_update_document_memberships(
                    corpus_id,
                    corpus_version,
                )
    finally:
        repository.close()
