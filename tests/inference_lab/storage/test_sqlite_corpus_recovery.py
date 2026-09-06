"""Tests for durable SQLite corpus-update recovery inventory and abort."""

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
    CommitCorpusUpdateOutcome,
    CorpusUpdate,
    CorpusUpdateState,
    RepositoryClosedError,
    SealCorpusUpdateOutcome,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CREATED_AT = datetime(2026, 9, 6, 14, 0, tzinfo=UTC)
_CORPUS_A = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"),
    "Private recovery corpus",
)
_CORPUS_B = CorpusMetadata(
    CorpusId("22222222-2222-4222-8222-222222222222"),
    "Other private recovery corpus",
)
_PARENT_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_READY_VERSION = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_PREPARING_VERSION = CorpusVersionId("55555555-5555-4555-8555-555555555555")
_ABORTED_VERSION = CorpusVersionId("66666666-6666-4666-8666-666666666666")
_FIRST_DOCUMENT_ID = DocumentId("77777777-7777-4777-8777-777777777777")
_SECOND_DOCUMENT_ID = DocumentId("88888888-8888-4888-8888-888888888888")
_FIRST_DOCUMENT_VERSION = DocumentVersion(
    DocumentVersionId("99999999-9999-4999-8999-999999999999"),
    _FIRST_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_SECOND_DOCUMENT_VERSION = DocumentVersion(
    DocumentVersionId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
    _SECOND_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "b" * 64),
)
_FIRST_MEMBERSHIP = CorpusDocumentVersionMembership(
    _CORPUS_A.corpus_id,
    _FIRST_DOCUMENT_ID,
    _FIRST_DOCUMENT_VERSION.document_version_id,
)
_SECOND_MEMBERSHIP = CorpusDocumentVersionMembership(
    _CORPUS_A.corpus_id,
    _SECOND_DOCUMENT_ID,
    _SECOND_DOCUMENT_VERSION.document_version_id,
)


def _fixed_created_at() -> datetime:
    return _CREATED_AT


def _open(path: Path, *, initialise: bool = False) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=initialise,
        created_at_factory=_fixed_created_at,
    )


def _add_catalogue(repository: SQLiteMetadataRepository) -> None:
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS_A)
        transaction.add_corpus(_CORPUS_B)
        for document_id, version, membership in (
            (_FIRST_DOCUMENT_ID, _FIRST_DOCUMENT_VERSION, _FIRST_MEMBERSHIP),
            (_SECOND_DOCUMENT_ID, _SECOND_DOCUMENT_VERSION, _SECOND_MEMBERSHIP),
        ):
            transaction.add_document(DocumentIdentity(document_id))
            transaction.add_document_version(version)
            transaction.add_corpus_document_version_membership(membership)


def _publish_parent(repository: SQLiteMetadataRepository) -> CommitCorpusUpdateOutcome:
    committed: object = None
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            _PARENT_VERSION,
            expected_corpus_version=None,
        )
        transaction.add_corpus_update_document_membership(
            _CORPUS_A.corpus_id,
            _PARENT_VERSION,
            _FIRST_MEMBERSHIP,
        )
        sealed = transaction.seal_corpus_update(
            _CORPUS_A.corpus_id,
            _PARENT_VERSION,
        )
        assert isinstance(sealed, SealCorpusUpdateOutcome)
        committed = transaction.commit_corpus_update(
            _CORPUS_A.corpus_id,
            _PARENT_VERSION,
        )
    assert isinstance(committed, CommitCorpusUpdateOutcome)
    return committed


def _create_recovery_inventory(
    repository: SQLiteMetadataRepository,
) -> tuple[CorpusUpdate, CorpusUpdate]:
    ready: CorpusUpdate | None = None
    preparing: CorpusUpdate | None = None
    with repository.transaction() as transaction:
        outcome = transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            _READY_VERSION,
            expected_corpus_version=_PARENT_VERSION,
        )
        transaction.add_corpus_update_document_membership(
            _CORPUS_A.corpus_id,
            _READY_VERSION,
            _SECOND_MEMBERSHIP,
        )
        sealed = transaction.seal_corpus_update(
            _CORPUS_A.corpus_id,
            _READY_VERSION,
        )
        assert isinstance(sealed, SealCorpusUpdateOutcome)
        ready = sealed.update
        assert outcome.update.state is CorpusUpdateState.PREPARING

    with repository.transaction() as transaction:
        preparing = transaction.begin_corpus_update(
            _CORPUS_B.corpus_id,
            _PREPARING_VERSION,
            expected_corpus_version=None,
        ).update

    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS_B.corpus_id,
            _ABORTED_VERSION,
            expected_corpus_version=None,
        )
        transaction.abort_corpus_update(
            _CORPUS_B.corpus_id,
            _ABORTED_VERSION,
        )

    assert ready is not None
    assert preparing is not None
    return ready, preparing


def test_restart_lists_only_non_terminal_work_without_publishing_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    first = _open(database_path, initialise=True)
    _add_catalogue(first)
    parent = _publish_parent(first)
    ready, preparing = _create_recovery_inventory(first)
    assert first.list_non_terminal_corpus_updates() == (ready, preparing)
    first.close()

    reopened = _open(database_path)
    try:
        assert reopened.list_non_terminal_corpus_updates() == (ready, preparing)
        with reopened.transaction() as transaction:
            assert transaction.list_non_terminal_corpus_updates() == (
                ready,
                preparing,
            )

        assert reopened.get_active_corpus_version(_CORPUS_A.corpus_id) == parent.version
        assert reopened.get_active_corpus_version(_CORPUS_B.corpus_id) is None
        assert reopened.list_corpus_versions(_CORPUS_A.corpus_id) == (parent.version,)
        assert reopened.list_corpus_versions(_CORPUS_B.corpus_id) == ()
        assert (
            reopened.get_corpus_version(
                _CORPUS_A.corpus_id,
                _READY_VERSION,
            )
            is None
        )
    finally:
        reopened.close()

    with pytest.raises(RepositoryClosedError):
        reopened.list_non_terminal_corpus_updates()


def test_ready_abort_is_durable_idempotent_and_preserves_staged_manifest(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    first = _open(database_path, initialise=True)
    _add_catalogue(first)
    parent = _publish_parent(first)
    ready, preparing = _create_recovery_inventory(first)
    first.close()

    reopened = _open(database_path)
    aborted: AbortCorpusUpdateOutcome | None = None
    try:
        with capture_logs() as logs:
            with reopened.transaction() as transaction:
                aborted = transaction.abort_corpus_update(
                    _CORPUS_A.corpus_id,
                    _READY_VERSION,
                )

        assert aborted is not None
        assert aborted.replayed is False
        assert aborted.update.state is CorpusUpdateState.ABORTED
        assert aborted.update.manifest_digest == ready.manifest_digest
        assert reopened.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _READY_VERSION,
        ) == (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP)
        assert reopened.list_non_terminal_corpus_updates() == (preparing,)
        assert reopened.get_active_corpus_version(_CORPUS_A.corpus_id) == parent.version
        assert (
            reopened.get_corpus_version(
                _CORPUS_A.corpus_id,
                _READY_VERSION,
            )
            is None
        )
        assert logs == [
            {
                "event": "corpus_update_aborted",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS_A.corpus_id.as_str(),
                "corpus_version": _READY_VERSION.as_str(),
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
        assert ready.manifest_digest is not None
        assert ready.manifest_digest not in rendered
        assert _CORPUS_A.display_name not in rendered
        assert _FIRST_DOCUMENT_ID.as_str() not in rendered
        assert str(database_path) not in rendered

        with pytest.raises(StorageConflictError):
            with reopened.transaction() as transaction:
                transaction.abort_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PARENT_VERSION,
                )
    finally:
        reopened.close()

    final = _open(database_path)
    replayed: AbortCorpusUpdateOutcome | None = None
    try:
        with final.transaction() as transaction:
            replayed = transaction.abort_corpus_update(
                _CORPUS_A.corpus_id,
                _READY_VERSION,
            )

        assert aborted is not None
        assert replayed is not None
        assert replayed.replayed is True
        assert replayed.update == aborted.update
        assert final.list_non_terminal_corpus_updates() == (preparing,)
        assert final.get_active_corpus_version(_CORPUS_A.corpus_id) == parent.version
    finally:
        final.close()
