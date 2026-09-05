"""Tests for atomic SQLite corpus-version publication."""

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
    CommitCorpusUpdateOutcome,
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusUpdateState,
    CorruptStorageError,
    SealCorpusUpdateOutcome,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CREATED_AT = datetime(2026, 9, 5, 16, 0, tzinfo=UTC)
_CORPUS_A = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"),
    "Private publication corpus",
)
_CORPUS_B = CorpusMetadata(
    CorpusId("22222222-2222-4222-8222-222222222222"),
    "Other private corpus",
)
_VERSION_A1 = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_VERSION_A2 = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_STALE_VERSION = CorpusVersionId("55555555-5555-4555-8555-555555555555")
_UNKNOWN_CORPUS_ID = CorpusId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_FIRST_DOCUMENT_ID = DocumentId("66666666-6666-4666-8666-666666666666")
_SECOND_DOCUMENT_ID = DocumentId("77777777-7777-4777-8777-777777777777")
_FIRST_DOCUMENT_VERSION_ID = DocumentVersionId("88888888-8888-4888-8888-888888888888")
_SECOND_DOCUMENT_VERSION_ID = DocumentVersionId("99999999-9999-4999-8999-999999999999")
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
_LEXICAL = ComponentKind("lexical")
_LEXICAL_GENERATION = ComponentGenerationReference(
    component_kind=_LEXICAL,
    generation_id="publication-generation-1",
    implementation="fake-index",
    format_version="1.0.0",
    contract_version="1.0.0",
    configuration_digest="c" * 64,
    source_manifest_digest="d" * 64,
    corpus_id=_CORPUS_A.corpus_id,
    corpus_version=_VERSION_A1,
    state=ComponentGenerationState.PREPARING,
)


class _AcceptingComponentValidator:
    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        return True


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
        for document_id, version, membership in (
            (_FIRST_DOCUMENT_ID, _FIRST_DOCUMENT_VERSION, _FIRST_MEMBERSHIP),
            (_SECOND_DOCUMENT_ID, _SECOND_DOCUMENT_VERSION, _SECOND_MEMBERSHIP),
        ):
            transaction.add_document(DocumentIdentity(document_id))
            transaction.add_document_version(version)
            transaction.add_corpus_document_version_membership(membership)


def _prepare_update(
    repository: SQLiteMetadataRepository,
    corpus_version: CorpusVersionId,
    *,
    expected_corpus_version: CorpusVersionId | None,
    memberships: tuple[CorpusDocumentVersionMembership, ...] = (),
    component: ComponentGenerationReference | None = None,
) -> SealCorpusUpdateOutcome:
    outcome: object = None
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            corpus_version,
            expected_corpus_version=expected_corpus_version,
            required_component_kinds=(component.component_kind,)
            if component is not None
            else (),
        )
        for membership in memberships:
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                corpus_version,
                membership,
            )
        if component is not None:
            transaction.register_corpus_update_component_generation(
                _CORPUS_A.corpus_id,
                corpus_version,
                component,
                validator=_AcceptingComponentValidator(),
            )
        outcome = transaction.seal_corpus_update(
            _CORPUS_A.corpus_id,
            corpus_version,
        )
    assert isinstance(outcome, SealCorpusUpdateOutcome)
    return outcome


def _commit(
    repository: SQLiteMetadataRepository,
    corpus_version: CorpusVersionId,
) -> CommitCorpusUpdateOutcome:
    outcome: CommitCorpusUpdateOutcome | None = None
    with repository.transaction() as transaction:
        outcome = transaction.commit_corpus_update(
            _CORPUS_A.corpus_id,
            corpus_version,
        )
    assert outcome is not None
    return outcome


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


def test_version_reads_fail_closed_when_committed_lifecycle_is_corrupt(
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
    _commit(repository, _VERSION_A1)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP TRIGGER corpus_updates_lifecycle_update")
        connection.execute(
            """UPDATE corpus_updates SET state = 'ready'
            WHERE corpus_id = ? AND corpus_version = ?""",
            (_CORPUS_A.corpus_id.as_str(), _VERSION_A1.as_str()),
        )
        connection.commit()
    finally:
        connection.close()

    try:
        with pytest.raises(CorruptStorageError):
            repository.get_active_corpus_version(_CORPUS_A.corpus_id)
        with pytest.raises(CorruptStorageError):
            repository.get_corpus_version(_CORPUS_A.corpus_id, _VERSION_A1)
        with pytest.raises(CorruptStorageError):
            repository.list_corpus_versions(_CORPUS_A.corpus_id)
    finally:
        repository.close()


def test_active_read_fails_closed_when_corpus_pointer_row_is_missing(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP TRIGGER corpus_active_versions_delete_immutable")
        connection.execute(
            "DELETE FROM corpus_active_versions WHERE corpus_id = ?",
            (_CORPUS_A.corpus_id.as_str(),),
        )
        connection.commit()
    finally:
        connection.close()

    try:
        with pytest.raises(CorruptStorageError):
            repository.get_active_corpus_version(_CORPUS_A.corpus_id)
    finally:
        repository.close()


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
