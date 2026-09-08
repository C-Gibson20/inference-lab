"""Tests for sealing durable SQLite corpus updates."""

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import (
    ComponentGenerationState,
    CorpusUpdateState,
    CorpusVersionManifest,
    CorruptStorageError,
    NoCorpusChangeOutcome,
    SealCorpusUpdateOutcome,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.storage.corpus_update_component_fixtures import (
    _CORPUS_A,
    _DENSE,
    _DOCUMENT_ID,
    _LEXICAL,
    _MEMBERSHIP,
    _OTHER_VERSION,
    _PRIVATE_DATABASE_FAILURE,
    _PROPOSED_VERSION,
    RecordingValidator,
    _create_committed_parent,
    _fixed_created_at,
    _generation,
    _install_trigger,
    _open,
    _prepare_update,
)


def test_seal_persists_canonical_manifest_and_logs_after_commit(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    repository = _open(database_path)
    _prepare_update(repository)
    dense = _generation(_DENSE, generation_id="generation-2")
    lexical = _generation(_LEXICAL)
    ready_dense = replace(dense, state=ComponentGenerationState.READY)
    ready_lexical = replace(lexical, state=ComponentGenerationState.READY)
    with repository.transaction() as transaction:
        transaction.register_corpus_update_component_generation(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            lexical,
            validator=RecordingValidator(),
        )
        transaction.register_corpus_update_component_generation(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            dense,
            validator=RecordingValidator(),
        )

    outcome: SealCorpusUpdateOutcome | None = None
    try:
        with capture_logs() as logs:
            with repository.transaction() as transaction:
                result = transaction.seal_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                )
                assert isinstance(result, SealCorpusUpdateOutcome)
                outcome = result

        expected_manifest = CorpusVersionManifest(
            corpus_id=_CORPUS_A.corpus_id,
            corpus_version=_PROPOSED_VERSION,
            document_memberships=(_MEMBERSHIP,),
            component_generations=(ready_dense, ready_lexical),
        )
        assert outcome is not None
        assert outcome.replayed is False
        assert outcome.manifest == expected_manifest
        assert outcome.update.state is CorpusUpdateState.READY
        assert outcome.update.manifest_digest == expected_manifest.manifest_digest
        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == outcome.update
        )
        assert logs == [
            {
                "event": "corpus_update_ready",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS_A.corpus_id.as_str(),
                "corpus_version": _PROPOSED_VERSION.as_str(),
                "update_state": "ready",
                "document_count": 1,
                "component_count": 2,
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
        assert _DOCUMENT_ID.as_str() not in rendered
        assert lexical.generation_id not in rendered
        assert str(database_path) not in rendered

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.remove_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    _MEMBERSHIP,
                )
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.replace_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    ready_lexical,
                    _generation(_LEXICAL, generation_id="generation-3"),
                    validator=RecordingValidator(),
                )
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    replayed: SealCorpusUpdateOutcome | NoCorpusChangeOutcome | None = None
    try:
        with reopened.transaction() as transaction:
            replayed = transaction.seal_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )

        assert isinstance(replayed, SealCorpusUpdateOutcome)
        assert replayed.replayed is True
        assert outcome is not None
        assert replayed.update == outcome.update
        assert replayed.manifest == outcome.manifest
    finally:
        reopened.close()


def test_seal_rejects_missing_required_component_without_changing_update(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _prepare_update(repository)
    with repository.transaction() as transaction:
        transaction.register_corpus_update_component_generation(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            _generation(_LEXICAL),
            validator=RecordingValidator(),
        )
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.seal_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                )

        update = repository.get_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        )
        assert update is not None
        assert update.state is CorpusUpdateState.PREPARING
        assert update.manifest_digest is None
        assert (
            len(
                repository.list_corpus_update_component_generations(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                )
            )
            == 1
        )
    finally:
        repository.close()


def test_semantically_equal_parent_with_new_generation_returns_durable_no_change(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    parent_manifest = _create_committed_parent(database_path, with_component=True)
    repository = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=_fixed_created_at,
    )
    outcome: NoCorpusChangeOutcome | None = None
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=_OTHER_VERSION,
                required_component_kinds=(_LEXICAL,),
            )
            transaction.register_corpus_update_component_generation(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _generation(generation_id="proposed-generation"),
                validator=RecordingValidator(),
            )

        with capture_logs() as logs:
            with repository.transaction() as transaction:
                result = transaction.seal_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                )
                assert isinstance(result, NoCorpusChangeOutcome)
                outcome = result

        assert outcome is not None
        assert outcome.replayed is False
        assert outcome.update.state is CorpusUpdateState.ABORTED
        assert outcome.update.manifest_digest == (
            outcome.proposed_manifest.manifest_digest
        )
        assert outcome.active_version.corpus_version == _OTHER_VERSION
        assert outcome.active_manifest == parent_manifest
        assert outcome.proposed_manifest.has_same_semantic_state(parent_manifest)
        assert (
            outcome.proposed_manifest.manifest_digest != parent_manifest.manifest_digest
        )
        assert [log["event"] for log in logs] == [
            "corpus_update_aborted",
            "transaction_completed",
        ]
        assert logs[0]["abort_reason"] == "no_change"
        assert logs[0]["replayed"] is False
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    replayed: SealCorpusUpdateOutcome | NoCorpusChangeOutcome | None = None
    try:
        with reopened.transaction() as transaction:
            replayed = transaction.seal_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )

        assert isinstance(replayed, NoCorpusChangeOutcome)
        assert replayed.replayed is True
        assert outcome is not None
        assert replayed.update == outcome.update
        assert replayed.active_version == outcome.active_version
        assert replayed.proposed_manifest == outcome.proposed_manifest

        connection = sqlite3.connect(database_path)
        try:
            assert connection.execute(
                "SELECT corpus_version FROM corpus_active_versions WHERE corpus_id = ?",
                (_CORPUS_A.corpus_id.as_str(),),
            ).fetchone() == (_OTHER_VERSION.as_str(),)
            assert connection.execute(
                "SELECT count(*) FROM corpus_versions WHERE corpus_id = ?",
                (_CORPUS_A.corpus_id.as_str(),),
            ).fetchone() == (1,)
        finally:
            connection.close()
    finally:
        reopened.close()


def test_changed_parent_membership_seals_as_ready(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=_fixed_created_at,
    )
    outcome: SealCorpusUpdateOutcome | NoCorpusChangeOutcome | None = None
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=_OTHER_VERSION,
            )
            transaction.remove_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _MEMBERSHIP,
            )
            outcome = transaction.seal_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )

        assert isinstance(outcome, SealCorpusUpdateOutcome)
        assert outcome.update.state is CorpusUpdateState.READY
        assert outcome.manifest.document_memberships == ()
    finally:
        repository.close()


def test_rolled_back_seal_remains_preparing_and_emits_no_ready_event(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _prepare_update(repository, required_component_kinds=())
    try:
        with capture_logs() as logs:
            with pytest.raises(RuntimeError, match="private-primary"):
                with repository.transaction() as transaction:
                    transaction.seal_corpus_update(
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
        assert update.manifest_digest is None
        assert [log["event"] for log in logs] == ["transaction_rolled_back"]
    finally:
        repository.close()


def test_seal_rejects_an_explicitly_aborted_update(tmp_path: Path) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _prepare_update(repository, required_component_kinds=())
    try:
        with repository.transaction() as transaction:
            transaction.abort_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.seal_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                )
    finally:
        repository.close()


def test_seal_database_failure_preserves_preparing_state_and_hides_payload(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _prepare_update(repository, required_component_kinds=())
    _install_trigger(
        database_path,
        f"""CREATE TRIGGER reject_manifest_seal
        BEFORE UPDATE ON corpus_updates
        WHEN NEW.state = 'ready'
        BEGIN
            SELECT RAISE(ABORT, '{_PRIVATE_DATABASE_FAILURE}');
        END""",
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError) as raised:
                with repository.transaction() as transaction:
                    transaction.seal_corpus_update(
                        _CORPUS_A.corpus_id,
                        _PROPOSED_VERSION,
                    )

        update = repository.get_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        )
        assert update is not None
        assert update.state is CorpusUpdateState.PREPARING
        assert update.manifest_digest is None
        assert "corpus_update_ready" not in {log["event"] for log in logs}
        assert _PRIVATE_DATABASE_FAILURE not in str(raised.value)
        assert _PRIVATE_DATABASE_FAILURE not in repr(logs)
    finally:
        repository.close()


def test_seal_replay_detects_persisted_manifest_digest_disagreement(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _prepare_update(repository, required_component_kinds=())
    repository.close()

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """UPDATE corpus_updates
            SET state = 'ready', manifest_digest = ?
            WHERE corpus_id = ? AND corpus_version = ?""",
            (
                "f" * 64,
                _CORPUS_A.corpus_id.as_str(),
                _PROPOSED_VERSION.as_str(),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        with pytest.raises(CorruptStorageError):
            with reopened.transaction() as transaction:
                transaction.seal_corpus_update(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                )
    finally:
        reopened.close()
