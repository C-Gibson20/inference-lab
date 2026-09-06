"""Tests for validated SQLite corpus-update component staging."""

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentGenerationValidator,
    ComponentKind,
    CorpusUpdateState,
    CorpusVersionManifest,
    CorruptStorageError,
    NoCorpusChangeOutcome,
    RepositoryUnavailableError,
    SealCorpusUpdateOutcome,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CREATED_AT = datetime(2026, 9, 5, 14, 0, tzinfo=UTC)
_CORPUS_A = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"),
    "Private component corpus",
)
_CORPUS_B = CorpusMetadata(
    CorpusId("22222222-2222-4222-8222-222222222222"),
    "Other private corpus",
)
_PROPOSED_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_OTHER_VERSION = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_DOCUMENT_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")
_DOCUMENT_VERSION = DocumentVersion(
    _DOCUMENT_VERSION_ID,
    _DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_MEMBERSHIP = CorpusDocumentVersionMembership(
    _CORPUS_A.corpus_id,
    _DOCUMENT_ID,
    _DOCUMENT_VERSION_ID,
)
_DENSE = ComponentKind("dense")
_LEXICAL = ComponentKind("lexical")
_PRIVATE_VALIDATOR_FAILURE = "private-validator-failure"
_PRIVATE_DATABASE_FAILURE = "private-database-failure"


def _fixed_created_at() -> datetime:
    return _CREATED_AT


def _open(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        created_at_factory=_fixed_created_at,
    )


def _install_trigger(path: Path, statement: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def _prepare_update(
    repository: SQLiteMetadataRepository,
    *,
    required_component_kinds: tuple[ComponentKind, ...] = (_DENSE, _LEXICAL),
) -> None:
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS_A)
        transaction.add_corpus(_CORPUS_B)
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(_DOCUMENT_VERSION)
        transaction.add_corpus_document_version_membership(_MEMBERSHIP)
        transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            expected_corpus_version=None,
            required_component_kinds=required_component_kinds,
        )
        transaction.add_corpus_update_document_membership(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            _MEMBERSHIP,
        )


def _create_committed_parent(
    path: Path,
    *,
    with_component: bool = False,
) -> CorpusVersionManifest:
    repository = _open(path)
    parent_manifest: CorpusVersionManifest | None = None
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_CORPUS_A)
            transaction.add_corpus(_CORPUS_B)
            transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
            transaction.add_document_version(_DOCUMENT_VERSION)
            transaction.add_corpus_document_version_membership(_MEMBERSHIP)
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _OTHER_VERSION,
                expected_corpus_version=None,
                required_component_kinds=(_LEXICAL,) if with_component else (),
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _OTHER_VERSION,
                _MEMBERSHIP,
            )
            if with_component:
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _OTHER_VERSION,
                    _generation(
                        corpus_version=_OTHER_VERSION,
                        generation_id="parent-generation",
                    ),
                    validator=RecordingValidator(),
                )
            sealed = transaction.seal_corpus_update(
                _CORPUS_A.corpus_id,
                _OTHER_VERSION,
            )
            assert isinstance(sealed, SealCorpusUpdateOutcome)
            committed = transaction.commit_corpus_update(
                _CORPUS_A.corpus_id,
                _OTHER_VERSION,
            )
            parent_manifest = committed.manifest
    finally:
        repository.close()
    assert parent_manifest is not None
    return parent_manifest


def _generation(
    kind: ComponentKind = _LEXICAL,
    *,
    generation_id: str = "generation-1",
    corpus_id: CorpusId = _CORPUS_A.corpus_id,
    corpus_version: CorpusVersionId = _PROPOSED_VERSION,
    state: ComponentGenerationState = ComponentGenerationState.PREPARING,
) -> ComponentGenerationReference:
    return ComponentGenerationReference(
        component_kind=kind,
        generation_id=generation_id,
        implementation="fake-index",
        format_version="1.0.0",
        contract_version="1.0.0",
        configuration_digest="b" * 64,
        source_manifest_digest="c" * 64,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        state=state,
    )


class RecordingValidator:
    """Deterministically accept generations and retain bounded call inputs."""

    def __init__(self) -> None:
        self.calls: list[
            tuple[
                ComponentGenerationReference,
                tuple[CorpusDocumentVersionMembership, ...],
            ]
        ] = []

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        self.calls.append((generation, document_memberships))
        return True


class RejectingValidator:
    """Deterministically reject a generation without exposing a reason."""

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        return False


class FailingValidator:
    """Raise a private adapter failure for boundary-translation tests."""

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        raise RuntimeError(_PRIVATE_VALIDATOR_FAILURE)


class InvalidResultValidator:
    """Return a non-boolean result despite its structural annotation."""

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        return cast("bool", "ready")


def test_registration_validates_exact_membership_and_reopens_in_order(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _prepare_update(repository)
    validator = RecordingValidator()
    lexical = _generation(_LEXICAL)
    dense = _generation(_DENSE, generation_id="generation-2")
    ready_lexical = replace(lexical, state=ComponentGenerationState.READY)
    ready_dense = replace(dense, state=ComponentGenerationState.READY)
    try:
        with repository.transaction() as transaction:
            assert (
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    lexical,
                    validator=validator,
                )
                == ready_lexical
            )
            assert (
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    dense,
                    validator=validator,
                )
                == ready_dense
            )

        assert validator.calls == [
            (lexical, (_MEMBERSHIP,)),
            (dense, (_MEMBERSHIP,)),
        ]
        assert repository.list_corpus_update_component_generations(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (ready_dense, ready_lexical)
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert reopened.list_corpus_update_component_generations(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (ready_dense, ready_lexical)
    finally:
        reopened.close()


def test_exact_registration_replay_does_not_repeat_external_validation(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _prepare_update(repository, required_component_kinds=(_LEXICAL,))
    generation = _generation()
    ready = replace(generation, state=ComponentGenerationState.READY)
    try:
        with repository.transaction() as transaction:
            transaction.register_corpus_update_component_generation(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                generation,
                validator=RecordingValidator(),
            )
            assert (
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    generation,
                    validator=FailingValidator(),
                )
                == ready
            )
            transaction.abort_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )

        with repository.transaction() as transaction:
            assert (
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    ready,
                    validator=FailingValidator(),
                )
                == ready
            )
    finally:
        repository.close()


@pytest.mark.parametrize(
    "generation",
    [
        _generation(ComponentKind("metadata")),
        _generation(corpus_id=_CORPUS_B.corpus_id),
        _generation(corpus_version=_OTHER_VERSION),
        _generation(state=ComponentGenerationState.READY),
    ],
    ids=("undeclared-kind", "wrong-corpus", "wrong-version", "already-ready"),
)
def test_registration_rejects_incompatible_generation_without_validation(
    tmp_path: Path,
    generation: ComponentGenerationReference,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _prepare_update(repository, required_component_kinds=(_LEXICAL,))
    validator = RecordingValidator()
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    generation,
                    validator=validator,
                )

        assert validator.calls == []
        assert (
            repository.list_corpus_update_component_generations(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == ()
        )
    finally:
        repository.close()


@pytest.mark.parametrize(
    "foreign_generation",
    [
        _generation(_DENSE, corpus_id=_CORPUS_B.corpus_id),
        _generation(_DENSE, corpus_version=_OTHER_VERSION),
    ],
    ids=("mixed-corpus", "mixed-version"),
)
def test_mixed_generation_failure_preserves_exact_active_version(
    tmp_path: Path,
    foreign_generation: ComponentGenerationReference,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    parent_manifest = _create_committed_parent(database_path)
    repository = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=_fixed_created_at,
    )
    validator = RecordingValidator()
    valid_generation = _generation(_LEXICAL)
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            expected_corpus_version=_OTHER_VERSION,
            required_component_kinds=(_DENSE, _LEXICAL),
        )

    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    valid_generation,
                    validator=validator,
                )
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    foreign_generation,
                    validator=validator,
                )

        assert validator.calls == [(valid_generation, (_MEMBERSHIP,))]
        assert (
            repository.list_corpus_update_component_generations(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == ()
        )
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
        active = repository.get_active_corpus_version(_CORPUS_A.corpus_id)
        assert active is not None
        assert active.corpus_version == _OTHER_VERSION
        assert active.sequence == 1
        assert repository.list_corpus_versions(_CORPUS_A.corpus_id) == (active,)
        assert (
            repository.get_corpus_version(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            is None
        )
        with repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id) as snapshot:
            assert snapshot.version.corpus_version == _OTHER_VERSION
            assert snapshot.manifest == parent_manifest
    finally:
        repository.close()


def test_replacement_requires_expected_generation_and_supports_exact_replay(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _prepare_update(repository, required_component_kinds=(_LEXICAL,))
    first = _generation()
    ready_first = replace(first, state=ComponentGenerationState.READY)
    second = _generation(generation_id="generation-2")
    ready_second = replace(second, state=ComponentGenerationState.READY)
    stale = replace(ready_first, generation_id="stale-generation")
    validator = RecordingValidator()
    try:
        with repository.transaction() as transaction:
            transaction.register_corpus_update_component_generation(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                first,
                validator=validator,
            )
            assert (
                transaction.replace_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    ready_first,
                    second,
                    validator=validator,
                )
                == ready_second
            )
            assert (
                transaction.replace_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    ready_first,
                    second,
                    validator=FailingValidator(),
                )
                == ready_second
            )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.replace_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    stale,
                    first,
                    validator=validator,
                )

        assert repository.list_corpus_update_component_generations(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (ready_second,)
        assert validator.calls == [
            (first, (_MEMBERSHIP,)),
            (second, (_MEMBERSHIP,)),
        ]
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("validator", "expected_error"),
    [
        (RejectingValidator(), StorageConflictError),
        (FailingValidator(), RepositoryUnavailableError),
        (InvalidResultValidator(), RepositoryUnavailableError),
    ],
    ids=("rejected", "adapter-failure", "invalid-result"),
)
def test_validation_failure_persists_nothing_and_hides_adapter_payload(
    tmp_path: Path,
    validator: object,
    expected_error: type[Exception],
) -> None:
    repository = _open(tmp_path / "private-path" / "metadata.sqlite3")
    _prepare_update(repository, required_component_kinds=(_LEXICAL,))
    try:
        with capture_logs() as logs:
            with pytest.raises(expected_error) as raised:
                with repository.transaction() as transaction:
                    transaction.register_corpus_update_component_generation(
                        _CORPUS_A.corpus_id,
                        _PROPOSED_VERSION,
                        _generation(),
                        validator=cast("ComponentGenerationValidator", validator),
                    )

        assert (
            repository.list_corpus_update_component_generations(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == ()
        )
        assert _PRIVATE_VALIDATOR_FAILURE not in str(raised.value)
        assert _PRIVATE_VALIDATOR_FAILURE not in repr(logs)
        assert _CORPUS_A.display_name not in repr(logs)
        assert _DOCUMENT_ID.as_str() not in repr(logs)
    finally:
        repository.close()


def test_component_registration_rolls_back_with_its_transaction(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _prepare_update(repository, required_component_kinds=(_LEXICAL,))
    try:
        with pytest.raises(RuntimeError, match="private-primary"):
            with repository.transaction() as transaction:
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    _generation(),
                    validator=RecordingValidator(),
                )
                raise RuntimeError("private-primary")

        assert (
            repository.list_corpus_update_component_generations(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == ()
        )
    finally:
        repository.close()


def test_component_inspection_rejects_unknown_or_cross_corpus_update(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _prepare_update(repository)
    try:
        for corpus_id, corpus_version in (
            (_CORPUS_B.corpus_id, _PROPOSED_VERSION),
            (_CORPUS_A.corpus_id, _OTHER_VERSION),
        ):
            with pytest.raises(StorageConflictError):
                repository.list_corpus_update_component_generations(
                    corpus_id,
                    corpus_version,
                )
    finally:
        repository.close()


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


def test_component_insert_failure_is_atomic_and_hides_database_payload(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _prepare_update(repository, required_component_kinds=(_LEXICAL,))
    _install_trigger(
        database_path,
        f"""CREATE TRIGGER reject_component_insert
        BEFORE INSERT ON corpus_update_component_generations
        BEGIN
            SELECT RAISE(ABORT, '{_PRIVATE_DATABASE_FAILURE}');
        END""",
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError) as raised:
                with repository.transaction() as transaction:
                    transaction.register_corpus_update_component_generation(
                        _CORPUS_A.corpus_id,
                        _PROPOSED_VERSION,
                        _generation(),
                        validator=RecordingValidator(),
                    )

        assert (
            repository.list_corpus_update_component_generations(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == ()
        )
        update = repository.get_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        )
        assert update is not None
        assert update.state is CorpusUpdateState.PREPARING
        assert _PRIVATE_DATABASE_FAILURE not in str(raised.value)
        assert _PRIVATE_DATABASE_FAILURE not in repr(logs)
    finally:
        repository.close()


def test_component_replace_failure_preserves_expected_generation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _prepare_update(repository, required_component_kinds=(_LEXICAL,))
    first = _generation()
    ready_first = replace(first, state=ComponentGenerationState.READY)
    with repository.transaction() as transaction:
        transaction.register_corpus_update_component_generation(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            first,
            validator=RecordingValidator(),
        )
    _install_trigger(
        database_path,
        f"""CREATE TRIGGER reject_component_replace
        BEFORE UPDATE ON corpus_update_component_generations
        BEGIN
            SELECT RAISE(ABORT, '{_PRIVATE_DATABASE_FAILURE}');
        END""",
    )
    try:
        with pytest.raises(StorageConflictError) as raised:
            with repository.transaction() as transaction:
                transaction.replace_corpus_update_component_generation(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    ready_first,
                    _generation(generation_id="generation-2"),
                    validator=RecordingValidator(),
                )

        assert repository.list_corpus_update_component_generations(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (ready_first,)
        assert _PRIVATE_DATABASE_FAILURE not in str(raised.value)
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
