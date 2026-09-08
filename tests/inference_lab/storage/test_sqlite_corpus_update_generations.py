"""Tests for staging validated SQLite corpus-update generations."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentGenerationValidator,
    ComponentKind,
    CorpusUpdateState,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.storage.corpus_update_component_fixtures import (
    _CORPUS_A,
    _CORPUS_B,
    _DENSE,
    _DOCUMENT_ID,
    _LEXICAL,
    _MEMBERSHIP,
    _OTHER_VERSION,
    _PRIVATE_DATABASE_FAILURE,
    _PRIVATE_VALIDATOR_FAILURE,
    _PROPOSED_VERSION,
    FailingValidator,
    InvalidResultValidator,
    RecordingValidator,
    RejectingValidator,
    _create_committed_parent,
    _fixed_created_at,
    _generation,
    _install_trigger,
    _open,
    _prepare_update,
)


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
