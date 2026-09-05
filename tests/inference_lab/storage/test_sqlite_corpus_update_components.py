"""Tests for validated SQLite corpus-update component staging."""

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
    RepositoryUnavailableError,
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


def _fixed_created_at() -> datetime:
    return _CREATED_AT


def _open(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        created_at_factory=_fixed_created_at,
    )


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
