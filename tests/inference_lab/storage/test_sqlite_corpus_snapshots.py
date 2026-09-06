"""Tests for dedicated version-pinned SQLite corpus snapshot lifecycles."""

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
    CommitCorpusUpdateOutcome,
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusSnapshot,
    RepositoryClosedError,
    SealCorpusUpdateOutcome,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CREATED_AT = datetime(2026, 9, 6, 10, 0, tzinfo=UTC)
_CORPUS_A = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"),
    "Private snapshot corpus",
)
_CORPUS_B = CorpusMetadata(
    CorpusId("22222222-2222-4222-8222-222222222222"),
    "Other private corpus",
)
_VERSION_A1 = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_VERSION_A2 = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_UNKNOWN_VERSION = CorpusVersionId("55555555-5555-4555-8555-555555555555")
_FIRST_DOCUMENT_ID = DocumentId("66666666-6666-4666-8666-666666666666")
_SECOND_DOCUMENT_ID = DocumentId("77777777-7777-4777-8777-777777777777")
_FIRST_DOCUMENT_VERSION = DocumentVersion(
    DocumentVersionId("88888888-8888-4888-8888-888888888888"),
    _FIRST_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_SECOND_DOCUMENT_VERSION = DocumentVersion(
    DocumentVersionId("99999999-9999-4999-8999-999999999999"),
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
_LEXICAL = ComponentKind("lexical")
_LEXICAL_GENERATION = ComponentGenerationReference(
    component_kind=_LEXICAL,
    generation_id="snapshot-generation-1",
    implementation="fake-index",
    format_version="1.0.0",
    contract_version="1.0.0",
    configuration_digest="c" * 64,
    source_manifest_digest="d" * 64,
    corpus_id=_CORPUS_A.corpus_id,
    corpus_version=_VERSION_A1,
    state=ComponentGenerationState.PREPARING,
)


class _AcceptingValidator:
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


def _publish(
    repository: SQLiteMetadataRepository,
    corpus_version: CorpusVersionId,
    *,
    expected_corpus_version: CorpusVersionId | None,
    memberships: tuple[CorpusDocumentVersionMembership, ...],
    component: ComponentGenerationReference | None = None,
) -> tuple[SealCorpusUpdateOutcome, CommitCorpusUpdateOutcome]:
    sealed: object = None
    committed: object = None
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
                validator=_AcceptingValidator(),
            )
        sealed = transaction.seal_corpus_update(
            _CORPUS_A.corpus_id,
            corpus_version,
        )
        committed = transaction.commit_corpus_update(
            _CORPUS_A.corpus_id,
            corpus_version,
        )
    assert isinstance(sealed, SealCorpusUpdateOutcome)
    assert isinstance(committed, CommitCorpusUpdateOutcome)
    return sealed, committed


def test_active_snapshot_loads_exact_manifest_and_logs_owned_lifecycle(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    sealed, committed = _publish(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
        component=_LEXICAL_GENERATION,
    )
    snapshot = repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id)
    try:
        assert isinstance(snapshot, CorpusSnapshot)
        assert snapshot.is_closed is False
        with pytest.raises(StorageTransactionError):
            _ = snapshot.version
        with pytest.raises(StorageTransactionError):
            _ = snapshot.manifest

        with capture_logs() as logs:
            with snapshot as opened:
                assert opened is snapshot
                assert snapshot.corpus_id == _CORPUS_A.corpus_id
                assert snapshot.version == committed.version
                assert snapshot.manifest == sealed.manifest
                assert snapshot.is_closed is False

            snapshot.close()

        assert snapshot.is_closed is True
        with pytest.raises(RepositoryClosedError):
            _ = snapshot.corpus_id
        with pytest.raises(RepositoryClosedError):
            _ = snapshot.version
        with pytest.raises(RepositoryClosedError):
            _ = snapshot.manifest
        with pytest.raises(RepositoryClosedError):
            snapshot.__enter__()
        assert logs == [
            {
                "event": "corpus_snapshot_opened",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS_A.corpus_id.as_str(),
                "corpus_version": _VERSION_A1.as_str(),
                "sequence": 1,
                "document_count": 1,
                "component_count": 1,
                "log_level": "info",
            },
            {
                "event": "corpus_snapshot_closed",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS_A.corpus_id.as_str(),
                "corpus_version": _VERSION_A1.as_str(),
                "sequence": 1,
                "log_level": "info",
            },
        ]
        rendered = repr(logs)
        assert _CORPUS_A.display_name not in rendered
        assert _FIRST_DOCUMENT_ID.as_str() not in rendered
        assert str(database_path) not in rendered
    finally:
        repository.close()


def test_open_snapshot_stays_pinned_while_writer_advances_active_version(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    first_seal, first_commit = _publish(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    snapshot = repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id)
    try:
        with snapshot:
            _publish(
                repository,
                _VERSION_A2,
                expected_corpus_version=_VERSION_A1,
                memberships=(_SECOND_MEMBERSHIP,),
            )

            assert snapshot.version == first_commit.version
            assert snapshot.manifest == first_seal.manifest
            with repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id) as current:
                assert current.version.corpus_version == _VERSION_A2
                assert current.manifest.document_memberships == (
                    _FIRST_MEMBERSHIP,
                    _SECOND_MEMBERSHIP,
                )

        with repository.open_corpus_snapshot(
            _CORPUS_A.corpus_id,
            _VERSION_A1,
        ) as historical:
            assert historical.version == first_commit.version
            assert historical.manifest == first_seal.manifest
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("corpus_id", "corpus_version"),
    [
        (_CORPUS_B.corpus_id, None),
        (_CORPUS_B.corpus_id, _VERSION_A1),
        (_CORPUS_A.corpus_id, _UNKNOWN_VERSION),
    ],
    ids=("no-active-version", "wrong-owner", "unknown-version"),
)
def test_snapshot_selection_failure_is_terminal_bounded_and_payload_free(
    tmp_path: Path,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId | None,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    _publish(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    snapshot = (
        repository.open_active_corpus_snapshot(corpus_id)
        if corpus_version is None
        else repository.open_corpus_snapshot(corpus_id, corpus_version)
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError):
                with snapshot:
                    pytest.fail("an unavailable snapshot must not open")

        assert snapshot.is_closed is True
        assert logs == [
            {
                "event": "corpus_snapshot_open_failed",
                "pipeline_stage": "storage",
                "corpus_id": corpus_id.as_str(),
                "corpus_version": (
                    None if corpus_version is None else corpus_version.as_str()
                ),
                "error_category": "conflict",
                "retryable": False,
                "log_level": "error",
            }
        ]
        assert _CORPUS_A.display_name not in repr(logs)
        assert str(database_path) not in repr(logs)
    finally:
        repository.close()


def test_snapshot_rejects_nested_entry_and_preserves_primary_exception(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    _publish(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    snapshot = repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id)
    try:
        with pytest.raises(RuntimeError, match="private-primary"):
            with snapshot:
                with pytest.raises(StorageTransactionError):
                    snapshot.__enter__()
                raise RuntimeError("private-primary")

        assert snapshot.is_closed is True
    finally:
        repository.close()


def test_snapshot_close_before_entry_is_idempotent_and_terminal(tmp_path: Path) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    snapshot = repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id)
    try:
        with capture_logs() as logs:
            snapshot.close()
            snapshot.close()

        assert snapshot.is_closed is True
        assert logs == []
        with pytest.raises(RepositoryClosedError):
            snapshot.__enter__()
    finally:
        repository.close()


def test_snapshot_factories_validate_inputs_and_repository_lifecycle(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    invalid_corpus = cast("CorpusId", object())
    invalid_version = cast("CorpusVersionId", object())
    try:
        with pytest.raises(TypeError, match="corpus_id"):
            repository.open_active_corpus_snapshot(invalid_corpus)
        with pytest.raises(TypeError, match="corpus_id"):
            repository.open_corpus_snapshot(invalid_corpus, _VERSION_A1)
        with pytest.raises(TypeError, match="corpus_version"):
            repository.open_corpus_snapshot(_CORPUS_A.corpus_id, invalid_version)
        with repository.transaction():
            with pytest.raises(StorageTransactionError):
                repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id)
    finally:
        repository.close()

    with pytest.raises(RepositoryClosedError):
        repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id)
