"""Tests for rollback publication as a new monotonic corpus version."""

from dataclasses import replace
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
    InvalidCorpusVersionContractError,
    RollbackCorpusVersionOutcome,
    SealCorpusUpdateOutcome,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CREATED_AT = datetime(2026, 9, 6, 9, 0, tzinfo=UTC)
_CORPUS = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"),
    "Private rollback corpus",
)
_OTHER_CORPUS = CorpusMetadata(
    CorpusId("12121212-1212-4212-8212-121212121212"),
    "Other private rollback corpus",
)
_VERSION_1 = CorpusVersionId("22222222-2222-4222-8222-222222222222")
_VERSION_2 = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_VERSION_3 = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_VERSION_4 = CorpusVersionId("55555555-5555-4555-8555-555555555555")
_UNKNOWN_VERSION = CorpusVersionId("66666666-6666-4666-8666-666666666666")
_OTHER_ACTIVE_VERSION = CorpusVersionId("13131313-1313-4313-8313-131313131313")
_DOCUMENT_1_ID = DocumentId("77777777-7777-4777-8777-777777777777")
_DOCUMENT_2_ID = DocumentId("88888888-8888-4888-8888-888888888888")
_DOCUMENT_VERSION_1_ID = DocumentVersionId("99999999-9999-4999-8999-999999999999")
_DOCUMENT_VERSION_2_ID = DocumentVersionId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_DOCUMENT_VERSION_1 = DocumentVersion(
    _DOCUMENT_VERSION_1_ID,
    _DOCUMENT_1_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_DOCUMENT_VERSION_2 = DocumentVersion(
    _DOCUMENT_VERSION_2_ID,
    _DOCUMENT_2_ID,
    ContentHash(HashAlgorithm.SHA256, "b" * 64),
)
_MEMBERSHIP_1 = CorpusDocumentVersionMembership(
    _CORPUS.corpus_id,
    _DOCUMENT_1_ID,
    _DOCUMENT_VERSION_1_ID,
)
_MEMBERSHIP_2 = CorpusDocumentVersionMembership(
    _CORPUS.corpus_id,
    _DOCUMENT_2_ID,
    _DOCUMENT_VERSION_2_ID,
)
_LEXICAL = ComponentKind("lexical")
_DENSE = ComponentKind("dense")
_PRIVATE_CANARY = "private-rollback-canary"


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


def _generation(
    corpus_version: CorpusVersionId,
    generation_id: str,
    *,
    configuration_digest: str = "c" * 64,
) -> ComponentGenerationReference:
    return ComponentGenerationReference(
        component_kind=_LEXICAL,
        generation_id=generation_id,
        implementation="fake-index",
        format_version="1.0.0",
        contract_version="1.0.0",
        configuration_digest=configuration_digest,
        source_manifest_digest="d" * 64,
        corpus_id=_CORPUS.corpus_id,
        corpus_version=corpus_version,
        state=ComponentGenerationState.PREPARING,
    )


def _add_catalogue(repository: SQLiteMetadataRepository) -> None:
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS)
        transaction.add_corpus(_OTHER_CORPUS)
        for document_id, version, membership in (
            (_DOCUMENT_1_ID, _DOCUMENT_VERSION_1, _MEMBERSHIP_1),
            (_DOCUMENT_2_ID, _DOCUMENT_VERSION_2, _MEMBERSHIP_2),
        ):
            transaction.add_document(DocumentIdentity(document_id))
            transaction.add_document_version(version)
            transaction.add_corpus_document_version_membership(membership)


def _publish(
    repository: SQLiteMetadataRepository,
    corpus_version: CorpusVersionId,
    *,
    expected_corpus_version: CorpusVersionId | None,
    memberships_to_add: tuple[CorpusDocumentVersionMembership, ...],
    generation: ComponentGenerationReference | None = None,
) -> CommitCorpusUpdateOutcome:
    committed: CommitCorpusUpdateOutcome | None = None
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS.corpus_id,
            corpus_version,
            expected_corpus_version=expected_corpus_version,
            required_component_kinds=(generation.component_kind,)
            if generation is not None
            else (),
        )
        for membership in memberships_to_add:
            transaction.add_corpus_update_document_membership(
                _CORPUS.corpus_id,
                corpus_version,
                membership,
            )
        if generation is not None:
            transaction.register_corpus_update_component_generation(
                _CORPUS.corpus_id,
                corpus_version,
                generation,
                validator=_AcceptingComponentValidator(),
            )
        sealed = transaction.seal_corpus_update(
            _CORPUS.corpus_id,
            corpus_version,
        )
        assert isinstance(sealed, SealCorpusUpdateOutcome)
        committed = transaction.commit_corpus_update(
            _CORPUS.corpus_id,
            corpus_version,
        )
    assert committed is not None
    return committed


def _component_semantics(
    generation: ComponentGenerationReference,
) -> tuple[object, ...]:
    return (
        generation.component_kind,
        generation.implementation,
        generation.format_version,
        generation.contract_version,
        generation.configuration_digest,
        generation.source_manifest_digest,
    )


def _seed_component_history(
    repository: SQLiteMetadataRepository,
) -> tuple[CommitCorpusUpdateOutcome, CommitCorpusUpdateOutcome]:
    first = _publish(
        repository,
        _VERSION_1,
        expected_corpus_version=None,
        memberships_to_add=(_MEMBERSHIP_1,),
        generation=_generation(_VERSION_1, "generation-v1"),
    )
    second = _publish(
        repository,
        _VERSION_2,
        expected_corpus_version=_VERSION_1,
        memberships_to_add=(_MEMBERSHIP_2,),
        generation=_generation(
            _VERSION_2,
            "generation-v2",
            configuration_digest="e" * 64,
        ),
    )
    return first, second


def test_rollback_publishes_new_version_and_exact_retry_survives_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    first, second = _seed_component_history(repository)
    rollback_generation = _generation(_VERSION_3, "generation-v3")
    outcome: RollbackCorpusVersionOutcome | None = None
    try:
        with capture_logs() as logs:
            with repository.transaction() as transaction:
                begun = transaction.begin_corpus_update(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                    expected_corpus_version=_VERSION_2,
                    restores_corpus_version=_VERSION_1,
                )
                assert begun.update.required_component_kinds == (_LEXICAL,)
                assert transaction.list_corpus_update_document_memberships(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                ) == (_MEMBERSHIP_1,)

            with repository.transaction() as transaction:
                transaction.register_corpus_update_component_generation(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                    rollback_generation,
                    validator=_AcceptingComponentValidator(),
                )
                sealed = transaction.seal_corpus_update(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                )
                assert isinstance(sealed, SealCorpusUpdateOutcome)

            with repository.transaction() as transaction:
                outcome = transaction.commit_corpus_rollback(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                )

        assert outcome is not None
        publication = outcome.publication
        assert publication.version.sequence == 3
        assert publication.version.parent_corpus_version == _VERSION_2
        assert publication.version.restores_corpus_version == _VERSION_1
        assert publication.manifest.document_memberships == (_MEMBERSHIP_1,)
        assert publication.manifest.manifest_digest != first.manifest.manifest_digest
        assert publication.manifest.component_generations[0].generation_id == (
            "generation-v3"
        )
        assert _component_semantics(
            publication.manifest.component_generations[0]
        ) == _component_semantics(first.manifest.component_generations[0])
        assert repository.get_active_corpus_version(_CORPUS.corpus_id) == (
            publication.version
        )
        assert repository.list_corpus_versions(_CORPUS.corpus_id) == (
            first.version,
            second.version,
            publication.version,
        )

        with repository.open_corpus_snapshot(
            _CORPUS.corpus_id,
            _VERSION_1,
        ) as historical:
            assert historical.manifest == first.manifest
        with repository.open_corpus_snapshot(
            _CORPUS.corpus_id,
            _VERSION_2,
        ) as previous:
            assert previous.manifest == second.manifest

        assert logs == [
            {
                "event": "corpus_update_started",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS.corpus_id.as_str(),
                "corpus_version": _VERSION_3.as_str(),
                "update_state": "preparing",
                "required_component_count": 1,
                "replayed": False,
                "restores_corpus_version": _VERSION_1.as_str(),
                "log_level": "info",
            },
            {
                "event": "transaction_completed",
                "pipeline_stage": "storage",
                "log_level": "info",
            },
            {
                "event": "corpus_update_ready",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS.corpus_id.as_str(),
                "corpus_version": _VERSION_3.as_str(),
                "update_state": "ready",
                "document_count": 1,
                "component_count": 1,
                "replayed": False,
                "restores_corpus_version": _VERSION_1.as_str(),
                "log_level": "info",
            },
            {
                "event": "transaction_completed",
                "pipeline_stage": "storage",
                "log_level": "info",
            },
            {
                "event": "corpus_version_rollback_committed",
                "pipeline_stage": "storage",
                "corpus_id": _CORPUS.corpus_id.as_str(),
                "corpus_version": _VERSION_3.as_str(),
                "restores_corpus_version": _VERSION_1.as_str(),
                "sequence": 3,
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
        rendered_logs = repr(logs)
        assert _CORPUS.display_name not in rendered_logs
        assert _PRIVATE_CANARY not in rendered_logs
        assert str(database_path) not in rendered_logs
    finally:
        repository.close()

    def unexpected_clock_call() -> datetime:
        raise AssertionError("an exact rollback replay must retain persisted time")

    reopened = SQLiteMetadataRepository.open(
        database_path,
        created_at_factory=unexpected_clock_call,
    )
    try:
        begun: object = None
        sealed: object = None
        replayed: RollbackCorpusVersionOutcome | None = None
        with reopened.transaction() as transaction:
            begun = transaction.begin_corpus_update(
                _CORPUS.corpus_id,
                _VERSION_3,
                expected_corpus_version=_VERSION_2,
                restores_corpus_version=_VERSION_1,
            )
            sealed = transaction.seal_corpus_update(
                _CORPUS.corpus_id,
                _VERSION_3,
            )
            replayed = transaction.commit_corpus_rollback(
                _CORPUS.corpus_id,
                _VERSION_3,
            )

        assert begun is not None
        assert begun.replayed is True
        assert begun.update.state is CorpusUpdateState.COMMITTED
        assert isinstance(sealed, SealCorpusUpdateOutcome)
        assert sealed.replayed is True
        assert replayed is not None
        assert replayed.publication.replayed is True
        assert outcome is not None
        assert replayed.publication.update == outcome.publication.update
        assert replayed.publication.version == outcome.publication.version
        assert replayed.publication.manifest == outcome.publication.manifest
        assert len(reopened.list_corpus_versions(_CORPUS.corpus_id)) == 3
    finally:
        reopened.close()


def test_rollback_rejects_membership_edits_and_component_semantic_drift(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    _seed_component_history(repository)
    with pytest.raises(StorageConflictError):
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS.corpus_id,
                _VERSION_3,
                expected_corpus_version=_VERSION_2,
                required_component_kinds=(_DENSE,),
                restores_corpus_version=_VERSION_1,
            )
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS.corpus_id,
            _VERSION_3,
            expected_corpus_version=_VERSION_2,
            restores_corpus_version=_VERSION_1,
        )

    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_corpus_update_document_membership(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                    _MEMBERSHIP_2,
                )
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.remove_corpus_update_document_membership(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                    _MEMBERSHIP_1,
                )

        incorrect = _generation(
            _VERSION_3,
            "incorrect-rollback-generation",
            configuration_digest="f" * 64,
        )
        with repository.transaction() as transaction:
            transaction.register_corpus_update_component_generation(
                _CORPUS.corpus_id,
                _VERSION_3,
                incorrect,
                validator=_AcceptingComponentValidator(),
            )
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.seal_corpus_update(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                )

        stored = repository.get_corpus_update(_CORPUS.corpus_id, _VERSION_3)
        assert stored is not None
        assert stored.state is CorpusUpdateState.PREPARING
        active = repository.get_active_corpus_version(_CORPUS.corpus_id)
        assert active is not None
        assert active.corpus_version == _VERSION_2

        corrected = replace(
            _generation(_VERSION_3, "corrected-rollback-generation"),
            state=ComponentGenerationState.PREPARING,
        )
        sealed: object = None
        with repository.transaction() as transaction:
            transaction.replace_corpus_update_component_generation(
                _CORPUS.corpus_id,
                _VERSION_3,
                replace(incorrect, state=ComponentGenerationState.READY),
                corrected,
                validator=_AcceptingComponentValidator(),
            )
            sealed = transaction.seal_corpus_update(
                _CORPUS.corpus_id,
                _VERSION_3,
            )
        assert isinstance(sealed, SealCorpusUpdateOutcome)
    finally:
        repository.close()


def test_rollback_requires_owned_historical_target_and_explicit_commit(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    _publish(
        repository,
        _VERSION_1,
        expected_corpus_version=None,
        memberships_to_add=(_MEMBERSHIP_1,),
    )
    _publish(
        repository,
        _VERSION_2,
        expected_corpus_version=_VERSION_1,
        memberships_to_add=(_MEMBERSHIP_2,),
    )
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _OTHER_CORPUS.corpus_id,
            _OTHER_ACTIVE_VERSION,
            expected_corpus_version=None,
        )
        other_sealed = transaction.seal_corpus_update(
            _OTHER_CORPUS.corpus_id,
            _OTHER_ACTIVE_VERSION,
        )
        assert isinstance(other_sealed, SealCorpusUpdateOutcome)
        transaction.commit_corpus_update(
            _OTHER_CORPUS.corpus_id,
            _OTHER_ACTIVE_VERSION,
        )
    try:
        with pytest.raises(InvalidCorpusVersionContractError, match="historical"):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                    expected_corpus_version=_VERSION_2,
                    restores_corpus_version=_VERSION_2,
                )
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                    expected_corpus_version=_VERSION_2,
                    restores_corpus_version=_UNKNOWN_VERSION,
                )
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    _OTHER_CORPUS.corpus_id,
                    _VERSION_4,
                    expected_corpus_version=_OTHER_ACTIVE_VERSION,
                    restores_corpus_version=_VERSION_1,
                )

        sealed: object = None
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS.corpus_id,
                _VERSION_3,
                expected_corpus_version=_VERSION_2,
                restores_corpus_version=_VERSION_1,
            )
            sealed = transaction.seal_corpus_update(
                _CORPUS.corpus_id,
                _VERSION_3,
            )
        assert isinstance(sealed, SealCorpusUpdateOutcome)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.commit_corpus_update(
                    _CORPUS.corpus_id,
                    _VERSION_3,
                )

        rolled_back: RollbackCorpusVersionOutcome | None = None
        with repository.transaction() as transaction:
            rolled_back = transaction.commit_corpus_rollback(
                _CORPUS.corpus_id,
                _VERSION_3,
            )
        assert rolled_back is not None
        assert rolled_back.publication.version.sequence == 3

        ordinary: object = None
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS.corpus_id,
                _VERSION_4,
                expected_corpus_version=_VERSION_3,
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS.corpus_id,
                _VERSION_4,
                _MEMBERSHIP_2,
            )
            ordinary = transaction.seal_corpus_update(
                _CORPUS.corpus_id,
                _VERSION_4,
            )
        assert isinstance(ordinary, SealCorpusUpdateOutcome)
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.commit_corpus_rollback(
                    _CORPUS.corpus_id,
                    _VERSION_4,
                )
    finally:
        repository.close()
