"""Shared catalogue and update builders for corpus-publication tests."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

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
    SealCorpusUpdateOutcome,
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
_PRIVATE_FAILURE = "private-publication-failure"


class _CommitFailingConnection(sqlite3.Connection):
    """Connection that deterministically rejects the transaction commit."""

    def commit(self) -> None:
        raise sqlite3.OperationalError(_PRIVATE_FAILURE)


class _UncertainCommitConnection(sqlite3.Connection):
    """Connection that commits durably before reporting an ambiguous failure."""

    def commit(self) -> None:
        super().commit()
        raise sqlite3.OperationalError(_PRIVATE_FAILURE)


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


def _execute_schema_statement(path: Path, statement: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


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
