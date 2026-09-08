"""Shared domain fixtures for corpus-update component tests."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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
    ComponentKind,
    CorpusVersionManifest,
    SealCorpusUpdateOutcome,
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
