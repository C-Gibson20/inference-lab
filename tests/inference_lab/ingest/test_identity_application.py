"""Acceptance tests for applying discovery identity outcomes atomically."""

from pathlib import Path

import pytest

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveredSource,
    FileIdentityHint,
    FileStatEvidence,
    IdentityApplicationResult,
    SourceClassification,
    SourceLocator,
    SourceSupport,
    StableFileEvidence,
    VerifiedSourceSnapshot,
    apply_discovery_identity_outcome,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    HashAlgorithm,
    IdentityResolutionOutcome,
    ResolutionKind,
    ResolutionReason,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage import (
    SourceFileIdentityHint,
    SourceFingerprint,
    SourceLocatorRevision,
    StorageConflictError,
)
from inference_lab.storage import _sqlite_transaction as sqlite_transaction_module
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_CORPUS = CorpusMetadata(_CORPUS_ID, "Identity application corpus")
_DOCUMENT_ID = DocumentId("22222222-2222-4222-8222-222222222222")
_PRIOR_VERSION_ID = DocumentVersionId("33333333-3333-4333-8333-333333333333")
_NEXT_VERSION_ID = DocumentVersionId("44444444-4444-4444-8444-444444444444")
_NEW_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_NEW_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")
_SOURCE_ID = SourceObservationId("77777777-7777-4777-8777-777777777777")
_NEW_SOURCE_ID = SourceObservationId("88888888-8888-4888-8888-888888888888")
_HASH_A = ContentHash(HashAlgorithm.SHA256, "a" * 64)
_HASH_B = ContentHash(HashAlgorithm.SHA256, "b" * 64)
_DOI = ExternalIdentifier("doi", "10.1000/example")
_PMID = ExternalIdentifier("pmid", "12345678")
_PRIOR_VERSION = DocumentVersion(
    _PRIOR_VERSION_ID,
    _DOCUMENT_ID,
    _HASH_A,
    (_DOI,),
)


def _open_repository(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(path, apply_migrations=True)


def _snapshot(path: Path, content_hash: ContentHash) -> VerifiedSourceSnapshot:
    evidence = FileStatEvidence(
        byte_count=42,
        modified_time_ns=123_456_789,
        identity_hint=FileIdentityHint(7, 11),
    )
    return VerifiedSourceSnapshot(
        locator=SourceLocator(path, 0, Path(path.name)),
        media_type="text/plain",
        encoding="utf-8",
        byte_count=42,
        content_hash=content_hash,
        file_evidence=StableFileEvidence(evidence, evidence, evidence, evidence),
    )


def _classification() -> SourceClassification:
    return SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
        classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
        parser_capability="text-native",
        encoding="utf-8",
    )


def _source(
    *,
    source_id: SourceObservationId,
    path: Path,
    content_hash: ContentHash,
    outcome: IdentityResolutionOutcome,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
) -> DiscoveredSource:
    return DiscoveredSource(
        ordinal=0,
        corpus_id=_CORPUS_ID,
        policy=CURRENT_DISCOVERY_POLICY,
        source_observation_id=source_id,
        snapshot=_snapshot(path, content_hash),
        external_identifiers=external_identifiers,
        family_candidates=(),
        classification=_classification(),
        identity_outcome=outcome,
    )


def _locator(
    path: Path,
    *,
    source_id: SourceObservationId = _SOURCE_ID,
    content_hash: ContentHash = _HASH_A,
) -> SourceLocatorRevision:
    return SourceLocatorRevision(
        corpus_id=_CORPUS_ID,
        source_observation_id=source_id,
        locator_revision=1,
        path=path,
        identity_hint=SourceFileIdentityHint(7, 11),
        fingerprint=SourceFingerprint(content_hash, 42, 123_456_789),
    )


def _add_corpus(repository: SQLiteMetadataRepository) -> None:
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS)


def _add_prior_observation(
    repository: SQLiteMetadataRepository,
    path: Path,
) -> None:
    with repository.transaction() as transaction:
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(_PRIOR_VERSION)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS_ID,
                _DOCUMENT_ID,
                _PRIOR_VERSION_ID,
            )
        )
        transaction.add_source_observation(
            SourceObservationState(
                _CORPUS_ID,
                _SOURCE_ID,
                1,
                _DOCUMENT_ID,
                _PRIOR_VERSION_ID,
            )
        )
        transaction.add_source_locator(_locator(path))


def _apply(
    repository: SQLiteMetadataRepository,
    source: DiscoveredSource,
) -> IdentityApplicationResult:
    result: IdentityApplicationResult | None = None
    with repository.transaction() as transaction:
        result = apply_discovery_identity_outcome(
            source,
            corpus_id=_CORPUS_ID,
            transaction=transaction,
        )
    assert result is not None
    return result


def test_external_identifier_new_version_creates_new_observation(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_corpus(repository)
    with repository.transaction() as transaction:
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(_PRIOR_VERSION)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS_ID,
                _DOCUMENT_ID,
                _PRIOR_VERSION_ID,
            )
        )
    path = (tmp_path / "related.txt").resolve()
    version = DocumentVersion(_NEXT_VERSION_ID, _DOCUMENT_ID, _HASH_B, (_DOI,))
    source = _source(
        source_id=_NEW_SOURCE_ID,
        path=path,
        content_hash=_HASH_B,
        external_identifiers=(_DOI,),
        outcome=IdentityResolutionOutcome(
            kind=ResolutionKind.NEW_VERSION,
            reason=ResolutionReason.EXTERNAL_IDENTIFIER_MATCH,
            document=DocumentIdentity(_DOCUMENT_ID),
            version=version,
            matched_version_ids=(_PRIOR_VERSION_ID,),
        ),
    )
    try:
        result = _apply(repository, source)

        assert result.created_document_version
        assert repository.get_source_observation(
            _CORPUS_ID,
            _NEW_SOURCE_ID,
        ) == SourceObservationState(
            _CORPUS_ID,
            _NEW_SOURCE_ID,
            1,
            _DOCUMENT_ID,
            _NEXT_VERSION_ID,
        )
        assert repository.get_source_locator(_CORPUS_ID, _NEW_SOURCE_ID) == _locator(
            path,
            source_id=_NEW_SOURCE_ID,
            content_hash=_HASH_B,
        )
    finally:
        repository.close()


def test_unchanged_adds_assertion_without_observation_or_locator_revision(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_corpus(repository)
    path = (tmp_path / "same.txt").resolve()
    _add_prior_observation(repository, path)
    assertion = ExternalIdentifierAssertion(_SOURCE_ID, _PRIOR_VERSION_ID, _PMID)
    source = _source(
        source_id=_SOURCE_ID,
        path=path,
        content_hash=_HASH_A,
        external_identifiers=(_DOI, _PMID),
        outcome=IdentityResolutionOutcome(
            kind=ResolutionKind.UNCHANGED,
            reason=ResolutionReason.EXACT_HASH_MATCH,
            document=DocumentIdentity(_DOCUMENT_ID),
            version=_PRIOR_VERSION,
            matched_version_ids=(_PRIOR_VERSION_ID,),
            identifier_assertions_to_add=(assertion,),
        ),
    )
    try:
        result = _apply(repository, source)

        assert result == IdentityApplicationResult(
            source,
            created_document_version=False,
        )
        assert repository.list_external_identifier_assertions(
            _CORPUS_ID,
            _PRIOR_VERSION_ID,
        ) == (assertion,)
        assert (
            len(repository.list_source_observation_history(_CORPUS_ID, _SOURCE_ID)) == 1
        )
        assert len(repository.list_source_locator_history(_CORPUS_ID, _SOURCE_ID)) == 1
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        (ResolutionKind.AMBIGUOUS, ResolutionReason.MULTIPLE_CANDIDATES),
        (ResolutionKind.CONFLICT, ResolutionReason.CONFLICTING_EVIDENCE),
    ],
)
def test_unassigned_outcome_withholds_every_write_and_version_creation(
    tmp_path: Path,
    kind: ResolutionKind,
    reason: ResolutionReason,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_corpus(repository)
    source = _source(
        source_id=_NEW_SOURCE_ID,
        path=(tmp_path / "withheld.txt").resolve(),
        content_hash=_HASH_A,
        outcome=IdentityResolutionOutcome(
            kind=kind,
            reason=reason,
            matched_version_ids=(
                (_PRIOR_VERSION_ID,) if kind is ResolutionKind.AMBIGUOUS else ()
            ),
        ),
    )
    try:
        result = _apply(repository, source)

        assert result == IdentityApplicationResult(
            source,
            created_document_version=False,
        )
        assert repository.list_corpus_document_version_memberships(_CORPUS_ID) == ()
        assert (
            repository.get_source_observation(
                _CORPUS_ID,
                _NEW_SOURCE_ID,
            )
            is None
        )
        assert (
            repository.get_source_locator(
                _CORPUS_ID,
                _NEW_SOURCE_ID,
            )
            is None
        )
    finally:
        repository.close()


def _new_document_source(path: Path) -> DiscoveredSource:
    version = DocumentVersion(_NEW_VERSION_ID, _NEW_DOCUMENT_ID, _HASH_A)
    return _source(
        source_id=_NEW_SOURCE_ID,
        path=path,
        content_hash=_HASH_A,
        outcome=IdentityResolutionOutcome(
            kind=ResolutionKind.NEW_DOCUMENT,
            reason=ResolutionReason.NO_MATCH,
            document=DocumentIdentity(_NEW_DOCUMENT_ID),
            version=version,
        ),
    )


def _fail_write(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
    primary: StorageConflictError,
) -> None:
    def fail_operation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise primary

    monkeypatch.setattr(sqlite_transaction_module, operation_name, fail_operation)


def _assert_new_identity_absent(repository: SQLiteMetadataRepository) -> None:
    assert repository.get_document(_NEW_DOCUMENT_ID) is None
    assert repository.get_document_version(_NEW_VERSION_ID) is None
    assert (
        repository.get_corpus_document_version_membership(
            _CORPUS_ID,
            _NEW_VERSION_ID,
        )
        is None
    )
    assert repository.get_source_observation(_CORPUS_ID, _NEW_SOURCE_ID) is None
    assert repository.get_source_locator(_CORPUS_ID, _NEW_SOURCE_ID) is None


def test_new_document_late_write_failure_rolls_back_complete_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_corpus(repository)
    source = _new_document_source((tmp_path / "new.txt").resolve())
    primary = StorageConflictError()
    _fail_write(monkeypatch, "add_source_locator", primary)
    try:
        with pytest.raises(StorageConflictError) as raised:
            _apply(repository, source)

        assert raised.value is primary
        _assert_new_identity_absent(repository)
    finally:
        repository.close()
