from datetime import UTC, datetime
from pathlib import Path

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
    HashAlgorithm,
    IdentityResolutionRequest,
    SourceObservationId,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 2, 9, 0, tzinfo=UTC)
_CORPUS_ID = CorpusId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_CORPUS = CorpusMetadata(_CORPUS_ID, "Candidate test corpus")
_SHARED_HASH = ContentHash(HashAlgorithm.SHA256, "a" * 64)
_OTHER_HASH = ContentHash(HashAlgorithm.SHA256, "b" * 64)
_FIRST_DOCUMENT_ID = DocumentId("11111111-1111-4111-8111-111111111111")
_SECOND_DOCUMENT_ID = DocumentId("22222222-2222-4222-8222-222222222222")
_THIRD_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_SOURCE_OBSERVATION_ID = SourceObservationId("44444444-4444-4444-8444-444444444444")

_FIRST = DocumentVersion(
    DocumentVersionId("99999999-9999-4999-8999-999999999999"),
    _FIRST_DOCUMENT_ID,
    _SHARED_HASH,
    (
        ExternalIdentifier("doi", "10.1000/shared"),
        ExternalIdentifier("pmid", "first-only"),
    ),
)
_SECOND = DocumentVersion(
    DocumentVersionId("11111111-1111-4111-8111-111111111111"),
    _SECOND_DOCUMENT_ID,
    _SHARED_HASH,
    (ExternalIdentifier("doi", "10.1000/shared"),),
)
_THIRD = DocumentVersion(
    DocumentVersionId("55555555-5555-4555-8555-555555555555"),
    _THIRD_DOCUMENT_ID,
    _OTHER_HASH,
    (
        ExternalIdentifier("arxiv", "third-only"),
        ExternalIdentifier("doi", "10.1000/other"),
    ),
)


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _populated_repository(path: Path) -> SQLiteMetadataRepository:
    repository = SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS)
        for version in (_THIRD, _SECOND, _FIRST):
            transaction.add_document(DocumentIdentity(version.document_id))
            transaction.add_document_version(version)
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _CORPUS_ID,
                    version.document_id,
                    version.document_version_id,
                )
            )
    return repository


def test_content_hash_lookup_returns_canonical_resolution_candidates(
    tmp_path: Path,
) -> None:
    repository = _populated_repository(tmp_path / "metadata.sqlite3")
    try:
        candidates = repository.find_document_versions_by_content_hash(
            _CORPUS_ID, _SHARED_HASH
        )

        assert candidates == (_FIRST, _SECOND)
        assert (
            IdentityResolutionRequest(
                _SHARED_HASH,
                _SOURCE_OBSERVATION_ID,
                content_hash_candidates=candidates,
            ).content_hash_candidates
            == candidates
        )
        assert repository.find_document_versions_by_content_hash(
            _CORPUS_ID, _OTHER_HASH
        ) == (_THIRD,)
        assert (
            repository.find_document_versions_by_content_hash(
                _CORPUS_ID, ContentHash(HashAlgorithm.SHA256, "f" * 64)
            )
            == ()
        )
    finally:
        repository.close()


def test_external_identifier_lookup_matches_any_and_removes_duplicates(
    tmp_path: Path,
) -> None:
    repository = _populated_repository(tmp_path / "metadata.sqlite3")
    supplied = (
        ExternalIdentifier("doi", "10.1000/shared"),
        ExternalIdentifier("pmid", "first-only"),
    )
    try:
        candidates = repository.find_document_versions_by_external_identifiers(
            _CORPUS_ID, supplied
        )

        assert candidates == (_FIRST, _SECOND)
        assert (
            IdentityResolutionRequest(
                _OTHER_HASH,
                _SOURCE_OBSERVATION_ID,
                external_identifiers=supplied,
                external_identifier_candidates=candidates,
            ).external_identifier_candidates
            == candidates
        )
    finally:
        repository.close()


def test_external_identifier_lookup_covers_zero_one_and_conflicting_matches(
    tmp_path: Path,
) -> None:
    repository = _populated_repository(tmp_path / "metadata.sqlite3")
    try:
        assert (
            repository.find_document_versions_by_external_identifiers(_CORPUS_ID, ())
            == ()
        )
        assert repository.find_document_versions_by_external_identifiers(
            _CORPUS_ID, (ExternalIdentifier("pmid", "first-only"),)
        ) == (_FIRST,)
        assert (
            repository.find_document_versions_by_external_identifiers(
                _CORPUS_ID, (ExternalIdentifier("pmid", "missing"),)
            )
            == ()
        )
        assert repository.find_document_versions_by_external_identifiers(
            _CORPUS_ID,
            (
                ExternalIdentifier("arxiv", "third-only"),
                ExternalIdentifier("doi", "10.1000/shared"),
            ),
        ) == (_FIRST, _SECOND, _THIRD)
    finally:
        repository.close()


def test_transaction_queries_include_its_uncommitted_version(tmp_path: Path) -> None:
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_CORPUS)
            transaction.add_document(DocumentIdentity(_FIRST.document_id))
            transaction.add_document_version(_FIRST)
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _CORPUS_ID,
                    _FIRST.document_id,
                    _FIRST.document_version_id,
                )
            )

            assert transaction.find_document_versions_by_content_hash(
                _CORPUS_ID, _SHARED_HASH
            ) == (_FIRST,)
            assert transaction.find_document_versions_by_external_identifiers(
                _CORPUS_ID, (ExternalIdentifier("doi", "10.1000/shared"),)
            ) == (_FIRST,)
    finally:
        repository.close()
