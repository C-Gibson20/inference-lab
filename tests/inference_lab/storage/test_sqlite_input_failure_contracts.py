"""Negative input contracts for the SQLite repository and transactions."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateRelationship,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage import SourceFileIdentityHint, SourceLocatorRevision
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from inference_lab.storage._sqlite_transaction import SQLiteMetadataTransaction

from .fixtures import open_repository as _open_repository  # noqa: TID252

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_CORPUS = CorpusMetadata(_CORPUS_ID, "Input failure corpus")
_VERSION_ID = DocumentVersionId("22222222-2222-4222-8222-222222222222")
_OTHER_VERSION_ID = DocumentVersionId("33333333-3333-4333-8333-333333333333")
_SOURCE_ID = SourceObservationId("44444444-4444-4444-8444-444444444444")
_OTHER_SOURCE_ID = SourceObservationId("55555555-5555-4555-8555-555555555555")
_IDENTIFIER = ExternalIdentifier("doi", "10.1000/input-failure")

type _ReadOwner = SQLiteMetadataRepository | SQLiteMetadataTransaction
type _ReadOperation = Callable[[_ReadOwner], object]
type _WriteOperation = Callable[[SQLiteMetadataTransaction], object]


def _invalid[Value](value_type: type[Value]) -> Value:
    """Return a deliberately wrong runtime value with a requested static type."""
    del value_type
    return cast("Value", object())


_READ_FAILURE_CASES: tuple[tuple[str, _ReadOperation, str], ...] = (
    (
        "get-corpus",
        lambda owner: owner.get_corpus(_invalid(CorpusId)),
        "corpus_id",
    ),
    (
        "get-membership-corpus",
        lambda owner: owner.get_corpus_document_version_membership(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "get-membership-version",
        lambda owner: owner.get_corpus_document_version_membership(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "list-memberships",
        lambda owner: owner.list_corpus_document_version_memberships(
            _invalid(CorpusId)
        ),
        "corpus_id",
    ),
    (
        "get-document",
        lambda owner: owner.get_document(_invalid(DocumentId)),
        "document_id",
    ),
    (
        "get-family",
        lambda owner: owner.get_document_family(_invalid(DocumentFamilyId)),
        "family_id",
    ),
    (
        "get-version",
        lambda owner: owner.get_document_version(_invalid(DocumentVersionId)),
        "document_version_id",
    ),
    (
        "list-versions",
        lambda owner: owner.list_document_versions(_invalid(DocumentId)),
        "document_id",
    ),
    (
        "find-hash-corpus",
        lambda owner: owner.find_document_versions_by_content_hash(
            _invalid(CorpusId), _invalid(ContentHash)
        ),
        "corpus_id",
    ),
    (
        "find-hash-value",
        lambda owner: owner.find_document_versions_by_content_hash(
            _CORPUS_ID, _invalid(ContentHash)
        ),
        "content_hash",
    ),
    (
        "find-identifiers-corpus",
        lambda owner: owner.find_document_versions_by_external_identifiers(
            _invalid(CorpusId), ()
        ),
        "corpus_id",
    ),
    (
        "source-current-corpus",
        lambda owner: owner.get_source_observation(_invalid(CorpusId), _SOURCE_ID),
        "corpus_id",
    ),
    (
        "source-current-id",
        lambda owner: owner.get_source_observation(
            _CORPUS_ID, _invalid(SourceObservationId)
        ),
        "source_observation_id",
    ),
    (
        "source-history-corpus",
        lambda owner: owner.list_source_observation_history(
            _invalid(CorpusId), _SOURCE_ID
        ),
        "corpus_id",
    ),
    (
        "source-history-id",
        lambda owner: owner.list_source_observation_history(
            _CORPUS_ID, _invalid(SourceObservationId)
        ),
        "source_observation_id",
    ),
    (
        "source-by-version-corpus",
        lambda owner: owner.find_source_observations_by_document_version(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "source-by-version-version",
        lambda owner: owner.find_source_observations_by_document_version(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "locator-current-corpus",
        lambda owner: owner.get_source_locator(_invalid(CorpusId), _SOURCE_ID),
        "corpus_id",
    ),
    (
        "locator-current-source",
        lambda owner: owner.get_source_locator(
            _CORPUS_ID,
            _invalid(SourceObservationId),
        ),
        "source_observation_id",
    ),
    (
        "locator-history-corpus",
        lambda owner: owner.list_source_locator_history(
            _invalid(CorpusId),
            _SOURCE_ID,
        ),
        "corpus_id",
    ),
    (
        "locator-history-source",
        lambda owner: owner.list_source_locator_history(
            _CORPUS_ID,
            _invalid(SourceObservationId),
        ),
        "source_observation_id",
    ),
    (
        "locator-path-corpus",
        lambda owner: owner.find_source_locator_by_path(
            _invalid(CorpusId),
            Path("/private/source.txt"),
        ),
        "corpus_id",
    ),
    (
        "locator-path-value",
        lambda owner: owner.find_source_locator_by_path(
            _CORPUS_ID,
            _invalid(Path),
        ),
        "path",
    ),
    (
        "locator-hint-corpus",
        lambda owner: owner.find_source_locators_by_identity_hint(
            _invalid(CorpusId),
            SourceFileIdentityHint(1, 2),
        ),
        "corpus_id",
    ),
    (
        "locator-hint-value",
        lambda owner: owner.find_source_locators_by_identity_hint(
            _CORPUS_ID,
            _invalid(SourceFileIdentityHint),
        ),
        "identity_hint",
    ),
    (
        "assertions-corpus",
        lambda owner: owner.list_external_identifier_assertions(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "assertions-version",
        lambda owner: owner.list_external_identifier_assertions(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "duplicates-corpus",
        lambda owner: owner.list_duplicate_relationships(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "duplicates-version",
        lambda owner: owner.list_duplicate_relationships(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "families-by-family-corpus",
        lambda owner: owner.list_family_memberships_by_family(
            _invalid(CorpusId), _invalid(DocumentFamilyId)
        ),
        "corpus_id",
    ),
    (
        "families-by-family-id",
        lambda owner: owner.list_family_memberships_by_family(
            _CORPUS_ID, _invalid(DocumentFamilyId)
        ),
        "family_id",
    ),
    (
        "families-by-document-corpus",
        lambda owner: owner.list_family_memberships_by_document(
            _invalid(CorpusId), _invalid(DocumentId)
        ),
        "corpus_id",
    ),
    (
        "families-by-document-id",
        lambda owner: owner.list_family_memberships_by_document(
            _CORPUS_ID, _invalid(DocumentId)
        ),
        "document_id",
    ),
    (
        "families-by-version-corpus",
        lambda owner: owner.list_family_memberships_by_document_version(
            _invalid(CorpusId), _invalid(DocumentVersionId)
        ),
        "corpus_id",
    ),
    (
        "families-by-version-id",
        lambda owner: owner.list_family_memberships_by_document_version(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
)


@pytest.mark.parametrize(
    ("operation", "expected_field"),
    [(operation, field) for _, operation, field in _READ_FAILURE_CASES],
    ids=[name for name, _, _ in _READ_FAILURE_CASES],
)
def test_repository_reads_reject_wrong_identity_types(
    tmp_path: Path,
    operation: _ReadOperation,
    expected_field: str,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(TypeError, match=expected_field):
            operation(repository)
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("operation", "expected_field"),
    [(operation, field) for _, operation, field in _READ_FAILURE_CASES],
    ids=[name for name, _, _ in _READ_FAILURE_CASES],
)
def test_transaction_read_failures_roll_back_prior_writes(
    tmp_path: Path,
    operation: _ReadOperation,
    expected_field: str,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(TypeError, match=expected_field):
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                operation(transaction)

        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()


_WRITE_FAILURE_CASES: tuple[tuple[str, _WriteOperation, str], ...] = (
    (
        "corpus",
        lambda transaction: transaction.add_corpus(_invalid(CorpusMetadata)),
        "corpus",
    ),
    (
        "corpus-name-id",
        lambda transaction: transaction.set_corpus_display_name(
            _invalid(CorpusId), "Renamed"
        ),
        "corpus_id",
    ),
    (
        "corpus-membership",
        lambda transaction: transaction.add_corpus_document_version_membership(
            _invalid(CorpusDocumentVersionMembership)
        ),
        "membership",
    ),
    (
        "document",
        lambda transaction: transaction.add_document(_invalid(DocumentIdentity)),
        "document",
    ),
    (
        "family",
        lambda transaction: transaction.add_document_family(_invalid(DocumentFamilyId)),
        "family_id",
    ),
    (
        "version",
        lambda transaction: transaction.add_document_version(_invalid(DocumentVersion)),
        "document_version",
    ),
    (
        "source",
        lambda transaction: transaction.add_source_observation(
            _invalid(SourceObservationState)
        ),
        "state",
    ),
    (
        "advance-corpus",
        lambda transaction: transaction.advance_source_observation(
            _invalid(CorpusId),
            _SOURCE_ID,
            expected_version_id=_VERSION_ID,
            new_version_id=_OTHER_VERSION_ID,
        ),
        "corpus_id",
    ),
    (
        "advance-source",
        lambda transaction: transaction.advance_source_observation(
            _CORPUS_ID,
            _invalid(SourceObservationId),
            expected_version_id=_VERSION_ID,
            new_version_id=_OTHER_VERSION_ID,
        ),
        "source_observation_id",
    ),
    (
        "advance-expected",
        lambda transaction: transaction.advance_source_observation(
            _CORPUS_ID,
            _SOURCE_ID,
            expected_version_id=_invalid(DocumentVersionId),
            new_version_id=_OTHER_VERSION_ID,
        ),
        "expected_version_id",
    ),
    (
        "advance-new",
        lambda transaction: transaction.advance_source_observation(
            _CORPUS_ID,
            _SOURCE_ID,
            expected_version_id=_VERSION_ID,
            new_version_id=_invalid(DocumentVersionId),
        ),
        "new_version_id",
    ),
    (
        "locator-add",
        lambda transaction: transaction.add_source_locator(
            _invalid(SourceLocatorRevision)
        ),
        "revision",
    ),
    (
        "locator-advance",
        lambda transaction: transaction.advance_source_locator(
            _invalid(SourceLocatorRevision)
        ),
        "revision",
    ),
    (
        "assertion-corpus",
        lambda transaction: transaction.add_external_identifier_assertions(
            _invalid(CorpusId), ()
        ),
        "corpus_id",
    ),
    (
        "duplicate",
        lambda transaction: transaction.add_duplicate_relationship(
            _invalid(DuplicateRelationship)
        ),
        "relationship",
    ),
    (
        "family-membership",
        lambda transaction: transaction.add_family_membership(
            _invalid(DocumentFamilyMembership)
        ),
        "membership",
    ),
)


@pytest.mark.parametrize(
    ("operation", "expected_field"),
    [(operation, field) for _, operation, field in _WRITE_FAILURE_CASES],
    ids=[name for name, _, _ in _WRITE_FAILURE_CASES],
)
def test_transaction_writes_reject_wrong_types_and_roll_back(
    tmp_path: Path,
    operation: _WriteOperation,
    expected_field: str,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(TypeError, match=expected_field):
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                operation(transaction)

        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()


def _assertion(
    source_id: SourceObservationId,
    version_id: DocumentVersionId,
    namespace: str,
) -> ExternalIdentifierAssertion:
    """Build a deterministic assertion for malformed-batch tests."""
    return ExternalIdentifierAssertion(
        source_id,
        version_id,
        ExternalIdentifier(namespace, f"value-{namespace}"),
    )


@pytest.mark.parametrize(
    ("assertions", "expected_error", "expected_message"),
    [
        ([], TypeError, "tuple"),
        ((object(),), TypeError, "ExternalIdentifierAssertion"),
        (
            (
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
            ),
            ValueError,
            "duplicates",
        ),
        (
            (
                _assertion(_SOURCE_ID, _VERSION_ID, "pmid"),
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
            ),
            ValueError,
            "canonical order",
        ),
        (
            (
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
                _assertion(_OTHER_SOURCE_ID, _VERSION_ID, "pmid"),
            ),
            ValueError,
            "one source-version association",
        ),
    ],
    ids=["list", "wrong-member", "duplicate", "order", "mixed-association"],
)
def test_malformed_assertion_batches_fail_before_persistence(
    tmp_path: Path,
    assertions: object,
    expected_error: type[Exception],
    expected_message: str,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(expected_error, match=expected_message):
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                transaction.add_external_identifier_assertions(
                    _CORPUS_ID,
                    cast("tuple[ExternalIdentifierAssertion, ...]", assertions),
                )

        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()
