from datetime import UTC, datetime
from pathlib import Path

import pytest

from inference_lab.provenance import (
    ContentHash,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
)
from inference_lab.storage import StorageConflictError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 2, 10, 30, tzinfo=UTC)
_FIRST_FAMILY_ID = DocumentFamilyId("11111111-1111-4111-8111-111111111111")
_SECOND_FAMILY_ID = DocumentFamilyId("22222222-2222-4222-8222-222222222222")
_UNKNOWN_FAMILY_ID = DocumentFamilyId("99999999-9999-4999-8999-999999999999")
_FIRST_DOCUMENT_ID = DocumentId("11111111-1111-4111-8111-111111111111")
_SECOND_DOCUMENT_ID = DocumentId("22222222-2222-4222-8222-222222222222")
_UNKNOWN_DOCUMENT_ID = DocumentId("99999999-9999-4999-8999-999999999999")
_FIRST_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_SECOND_VERSION_ID = DocumentVersionId("22222222-2222-4222-8222-222222222222")

_FIRST_DOCUMENT_MEMBERSHIP = DocumentFamilyMembership(
    _FIRST_FAMILY_ID,
    _FIRST_DOCUMENT_ID,
)
_FIRST_VERSION_MEMBERSHIP = DocumentFamilyMembership(
    _FIRST_FAMILY_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
)
_SECOND_VERSION_MEMBERSHIP = DocumentFamilyMembership(
    _FIRST_FAMILY_ID,
    _FIRST_DOCUMENT_ID,
    _SECOND_VERSION_ID,
)
_OTHER_DOCUMENT_MEMBERSHIP = DocumentFamilyMembership(
    _FIRST_FAMILY_ID,
    _SECOND_DOCUMENT_ID,
)
_SECOND_FAMILY_DOCUMENT_MEMBERSHIP = DocumentFamilyMembership(
    _SECOND_FAMILY_ID,
    _FIRST_DOCUMENT_ID,
)
_SECOND_FAMILY_VERSION_MEMBERSHIP = DocumentFamilyMembership(
    _SECOND_FAMILY_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
)


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _open_repository(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )


def _version(version_id: DocumentVersionId, digest: str) -> DocumentVersion:
    return DocumentVersion(
        version_id,
        _FIRST_DOCUMENT_ID,
        ContentHash(HashAlgorithm.SHA256, digest * 64),
    )


def _add_membership_parents(repository: SQLiteMetadataRepository) -> None:
    with repository.transaction() as transaction:
        transaction.add_document_family(_FIRST_FAMILY_ID)
        transaction.add_document_family(_SECOND_FAMILY_ID)
        transaction.add_document(DocumentIdentity(_FIRST_DOCUMENT_ID))
        transaction.add_document(DocumentIdentity(_SECOND_DOCUMENT_ID))
        transaction.add_document_version(_version(_FIRST_VERSION_ID, "a"))
        transaction.add_document_version(_version(_SECOND_VERSION_ID, "b"))


def test_family_memberships_round_trip_in_canonical_order(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    first = _open_repository(database_path)
    _add_membership_parents(first)
    memberships = (
        _SECOND_FAMILY_VERSION_MEMBERSHIP,
        _OTHER_DOCUMENT_MEMBERSHIP,
        _SECOND_VERSION_MEMBERSHIP,
        _FIRST_VERSION_MEMBERSHIP,
        _SECOND_FAMILY_DOCUMENT_MEMBERSHIP,
        _FIRST_DOCUMENT_MEMBERSHIP,
    )
    with first.transaction() as transaction:
        for membership in memberships:
            transaction.add_family_membership(membership)
        transaction.add_family_membership(_FIRST_DOCUMENT_MEMBERSHIP)
    first.close()

    second = SQLiteMetadataRepository.open(database_path)
    try:
        by_family = second.list_family_memberships_by_family(_FIRST_FAMILY_ID)
        by_document = second.list_family_memberships_by_document(_FIRST_DOCUMENT_ID)
        by_version = second.list_family_memberships_by_document_version(
            _FIRST_VERSION_ID
        )

        assert by_family == (
            _FIRST_DOCUMENT_MEMBERSHIP,
            _FIRST_VERSION_MEMBERSHIP,
            _SECOND_VERSION_MEMBERSHIP,
            _OTHER_DOCUMENT_MEMBERSHIP,
        )
        assert by_document == (
            _FIRST_DOCUMENT_MEMBERSHIP,
            _FIRST_VERSION_MEMBERSHIP,
            _SECOND_VERSION_MEMBERSHIP,
            _SECOND_FAMILY_DOCUMENT_MEMBERSHIP,
            _SECOND_FAMILY_VERSION_MEMBERSHIP,
        )
        assert by_version == (
            _FIRST_VERSION_MEMBERSHIP,
            _SECOND_FAMILY_VERSION_MEMBERSHIP,
        )
        assert not hasattr(by_family[0], "row_id")
        assert not hasattr(by_family[0], "storage_id")
    finally:
        second.close()


def test_unknown_family_document_and_version_have_no_memberships(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        assert repository.list_family_memberships_by_family(_UNKNOWN_FAMILY_ID) == ()
        assert (
            repository.list_family_memberships_by_document(_UNKNOWN_DOCUMENT_ID) == ()
        )
        assert (
            repository.list_family_memberships_by_document_version(
                DocumentVersionId(_UNKNOWN_DOCUMENT_ID.as_str())
            )
            == ()
        )
    finally:
        repository.close()


@pytest.mark.parametrize(
    "membership",
    [
        DocumentFamilyMembership(_UNKNOWN_FAMILY_ID, _FIRST_DOCUMENT_ID),
        DocumentFamilyMembership(_FIRST_FAMILY_ID, _UNKNOWN_DOCUMENT_ID),
        DocumentFamilyMembership(
            _FIRST_FAMILY_ID,
            _SECOND_DOCUMENT_ID,
            _FIRST_VERSION_ID,
        ),
    ],
    ids=["missing-family", "missing-document", "version-owned-by-other-document"],
)
def test_family_membership_constraints_fail_without_persisting(
    tmp_path: Path,
    membership: DocumentFamilyMembership,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_membership_parents(repository)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_family_membership(membership)

        assert repository.list_family_memberships_by_family(membership.family_id) == ()
    finally:
        repository.close()
