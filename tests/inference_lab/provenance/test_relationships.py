"""Tests for immutable duplicate and family relationships."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    InvalidRelationshipError,
)

FIRST_VERSION_ID = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000001")
SECOND_VERSION_ID = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000002")
FAMILY_ID = DocumentFamilyId.from_str("00000000-0000-4000-8000-000000000003")
DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000004")


def test_duplicate_relationship_requires_canonical_endpoint_order() -> None:
    relationship = DuplicateRelationship(
        first_version_id=FIRST_VERSION_ID,
        second_version_id=SECOND_VERSION_ID,
        duplicate_class=DuplicateClass.BYTE_IDENTICAL,
    )

    assert relationship.key() == (
        "byte_identical",
        FIRST_VERSION_ID.as_str(),
        SECOND_VERSION_ID.as_str(),
    )


def test_duplicate_relationship_rejects_reversed_endpoints() -> None:
    with pytest.raises(InvalidRelationshipError, match="canonical order"):
        DuplicateRelationship(
            first_version_id=SECOND_VERSION_ID,
            second_version_id=FIRST_VERSION_ID,
            duplicate_class=DuplicateClass.BYTE_IDENTICAL,
        )


def test_duplicate_relationship_rejects_self_link() -> None:
    with pytest.raises(InvalidRelationshipError, match="itself"):
        DuplicateRelationship(
            first_version_id=FIRST_VERSION_ID,
            second_version_id=FIRST_VERSION_ID,
            duplicate_class=DuplicateClass.BYTE_IDENTICAL,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("first_version_id", None, "first_version_id"),
        ("second_version_id", None, "second_version_id"),
        ("duplicate_class", "byte_identical", "duplicate_class"),
    ],
)
def test_duplicate_relationship_rejects_wrong_field_types(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "first_version_id": FIRST_VERSION_ID,
        "second_version_id": SECOND_VERSION_ID,
        "duplicate_class": DuplicateClass.BYTE_IDENTICAL,
    }
    values[field] = value

    with pytest.raises(TypeError, match=message):
        DuplicateRelationship(**values)  # type: ignore[arg-type]


def test_duplicate_relationship_is_immutable_and_hashable() -> None:
    relationship = DuplicateRelationship(
        FIRST_VERSION_ID,
        SECOND_VERSION_ID,
        DuplicateClass.BYTE_IDENTICAL,
    )

    assert relationship == relationship
    assert len({relationship}) == 1

    with pytest.raises(FrozenInstanceError):
        relationship.duplicate_class = DuplicateClass.BYTE_IDENTICAL  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("family_id", None, "family_id"),
        ("document_id", None, "document_id"),
        ("document_version_id", "version-id", "document_version_id"),
    ],
)
def test_family_membership_rejects_wrong_field_types(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "family_id": FAMILY_ID,
        "document_id": DOCUMENT_ID,
        "document_version_id": None,
    }
    values[field] = value

    with pytest.raises(TypeError, match=message):
        DocumentFamilyMembership(**values)  # type: ignore[arg-type]


def test_family_membership_supports_document_and_version_scope() -> None:
    document_membership = DocumentFamilyMembership(FAMILY_ID, DOCUMENT_ID)
    version_membership = DocumentFamilyMembership(
        FAMILY_ID,
        DOCUMENT_ID,
        FIRST_VERSION_ID,
    )

    assert document_membership.key() == (
        FAMILY_ID.as_str(),
        DOCUMENT_ID.as_str(),
        None,
    )
    assert version_membership.key() == (
        FAMILY_ID.as_str(),
        DOCUMENT_ID.as_str(),
        FIRST_VERSION_ID.as_str(),
    )


def test_family_membership_is_immutable() -> None:
    membership = DocumentFamilyMembership(FAMILY_ID, DOCUMENT_ID)

    with pytest.raises(FrozenInstanceError):
        membership.family_id = FAMILY_ID  # type: ignore[misc]


def test_corpus_document_version_membership_requires_typed_identifiers() -> None:
    values: dict[str, object] = {
        "corpus_id": CorpusId.generate(),
        "document_id": DocumentId.generate(),
        "document_version_id": DocumentVersionId.generate(),
    }

    membership = CorpusDocumentVersionMembership(**values)  # type: ignore[arg-type]
    assert membership.corpus_id == values["corpus_id"]

    for field in values:
        invalid = values | {field: None}
        with pytest.raises(TypeError, match=field):
            CorpusDocumentVersionMembership(**invalid)  # type: ignore[arg-type]
