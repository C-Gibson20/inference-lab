"""Tests for immutable document identity and version records."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.provenance import (
    ContentHash,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    ExternalIdentifier,
    HashAlgorithm,
)

VALID_DIGEST = "a" * 64


def _content_hash() -> ContentHash:
    return ContentHash(HashAlgorithm.SHA256, VALID_DIGEST)


def _external_identifiers() -> tuple[ExternalIdentifier, ...]:
    return (
        ExternalIdentifier("doi", "10.1000/example"),
        ExternalIdentifier("publication_number", "EP1234567A1"),
    )


@pytest.mark.parametrize("namespace", [None, 1, ""])
def test_external_identifier_rejects_invalid_namespace(namespace: object) -> None:
    with pytest.raises((TypeError, ValueError), match="namespace"):
        ExternalIdentifier(namespace, "value")  # type: ignore[arg-type]


@pytest.mark.parametrize("namespace", ["DOI", "doi.", "doi namespace", "doi/"])
def test_external_identifier_rejects_noncanonical_namespace(namespace: str) -> None:
    with pytest.raises(ValueError, match="namespace"):
        ExternalIdentifier(namespace, "value")


@pytest.mark.parametrize("value", [None, 1])
def test_external_identifier_rejects_invalid_value_type(value: object) -> None:
    with pytest.raises(TypeError, match="value"):
        ExternalIdentifier("doi", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", " ", "\t\n"])
def test_external_identifier_rejects_empty_value(value: str) -> None:
    with pytest.raises(ValueError, match="value"):
        ExternalIdentifier("doi", value)


def test_external_identifier_is_immutable_and_serialises() -> None:
    identifier = ExternalIdentifier("doi", "10.1000/example")

    assert identifier.as_tuple() == ("doi", "10.1000/example")

    with pytest.raises(FrozenInstanceError):
        identifier.value = "changed"  # type: ignore[misc]


def test_document_identity_accepts_only_document_id() -> None:
    document_id = DocumentId.generate()
    identity = DocumentIdentity(document_id)

    assert identity.document_id == document_id

    with pytest.raises(FrozenInstanceError):
        identity.document_id = DocumentId.generate()  # type: ignore[misc]


@pytest.mark.parametrize(
    "document_id", [None, "document-id", DocumentVersionId.generate()]
)
def test_document_identity_rejects_wrong_identifier_type(document_id: object) -> None:
    with pytest.raises(TypeError, match="document_id"):
        DocumentIdentity(document_id)  # type: ignore[arg-type]


def test_document_version_accepts_canonical_values() -> None:
    document_version = DocumentVersion(
        document_version_id=DocumentVersionId.generate(),
        document_id=DocumentId.generate(),
        content_hash=_content_hash(),
        external_identifiers=_external_identifiers(),
    )

    assert document_version.external_identifiers == _external_identifiers()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("document_version_id", None, "document_version_id"),
        ("document_id", None, "document_id"),
        ("content_hash", None, "content_hash"),
    ],
)
def test_document_version_rejects_wrong_field_types(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "document_version_id": DocumentVersionId.generate(),
        "document_id": DocumentId.generate(),
        "content_hash": _content_hash(),
    }
    values[field] = value

    with pytest.raises(TypeError, match=message):
        DocumentVersion(**values)  # type: ignore[arg-type]


def test_document_version_requires_tuple_of_external_identifiers() -> None:
    with pytest.raises(TypeError, match="tuple"):
        DocumentVersion(
            DocumentVersionId.generate(),
            DocumentId.generate(),
            _content_hash(),
            [_external_identifiers()[0]],  # type: ignore[arg-type]
        )


def test_document_version_rejects_invalid_identifier_members() -> None:
    with pytest.raises(TypeError, match="ExternalIdentifier"):
        DocumentVersion(
            DocumentVersionId.generate(),
            DocumentId.generate(),
            _content_hash(),
            ("not-an-external-identifier",),  # type: ignore[arg-type]
        )


def test_document_version_rejects_duplicate_external_identifiers() -> None:
    identifier = ExternalIdentifier("doi", "10.1000/example")

    with pytest.raises(ValueError, match="duplicates"):
        DocumentVersion(
            DocumentVersionId.generate(),
            DocumentId.generate(),
            _content_hash(),
            (identifier, identifier),
        )


def test_document_version_requires_canonical_identifier_order() -> None:
    identifiers = _external_identifiers()

    with pytest.raises(ValueError, match="canonical order"):
        DocumentVersion(
            DocumentVersionId.generate(),
            DocumentId.generate(),
            _content_hash(),
            tuple(reversed(identifiers)),
        )


def test_document_version_is_immutable() -> None:
    document_version = DocumentVersion(
        DocumentVersionId.generate(),
        DocumentId.generate(),
        _content_hash(),
    )

    with pytest.raises(FrozenInstanceError):
        document_version.content_hash = _content_hash()  # type: ignore[misc]
