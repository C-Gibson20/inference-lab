"""Tests for typed, immutable provenance identifiers."""

import itertools
import uuid

import pytest

from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceObservationId,
    SourceSpanId,
)
from inference_lab.provenance import identifiers as identifiers_module

KNOWN_UUID = uuid.UUID("12345678-1234-4abc-8def-123456789abc")

type Identifier = (
    CorpusId
    | CorpusVersionId
    | DocumentId
    | DocumentVersionId
    | RecordId
    | SourceObservationId
    | SourceSpanId
    | DocumentFamilyId
)
type IdentifierType = type[Identifier]

IDENTIFIER_TYPES: tuple[IdentifierType, ...] = (
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceObservationId,
    CorpusId,
    CorpusVersionId,
    SourceSpanId,
    DocumentFamilyId,
)


@pytest.mark.parametrize("identifier_type", IDENTIFIER_TYPES)
def test_generate_returns_canonical_uuid_for_requested_domain(
    identifier_type: IdentifierType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identifiers_module.uuid, "uuid4", lambda: KNOWN_UUID)

    identifier = identifier_type.generate()

    assert isinstance(identifier, identifier_type)
    assert identifier.as_str() == str(KNOWN_UUID)
    assert str(identifier) == str(KNOWN_UUID)


@pytest.mark.parametrize("identifier_type", IDENTIFIER_TYPES)
def test_from_str_round_trips_canonical_value(identifier_type: IdentifierType) -> None:
    original = identifier_type.generate()
    restored = identifier_type.from_str(str(original))

    assert restored == original
    assert hash(restored) == hash(original)
    assert repr(restored) == f"{identifier_type.__name__}({str(original)!r})"


@pytest.mark.parametrize("identifier_type", IDENTIFIER_TYPES)
@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-uuid",
        "{12345678-1234-5678-1234-567812345678}",
        "12345678-1234-5678-1234-567812345678 ",
        "12345678123456781234567812345678",
        "12345678-1234-5678-1234-56781234567G",
        "ABCDEFAB-1234-4ABC-8DEF-ABCDEFABCDEF",
    ],
)
def test_rejects_noncanonical_values(
    identifier_type: IdentifierType,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        identifier_type.from_str(value)


@pytest.mark.parametrize("identifier_type", IDENTIFIER_TYPES)
def test_rejects_non_string_values(identifier_type: IdentifierType) -> None:
    with pytest.raises(TypeError):
        identifier_type.from_str(None)  # type: ignore[arg-type]


def test_identifiers_are_immutable() -> None:
    identifier = DocumentId.generate()

    assert not hasattr(identifier, "__dict__")
    with pytest.raises(AttributeError, match="immutable"):
        identifier._value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("left_type", "right_type"), tuple(itertools.combinations(IDENTIFIER_TYPES, 2))
)
def test_identifier_equality_includes_concrete_domain(
    left_type: IdentifierType,
    right_type: IdentifierType,
) -> None:
    value = str(KNOWN_UUID)

    assert left_type.from_str(value) != right_type.from_str(value)


def test_identifier_equality_and_hash_use_domain_and_value() -> None:
    value = str(KNOWN_UUID)
    first = DocumentId.from_str(value)
    second = DocumentId.from_str(value)

    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_document_family_id_can_be_created_and_compared() -> None:
    family_id = DocumentFamilyId.generate()

    assert isinstance(family_id, DocumentFamilyId)
    assert family_id != DocumentId.from_str(family_id.as_str())
