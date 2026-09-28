"""PROV-01 contract journey across every public identifier domain."""

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
IDENTIFIER_TYPES = (
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceObservationId,
    CorpusId,
    CorpusVersionId,
    SourceSpanId,
    DocumentFamilyId,
)


def test_domains_generate_round_trip_and_keep_identity_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identifiers_module.uuid, "uuid4", lambda: KNOWN_UUID)
    values = []
    for identifier_type in IDENTIFIER_TYPES:
        identifier = identifier_type.generate()
        restored = identifier_type.from_str(str(identifier))
        assert identifier.as_str() == str(KNOWN_UUID)
        assert restored == identifier
        assert hash(restored) == hash(identifier)
        assert repr(identifier) == f"{identifier_type.__name__}({str(KNOWN_UUID)!r})"
        values.append(identifier)
    assert len(set(values)) == len(IDENTIFIER_TYPES)
    assert all(
        left != right
        for left, right in itertools.combinations(
            (kind.from_str(str(KNOWN_UUID)) for kind in IDENTIFIER_TYPES), 2
        )
    )


def test_every_domain_rejects_noncanonical_values_and_wrong_types() -> None:
    invalid_values = (
        "",
        "not-a-uuid",
        "{12345678-1234-5678-1234-567812345678}",
        "12345678-1234-5678-1234-567812345678 ",
        "12345678123456781234567812345678",
        "12345678-1234-5678-1234-56781234567G",
        "ABCDEFAB-1234-4ABC-8DEF-ABCDEFABCDEF",
    )
    for identifier_type in IDENTIFIER_TYPES:
        for invalid in invalid_values:
            with pytest.raises(ValueError):
                identifier_type.from_str(invalid)
        with pytest.raises(TypeError):
            identifier_type.from_str(None)  # type: ignore[arg-type]


def test_identifiers_are_immutable_and_value_equal_within_domain() -> None:
    first = DocumentId.from_str(str(KNOWN_UUID))
    second = DocumentId.from_str(str(KNOWN_UUID))
    assert first == second
    assert len({first, second}) == 1
    assert not hasattr(first, "__dict__")
    with pytest.raises(AttributeError, match="immutable"):
        first._value = "changed"  # type: ignore[misc]
