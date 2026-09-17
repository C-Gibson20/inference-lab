"""Tests for storage-owned canonical chunk records."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.provenance import DocumentVersionId, SourceSpanId
from inference_lab.storage import (
    CURRENT_CHUNK_PAYLOAD_SCHEMA_VERSION,
    ChunkerVersionRecord,
    ChunkingPolicyRecord,
    ChunkMinimumExceptionRecord,
    ChunkOversizeUnitPolicy,
    ChunkParserVersionRecord,
    ChunkSourceReferenceRecord,
    ChunkSplitReasonRecord,
    ChunkStructuralBoundaryPolicy,
    ChunkTokeniserIdentityRecord,
    StructuralBlockKind,
)

_DOCUMENT_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_SOURCE_SPAN_ID = SourceSpanId("22222222-2222-4222-8222-222222222222")


def _parser_version(**overrides: object) -> ChunkParserVersionRecord:
    values: dict[str, object] = {
        "implementation": "text-native",
        "contract_version": "3.0.0",
        "normalisation_version": "1.0.0",
    }
    values.update(overrides)
    return ChunkParserVersionRecord(**values)  # type: ignore[arg-type]


def _chunker_version(**overrides: object) -> ChunkerVersionRecord:
    values: dict[str, object] = {
        "implementation": "structure-aware-greedy",
        "contract_version": "1.0.1",
    }
    values.update(overrides)
    return ChunkerVersionRecord(**values)  # type: ignore[arg-type]


def _tokeniser_identity(**overrides: object) -> ChunkTokeniserIdentityRecord:
    values: dict[str, object] = {
        "implementation": "deterministic-tokeniser",
        "contract_version": "1.0.0",
        "artefact_identity": "tokeniser-model",
        "artefact_revision": "2026-01",
        "options": (("normalise", "unicode"), ("provider", "fixture")),
    }
    values.update(overrides)
    return ChunkTokeniserIdentityRecord(**values)  # type: ignore[arg-type]


def _policy(**overrides: object) -> ChunkingPolicyRecord:
    values: dict[str, object] = {
        "min_tokens": 256,
        "max_tokens": 512,
        "overlap_numerator": 15,
        "overlap_denominator": 100,
        "structural_boundary_policy": ChunkStructuralBoundaryPolicy.GREEDY,
        "oversize_unit_policy": ChunkOversizeUnitPolicy.TOKEN_WINDOW,
    }
    values.update(overrides)
    return ChunkingPolicyRecord(**values)  # type: ignore[arg-type]


def _reference(**overrides: object) -> ChunkSourceReferenceRecord:
    values: dict[str, object] = {
        "source_span_id": _SOURCE_SPAN_ID,
        "document_version_id": _DOCUMENT_VERSION_ID,
        "block_ordinal": 2,
        "block_kind": StructuralBlockKind.PARAGRAPH,
        "source_start": 10,
        "source_end": 20,
    }
    values.update(overrides)
    return ChunkSourceReferenceRecord(**values)  # type: ignore[arg-type]


def test_chunk_source_reference_is_immutable_and_canonically_orderable() -> None:
    reference = _reference()

    assert reference.source_range == (10, 20)
    assert reference.key() == (
        2,
        "paragraph",
        _SOURCE_SPAN_ID.as_str(),
        10,
        20,
    )
    with pytest.raises(FrozenInstanceError):
        reference.source_start = 11  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("source_span_id", "not-an-id", TypeError),
        ("document_version_id", "not-an-id", TypeError),
        ("block_ordinal", True, TypeError),
        ("block_ordinal", -1, ValueError),
        ("block_kind", "paragraph", TypeError),
        ("source_start", True, TypeError),
        ("source_end", 1.5, TypeError),
    ],
)
def test_chunk_source_reference_rejects_invalid_fields(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match=field_name):
        _reference(**{field_name: value})


@pytest.mark.parametrize(
    ("source_start", "source_end"),
    [(0, 0), (20, 10), (-1, 10)],
)
def test_chunk_source_reference_rejects_empty_or_reversed_ranges(
    source_start: int,
    source_end: int,
) -> None:
    with pytest.raises(ValueError, match="source range"):
        _reference(source_start=source_start, source_end=source_end)


def test_chunk_parser_version_is_immutable_and_canonical() -> None:
    parser_version = _parser_version()

    assert parser_version.value == ("text-native/contract-3.0.0/normalisation-1.0.0")
    with pytest.raises(FrozenInstanceError):
        parser_version.contract_version = "4.0.0"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("implementation", 1, TypeError),
        ("implementation", "Text Native", ValueError),
        ("contract_version", 1, TypeError),
        ("contract_version", "3", ValueError),
        ("normalisation_version", 1, TypeError),
        ("normalisation_version", "latest", ValueError),
    ],
)
def test_chunk_parser_version_rejects_invalid_fields(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match=field_name):
        _parser_version(**{field_name: value})


def test_chunker_version_is_immutable_and_canonical() -> None:
    chunker_version = _chunker_version()

    assert chunker_version.value == "structure-aware-greedy/contract-1.0.1"
    with pytest.raises(FrozenInstanceError):
        chunker_version.contract_version = "2.0.0"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("implementation", 1, TypeError),
        ("implementation", "Structure Aware", ValueError),
        ("contract_version", 1, TypeError),
        ("contract_version", "1", ValueError),
    ],
)
def test_chunker_version_rejects_invalid_fields(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match=field_name):
        _chunker_version(**{field_name: value})


def test_chunk_tokeniser_identity_is_immutable_and_canonical() -> None:
    identity = _tokeniser_identity()

    assert identity.as_str() == (
        "deterministic-tokeniser/contract-1.0.0/"
        "artefact-tokeniser-model@2026-01/"
        "options-normalise=unicode,provider=fixture"
    )
    with pytest.raises(FrozenInstanceError):
        identity.artefact_revision = "2027-01"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("implementation", 1, TypeError),
        ("implementation", "Deterministic", ValueError),
        ("contract_version", 1, TypeError),
        ("contract_version", "latest", ValueError),
        ("artefact_identity", 1, TypeError),
        ("artefact_identity", "", ValueError),
        ("artefact_identity", "model/path", ValueError),
        ("artefact_revision", 1, TypeError),
        ("artefact_revision", "", ValueError),
        ("artefact_revision", "revision\\path", ValueError),
        ("options", [("provider", "fixture")], TypeError),
    ],
)
def test_chunk_tokeniser_identity_rejects_invalid_fields(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match=field_name):
        _tokeniser_identity(**{field_name: value})


@pytest.mark.parametrize(
    "options",
    [
        (("provider",),),
        (("provider", "fixture", "extra"),),
        ((1, "fixture"),),
        (("provider", 1),),
        (("Provider", "fixture"),),
        (("", "fixture"),),
        (("with=separator", "fixture"),),
        (("provider", ""),),
        (("provider", "fixture/path"),),
        (("provider", "fixture\\path"),),
        (("provider", "fixture\x00value"),),
        (("provider", "fixture"), ("algorithm", "fast")),
        (("provider", "fixture"), ("provider", "other")),
    ],
)
def test_chunk_tokeniser_identity_rejects_malformed_options(
    options: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="options|option"):
        _tokeniser_identity(options=options)


def test_chunking_policy_is_immutable_and_canonically_orderable() -> None:
    policy = _policy()

    assert policy.key() == (
        256,
        512,
        15,
        100,
        "structure-aware-greedy",
        "token-window",
    )
    with pytest.raises(FrozenInstanceError):
        policy.max_tokens = 1024  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("min_tokens", True, TypeError),
        ("min_tokens", 0, ValueError),
        ("min_tokens", -1, ValueError),
        ("max_tokens", 1.5, TypeError),
        ("max_tokens", 0, ValueError),
        ("overlap_numerator", True, TypeError),
        ("overlap_numerator", -1, ValueError),
        ("overlap_denominator", 1.5, TypeError),
        ("overlap_denominator", 0, ValueError),
        (
            "structural_boundary_policy",
            "structure-aware-greedy",
            TypeError,
        ),
        ("oversize_unit_policy", "token-window", TypeError),
    ],
)
def test_chunking_policy_rejects_invalid_fields(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match=field_name):
        _policy(**{field_name: value})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"min_tokens": 513, "max_tokens": 512}, "min_tokens"),
        (
            {"overlap_numerator": 101, "overlap_denominator": 100},
            "overlap_numerator",
        ),
    ],
)
def test_chunking_policy_rejects_inconsistent_limits(
    changes: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _policy(**changes)


def test_chunk_annotation_vocabularies_are_stable() -> None:
    assert CURRENT_CHUNK_PAYLOAD_SCHEMA_VERSION == "1.0.0"
    assert ChunkMinimumExceptionRecord.SHORT_DOCUMENT.value == "short_document"
    assert (
        ChunkMinimumExceptionRecord.PROTECTED_STRUCTURE_NEIGHBOUR.value
        == "protected_structure_neighbour"
    )
    assert (
        ChunkMinimumExceptionRecord.FINAL_REMAINDER.value
        == "final_remainder_minimum_exception"
    )
    assert ChunkSplitReasonRecord.OVERSIZE_STRUCTURE.value == "oversize_structure"


@pytest.mark.parametrize("value", ["", "unknown", "short-document"])
def test_chunk_annotation_vocabularies_reject_unknown_values(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        ChunkMinimumExceptionRecord(value)


@pytest.mark.parametrize("value", ["", "unknown", "oversize-structure"])
def test_chunk_split_reason_vocabulary_rejects_unknown_values(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        ChunkSplitReasonRecord(value)
