"""Tests for immutable derived-record metadata and version semantics."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.provenance.derived import (
    DerivationFingerprint,
    DerivedRecordKind,
    DerivedRecordMetadata,
    OutputDigest,
)
from inference_lab.provenance.errors import (
    InvalidDerivedRecordError,
    UnsupportedLineageSchemaVersionError,
)
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.lineage import (
    LineageInputs,
    ParentRecordReference,
    SourceAnchor,
)
from inference_lab.provenance.operations import (
    ConfigurationIdentity,
    DigestAlgorithm,
    OperationKind,
    ProducingOperation,
)
from inference_lab.provenance.versioning import CURRENT_LINEAGE_SCHEMA_VERSION

_RECORD_A = RecordId.from_str("11111111-1111-4111-8111-111111111111")
_RECORD_B = RecordId.from_str("22222222-2222-4222-8222-222222222222")
_DOCUMENT_VERSION = DocumentVersionId.from_str("33333333-3333-4333-8333-333333333333")
_SOURCE_SPAN = SourceSpanId.from_str("44444444-4444-4444-8444-444444444444")
_CONFIGURATION_DIGEST = "a" * 64
_OUTPUT_DIGEST = "b" * 64
_FINGERPRINT = "c" * 64


def _operation() -> ProducingOperation:
    return ProducingOperation(
        kind=OperationKind("chunking"),
        implementation="structure-aware-chunker",
        contract_version="1.0.0",
        configuration_identity=ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest=_CONFIGURATION_DIGEST,
        ),
    )


def _source_inputs() -> LineageInputs:
    return LineageInputs(
        source_anchors=(
            SourceAnchor(
                document_version_id=_DOCUMENT_VERSION,
                source_span_id=_SOURCE_SPAN,
            ),
        )
    )


def _metadata(
    *,
    record_id: RecordId = _RECORD_A,
    record_kind: DerivedRecordKind | None = None,
    inputs: LineageInputs | None = None,
    output_digest: OutputDigest | None = None,
    fingerprint: DerivationFingerprint | None = None,
    supersedes_record_id: RecordId | None = None,
) -> DerivedRecordMetadata:
    return DerivedRecordMetadata(
        record_id=record_id,
        record_kind=record_kind or DerivedRecordKind("chunk"),
        lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
        inputs=inputs or _source_inputs(),
        operation=_operation(),
        output_digest=output_digest
        or OutputDigest(DigestAlgorithm.SHA256, _OUTPUT_DIGEST),
        derivation_fingerprint=fingerprint
        or DerivationFingerprint(DigestAlgorithm.SHA256, _FINGERPRINT),
        supersedes_record_id=supersedes_record_id,
    )


def test_derived_record_metadata_composes_complete_direct_lineage() -> None:
    metadata = _metadata()

    assert metadata.record_id == _RECORD_A
    assert metadata.record_kind.as_str() == "chunk"
    assert metadata.lineage_schema_version == CURRENT_LINEAGE_SCHEMA_VERSION
    assert metadata.inputs == _source_inputs()
    assert metadata.operation == _operation()
    assert metadata.output_digest.as_str() == f"sha256:{_OUTPUT_DIGEST}"
    assert metadata.derivation_fingerprint.as_str() == f"sha256:{_FINGERPRINT}"
    assert metadata.supersedes_record_id is None


def test_parent_only_derived_record_retains_derived_lineage() -> None:
    inputs = LineageInputs(parent_records=(ParentRecordReference(record_id=_RECORD_A),))

    metadata = _metadata(
        record_id=_RECORD_B,
        record_kind=DerivedRecordKind("topic-summary"),
        inputs=inputs,
    )

    assert metadata.inputs.parent_records[0].record_id == _RECORD_A
    assert not metadata.inputs.source_anchors


def test_replacement_has_new_identity_and_explicit_supersession() -> None:
    original = _metadata()
    replacement = _metadata(
        record_id=_RECORD_B,
        output_digest=OutputDigest(DigestAlgorithm.SHA256, "d" * 64),
        fingerprint=DerivationFingerprint(DigestAlgorithm.SHA256, "e" * 64),
        supersedes_record_id=original.record_id,
    )

    assert replacement.record_id != original.record_id
    assert replacement.output_digest != original.output_digest
    assert replacement.derivation_fingerprint != original.derivation_fingerprint
    assert replacement.supersedes_record_id == original.record_id


def test_fingerprint_does_not_replace_record_identity() -> None:
    first = _metadata(record_id=_RECORD_A)
    second = _metadata(record_id=_RECORD_B)

    assert first.record_id != second.record_id
    assert first.derivation_fingerprint == second.derivation_fingerprint


def test_derived_record_metadata_is_immutable() -> None:
    metadata = _metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.record_id = _RECORD_B  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        metadata.supersedes_record_id = _RECORD_B  # type: ignore[misc]


@pytest.mark.parametrize("value", ["", "Chunk", "topic_summary", "chunk text"])
def test_record_kind_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(InvalidDerivedRecordError, match="lowercase slug"):
        DerivedRecordKind(value)


@pytest.mark.parametrize("value", [None, True, 1, 1.0])
def test_record_kind_rejects_wrong_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="record_kind must be a string"):
        DerivedRecordKind(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "digest_type",
    [OutputDigest, DerivationFingerprint],
)
@pytest.mark.parametrize("digest", ["", "A" * 64, "a" * 63, "g" * 64])
def test_record_digests_require_canonical_sha256(
    digest_type: type[OutputDigest] | type[DerivationFingerprint],
    digest: str,
) -> None:
    with pytest.raises(InvalidDerivedRecordError, match="digest"):
        digest_type(DigestAlgorithm.SHA256, digest)


def test_record_digests_reject_wrong_runtime_types() -> None:
    with pytest.raises(TypeError, match="output algorithm"):
        OutputDigest("sha256", _OUTPUT_DIGEST)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fingerprint digest"):
        DerivationFingerprint(
            DigestAlgorithm.SHA256,
            {"payload": "not-a-digest"},  # type: ignore[arg-type]
        )


def test_metadata_rejects_unsupported_lineage_schema_version() -> None:
    with pytest.raises(UnsupportedLineageSchemaVersionError):
        DerivedRecordMetadata(
            record_id=_RECORD_A,
            record_kind=DerivedRecordKind("chunk"),
            lineage_schema_version="2.0.0",
            inputs=_source_inputs(),
            operation=_operation(),
            output_digest=OutputDigest(DigestAlgorithm.SHA256, _OUTPUT_DIGEST),
            derivation_fingerprint=DerivationFingerprint(
                DigestAlgorithm.SHA256,
                _FINGERPRINT,
            ),
        )


def test_metadata_enforces_typed_contract_fields() -> None:
    valid = {
        "record_id": _RECORD_A,
        "record_kind": DerivedRecordKind("chunk"),
        "lineage_schema_version": CURRENT_LINEAGE_SCHEMA_VERSION,
        "inputs": _source_inputs(),
        "operation": _operation(),
        "output_digest": OutputDigest(DigestAlgorithm.SHA256, _OUTPUT_DIGEST),
        "derivation_fingerprint": DerivationFingerprint(
            DigestAlgorithm.SHA256,
            _FINGERPRINT,
        ),
    }

    invalid_fields = {
        "record_id": _DOCUMENT_VERSION,
        "record_kind": "chunk",
        "inputs": (),
        "operation": "chunking",
        "output_digest": _OUTPUT_DIGEST,
        "derivation_fingerprint": _FINGERPRINT,
    }
    for field_name, invalid_value in invalid_fields.items():
        with pytest.raises(TypeError, match=field_name):
            DerivedRecordMetadata(
                **(valid | {field_name: invalid_value})  # type: ignore[arg-type]
            )


def test_metadata_rejects_invalid_supersession_values() -> None:
    with pytest.raises(TypeError, match="supersedes_record_id"):
        _metadata(supersedes_record_id=_DOCUMENT_VERSION)  # type: ignore[arg-type]
    with pytest.raises(InvalidDerivedRecordError, match="supersede itself"):
        _metadata(supersedes_record_id=_RECORD_A)


def test_metadata_has_no_unvalidated_payload_or_observation_fields() -> None:
    with pytest.raises(TypeError):
        DerivedRecordMetadata(
            record_id=_RECORD_A,
            record_kind=DerivedRecordKind("chunk"),
            lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
            inputs=_source_inputs(),
            operation=_operation(),
            output_digest=OutputDigest(DigestAlgorithm.SHA256, _OUTPUT_DIGEST),
            derivation_fingerprint=DerivationFingerprint(
                DigestAlgorithm.SHA256,
                _FINGERPRINT,
            ),
            payload="private chunk text",  # type: ignore[call-arg]
        )


def test_invalid_derived_record_errors_do_not_echo_rejected_values() -> None:
    private_marker = "private record kind"

    with pytest.raises(InvalidDerivedRecordError) as captured:
        DerivedRecordKind(private_marker)

    assert private_marker not in str(captured.value)
