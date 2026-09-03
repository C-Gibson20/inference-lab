"""End-to-end acceptance coverage for derived-record lineage contracts."""

import pytest

from inference_lab.provenance import (
    CURRENT_LINEAGE_SCHEMA_VERSION,
    ConfigurationIdentity,
    DerivedRecordKind,
    DerivedRecordMetadata,
    DigestAlgorithm,
    DocumentVersionId,
    InvalidLineageReferenceError,
    KnownSourceSpan,
    LineageBundle,
    LineageInputs,
    OperationKind,
    OutputDigest,
    ParentRecordReference,
    ProducingOperation,
    RecordId,
    SourceAnchor,
    SourceSpanId,
    build_derivation_fingerprint,
    deserialise_lineage_bundle,
    serialise_lineage_bundle,
    validate_lineage_bundle,
)

_DOCUMENT_A = DocumentVersionId.from_str("11111111-1111-4111-8111-111111111111")
_DOCUMENT_B = DocumentVersionId.from_str("22222222-2222-4222-8222-222222222222")
_SPAN_A = SourceSpanId.from_str("33333333-3333-4333-8333-333333333333")
_SPAN_B = SourceSpanId.from_str("44444444-4444-4444-8444-444444444444")
_CHUNK = RecordId.from_str("55555555-5555-4555-8555-555555555555")
_FACT = RecordId.from_str("66666666-6666-4666-8666-666666666666")
_DOCUMENT_SUMMARY_A = RecordId.from_str("77777777-7777-4777-8777-777777777777")
_DOCUMENT_SUMMARY_B = RecordId.from_str("88888888-8888-4888-8888-888888888888")
_TOPIC_SUMMARY = RecordId.from_str("99999999-9999-4999-8999-999999999999")
_CORPUS_SUMMARY = RecordId.from_str("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _source_inputs(
    document_version_id: DocumentVersionId,
    source_span_id: SourceSpanId,
) -> LineageInputs:
    return LineageInputs(
        source_anchors=(
            SourceAnchor(
                document_version_id=document_version_id,
                source_span_id=source_span_id,
            ),
        )
    )


def _parent_inputs(*record_ids: RecordId) -> LineageInputs:
    return LineageInputs(
        parent_records=tuple(
            ParentRecordReference(record_id=record_id) for record_id in record_ids
        )
    )


def _record(
    record_id: RecordId,
    *,
    kind: str,
    inputs: LineageInputs,
    digest_character: str,
) -> DerivedRecordMetadata:
    record_kind = DerivedRecordKind(kind)
    operation = ProducingOperation(
        kind=OperationKind(f"{kind}-production"),
        implementation=f"canonical-{kind}-producer",
        contract_version="1.0.0",
        configuration_identity=ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest=digest_character * 64,
        ),
    )
    output_digest = OutputDigest(
        algorithm=DigestAlgorithm.SHA256,
        digest=digest_character * 64,
    )
    return DerivedRecordMetadata(
        record_id=record_id,
        record_kind=record_kind,
        lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
        inputs=inputs,
        operation=operation,
        output_digest=output_digest,
        derivation_fingerprint=build_derivation_fingerprint(
            record_kind=record_kind,
            lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
            inputs=inputs,
            operation=operation,
            output_digest=output_digest,
        ),
    )


def test_original_source_directly_supports_chunk_and_fact_records() -> None:
    inputs = _source_inputs(_DOCUMENT_A, _SPAN_A)
    chunk = _record(_CHUNK, kind="chunk", inputs=inputs, digest_character="1")
    fact = _record(_FACT, kind="fact", inputs=inputs, digest_character="2")
    bundle = LineageBundle.from_records(
        source_spans=(KnownSourceSpan(_DOCUMENT_A, _SPAN_A),),
        records=(chunk, fact),
    )

    assert validate_lineage_bundle(bundle) is bundle
    assert {record.record_kind.as_str() for record in bundle.records} == {
        "chunk",
        "fact",
    }
    assert all(
        edge.input_reference == inputs.source_anchors[0] for edge in bundle.edges
    )
    assert {edge.child_record_id for edge in bundle.edges} == {_CHUNK, _FACT}


def test_summary_chain_round_trips_to_original_source_evidence() -> None:
    document_a = _record(
        _DOCUMENT_SUMMARY_A,
        kind="document-summary",
        inputs=_source_inputs(_DOCUMENT_A, _SPAN_A),
        digest_character="3",
    )
    document_b = _record(
        _DOCUMENT_SUMMARY_B,
        kind="document-summary",
        inputs=_source_inputs(_DOCUMENT_B, _SPAN_B),
        digest_character="4",
    )
    topic = _record(
        _TOPIC_SUMMARY,
        kind="topic-summary",
        inputs=_parent_inputs(_DOCUMENT_SUMMARY_A, _DOCUMENT_SUMMARY_B),
        digest_character="5",
    )
    corpus = _record(
        _CORPUS_SUMMARY,
        kind="corpus-summary",
        inputs=_parent_inputs(_TOPIC_SUMMARY),
        digest_character="6",
    )
    bundle = LineageBundle.from_records(
        source_spans=(
            KnownSourceSpan(_DOCUMENT_A, _SPAN_A),
            KnownSourceSpan(_DOCUMENT_B, _SPAN_B),
        ),
        records=(document_a, document_b, topic, corpus),
    )

    encoded = serialise_lineage_bundle(bundle)

    assert serialise_lineage_bundle(bundle) == encoded
    assert deserialise_lineage_bundle(encoded) == bundle
    assert tuple(record.record_kind.as_str() for record in bundle.records) == (
        "document-summary",
        "document-summary",
        "topic-summary",
        "corpus-summary",
    )


def test_source_less_root_cannot_form_a_lineage_chain() -> None:
    with pytest.raises(InvalidLineageReferenceError, match="at least one"):
        LineageInputs()
