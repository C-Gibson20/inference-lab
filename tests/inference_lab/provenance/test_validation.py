"""Tests for complete derived-record lineage graph validation."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.provenance.derived import (
    DerivedRecordKind,
    DerivedRecordMetadata,
    OutputDigest,
    build_derivation_fingerprint,
)
from inference_lab.provenance.errors import InvalidLineageGraphError
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.lineage import (
    LineageEdge,
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
from inference_lab.provenance.validation import (
    KnownSourceSpan,
    LineageBundle,
    validate_lineage_bundle,
)
from inference_lab.provenance.versioning import CURRENT_LINEAGE_SCHEMA_VERSION

_DOCUMENT_A = DocumentVersionId.from_str("11111111-1111-4111-8111-111111111111")
_DOCUMENT_B = DocumentVersionId.from_str("22222222-2222-4222-8222-222222222222")
_DOCUMENT_C = DocumentVersionId.from_str("33333333-3333-4333-8333-333333333333")
_SPAN_A = SourceSpanId.from_str("44444444-4444-4444-8444-444444444444")
_SPAN_B = SourceSpanId.from_str("55555555-5555-4555-8555-555555555555")
_RECORD_A = RecordId.from_str("66666666-6666-4666-8666-666666666666")
_RECORD_B = RecordId.from_str("77777777-7777-4777-8777-777777777777")
_RECORD_C = RecordId.from_str("88888888-8888-4888-8888-888888888888")
_RECORD_D = RecordId.from_str("99999999-9999-4999-8999-999999999999")
_RECORD_E = RecordId.from_str("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_UNKNOWN_RECORD = RecordId.from_str("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


def _known_sources() -> tuple[KnownSourceSpan, ...]:
    return (
        KnownSourceSpan(document_version_id=_DOCUMENT_A, source_span_id=_SPAN_A),
        KnownSourceSpan(document_version_id=_DOCUMENT_B, source_span_id=_SPAN_B),
    )


def _source_inputs(
    document_version_id: DocumentVersionId = _DOCUMENT_A,
    source_span_id: SourceSpanId = _SPAN_A,
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


def _operation() -> ProducingOperation:
    return ProducingOperation(
        kind=OperationKind("summarisation"),
        implementation="traceable-summariser",
        contract_version="1.0.0",
        configuration_identity=ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest="c" * 64,
        ),
    )


def _record(
    record_id: RecordId,
    *,
    inputs: LineageInputs,
    kind: str = "summary",
    output_digest: str = "d" * 64,
    supersedes_record_id: RecordId | None = None,
) -> DerivedRecordMetadata:
    record_kind = DerivedRecordKind(kind)
    operation = _operation()
    digest = OutputDigest(DigestAlgorithm.SHA256, output_digest)
    fingerprint = build_derivation_fingerprint(
        record_kind=record_kind,
        lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
        inputs=inputs,
        operation=operation,
        output_digest=digest,
    )
    return DerivedRecordMetadata(
        record_id=record_id,
        record_kind=record_kind,
        lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
        inputs=inputs,
        operation=operation,
        output_digest=digest,
        derivation_fingerprint=fingerprint,
        supersedes_record_id=supersedes_record_id,
    )


def _valid_chain() -> tuple[DerivedRecordMetadata, ...]:
    document_a = _record(_RECORD_A, inputs=_source_inputs())
    document_b = _record(
        _RECORD_B,
        inputs=_source_inputs(_DOCUMENT_B, _SPAN_B),
    )
    topic = _record(_RECORD_C, inputs=_parent_inputs(_RECORD_A, _RECORD_B))
    corpus = _record(_RECORD_D, inputs=_parent_inputs(_RECORD_C))
    return document_a, document_b, topic, corpus


def test_complete_multi_parent_summary_chain_is_valid() -> None:
    bundle = LineageBundle.from_records(
        source_spans=tuple(reversed(_known_sources())),
        records=tuple(reversed(_valid_chain())),
    )

    assert validate_lineage_bundle(bundle) is bundle
    assert bundle.source_spans == _known_sources()
    assert tuple(record.record_id for record in bundle.records) == (
        _RECORD_A,
        _RECORD_B,
        _RECORD_C,
        _RECORD_D,
    )
    assert bundle.edges == tuple(sorted(bundle.edges, key=lambda edge: edge.key()))


def test_empty_lineage_bundle_is_valid() -> None:
    bundle = LineageBundle()

    assert validate_lineage_bundle(bundle) is bundle


def test_valid_supersession_chain_retains_record_kind() -> None:
    original = _record(_RECORD_A, inputs=_source_inputs(), output_digest="1" * 64)
    replacement = _record(
        _RECORD_B,
        inputs=_source_inputs(),
        output_digest="2" * 64,
        supersedes_record_id=_RECORD_A,
    )
    latest = _record(
        _RECORD_C,
        inputs=_source_inputs(),
        output_digest="3" * 64,
        supersedes_record_id=_RECORD_B,
    )
    bundle = LineageBundle.from_records(
        source_spans=(_known_sources()[0],),
        records=(latest, original, replacement),
    )

    assert validate_lineage_bundle(bundle) is bundle


def test_lineage_bundle_values_are_immutable() -> None:
    bundle = LineageBundle.from_records(
        source_spans=_known_sources(),
        records=_valid_chain(),
    )

    with pytest.raises(FrozenInstanceError):
        bundle.edges = ()  # type: ignore[misc]

    source = _known_sources()[0]
    with pytest.raises(FrozenInstanceError):
        source.source_span_id = _SPAN_B  # type: ignore[misc]


def test_missing_source_span_is_rejected() -> None:
    record = _record(_RECORD_A, inputs=_source_inputs())
    bundle = LineageBundle.from_records(source_spans=(), records=(record,))

    with pytest.raises(InvalidLineageGraphError, match="missing source span"):
        validate_lineage_bundle(bundle)


def test_source_span_document_version_mismatch_is_rejected() -> None:
    record = _record(_RECORD_A, inputs=_source_inputs())
    bundle = LineageBundle.from_records(
        source_spans=(
            KnownSourceSpan(
                document_version_id=_DOCUMENT_C,
                source_span_id=_SPAN_A,
            ),
        ),
        records=(record,),
    )

    with pytest.raises(InvalidLineageGraphError, match="wrong document version"):
        validate_lineage_bundle(bundle)


def test_missing_parent_record_is_rejected() -> None:
    record = _record(_RECORD_A, inputs=_parent_inputs(_UNKNOWN_RECORD))
    bundle = LineageBundle.from_records(source_spans=(), records=(record,))

    with pytest.raises(InvalidLineageGraphError, match="missing parent record"):
        validate_lineage_bundle(bundle)


def test_self_parent_relationship_is_rejected() -> None:
    record = _record(_RECORD_A, inputs=_parent_inputs(_RECORD_A))
    bundle = LineageBundle.from_records(source_spans=(), records=(record,))

    with pytest.raises(InvalidLineageGraphError, match="own parent"):
        validate_lineage_bundle(bundle)


def test_duplicate_records_source_entries_and_edges_are_rejected() -> None:
    record = _record(_RECORD_A, inputs=_source_inputs())
    base = LineageBundle.from_records(
        source_spans=(_known_sources()[0],),
        records=(record,),
    )

    duplicate_records = LineageBundle(
        source_spans=base.source_spans,
        records=(record, record),
        edges=base.edges,
    )
    with pytest.raises(InvalidLineageGraphError, match="duplicate records"):
        validate_lineage_bundle(duplicate_records)

    duplicate_sources = LineageBundle(
        source_spans=(base.source_spans[0], base.source_spans[0]),
        records=base.records,
        edges=base.edges,
    )
    with pytest.raises(InvalidLineageGraphError, match="duplicate source-span"):
        validate_lineage_bundle(duplicate_sources)

    duplicate_edges = LineageBundle(
        source_spans=base.source_spans,
        records=base.records,
        edges=(*base.edges, base.edges[0]),
    )
    with pytest.raises(InvalidLineageGraphError, match="duplicate edges"):
        validate_lineage_bundle(duplicate_edges)


def test_edges_must_exactly_match_record_inputs() -> None:
    base = LineageBundle.from_records(
        source_spans=_known_sources(),
        records=_valid_chain(),
    )
    missing_edge = LineageBundle(
        source_spans=base.source_spans,
        records=base.records,
        edges=base.edges[:-1],
    )

    with pytest.raises(InvalidLineageGraphError, match="exactly match"):
        validate_lineage_bundle(missing_edge)


def test_edge_with_missing_child_is_rejected() -> None:
    record = _record(_RECORD_A, inputs=_source_inputs())
    edge = LineageEdge(
        input_reference=record.inputs.source_anchors[0],
        child_record_id=_UNKNOWN_RECORD,
    )
    bundle = LineageBundle(
        source_spans=(_known_sources()[0],),
        records=(record,),
        edges=(*record.inputs.edges_to(record.record_id), edge),
    )

    with pytest.raises(InvalidLineageGraphError, match="missing child"):
        validate_lineage_bundle(bundle)


def test_direct_parent_cycle_is_rejected() -> None:
    first = _record(_RECORD_A, inputs=_parent_inputs(_RECORD_B))
    second = _record(_RECORD_B, inputs=_parent_inputs(_RECORD_A))
    bundle = LineageBundle.from_records(source_spans=(), records=(first, second))

    with pytest.raises(InvalidLineageGraphError, match="lineage contains a cycle"):
        validate_lineage_bundle(bundle)


def test_indirect_parent_cycle_is_rejected() -> None:
    first = _record(_RECORD_A, inputs=_parent_inputs(_RECORD_C))
    second = _record(_RECORD_B, inputs=_parent_inputs(_RECORD_A))
    third = _record(_RECORD_C, inputs=_parent_inputs(_RECORD_B))
    bundle = LineageBundle.from_records(
        source_spans=(),
        records=(first, second, third),
    )

    with pytest.raises(InvalidLineageGraphError, match="lineage contains a cycle"):
        validate_lineage_bundle(bundle)


def test_missing_supersession_predecessor_is_rejected() -> None:
    record = _record(
        _RECORD_A,
        inputs=_source_inputs(),
        supersedes_record_id=_UNKNOWN_RECORD,
    )
    bundle = LineageBundle.from_records(
        source_spans=(_known_sources()[0],),
        records=(record,),
    )

    with pytest.raises(InvalidLineageGraphError, match="missing predecessor"):
        validate_lineage_bundle(bundle)


def test_supersession_kind_change_is_rejected() -> None:
    original = _record(_RECORD_A, inputs=_source_inputs(), kind="chunk")
    replacement = _record(
        _RECORD_B,
        inputs=_source_inputs(),
        kind="summary",
        supersedes_record_id=_RECORD_A,
    )
    bundle = LineageBundle.from_records(
        source_spans=(_known_sources()[0],),
        records=(original, replacement),
    )

    with pytest.raises(
        InvalidLineageGraphError, match="retain the derived record kind"
    ):
        validate_lineage_bundle(bundle)


def test_supersession_branch_is_rejected() -> None:
    original = _record(_RECORD_A, inputs=_source_inputs())
    first = _record(
        _RECORD_B,
        inputs=_source_inputs(),
        output_digest="1" * 64,
        supersedes_record_id=_RECORD_A,
    )
    second = _record(
        _RECORD_C,
        inputs=_source_inputs(),
        output_digest="2" * 64,
        supersedes_record_id=_RECORD_A,
    )
    bundle = LineageBundle.from_records(
        source_spans=(_known_sources()[0],),
        records=(original, first, second),
    )

    with pytest.raises(InvalidLineageGraphError, match="multiple replacements"):
        validate_lineage_bundle(bundle)


def test_supersession_cycle_is_rejected() -> None:
    first = _record(
        _RECORD_A,
        inputs=_source_inputs(),
        supersedes_record_id=_RECORD_B,
    )
    second = _record(
        _RECORD_B,
        inputs=_source_inputs(),
        supersedes_record_id=_RECORD_A,
    )
    bundle = LineageBundle.from_records(
        source_spans=(_known_sources()[0],),
        records=(first, second),
    )

    with pytest.raises(InvalidLineageGraphError, match="supersession lineage"):
        validate_lineage_bundle(bundle)


def test_bundle_collections_and_members_are_strictly_typed() -> None:
    record = _record(_RECORD_A, inputs=_source_inputs())
    source = _known_sources()[0]
    edge = record.inputs.edges_to(record.record_id)[0]

    with pytest.raises(TypeError, match="source_spans must be a tuple"):
        LineageBundle(source_spans=[source])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KnownSourceSpan"):
        LineageBundle(source_spans=(record,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="records must be a tuple"):
        LineageBundle(records=[record])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DerivedRecordMetadata"):
        LineageBundle(records=(source,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="edges must be a tuple"):
        LineageBundle(edges=[edge])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LineageEdge"):
        LineageBundle(edges=(record,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bundle must be a LineageBundle"):
        validate_lineage_bundle((source, record, edge))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="document_version_id"):
        KnownSourceSpan(
            document_version_id=_RECORD_A,  # type: ignore[arg-type]
            source_span_id=_SPAN_A,
        )
    with pytest.raises(TypeError, match="source_span_id"):
        KnownSourceSpan(
            document_version_id=_DOCUMENT_A,
            source_span_id=_RECORD_A,  # type: ignore[arg-type]
        )


def test_graph_errors_do_not_echo_provenance_identifiers() -> None:
    record = _record(_RECORD_A, inputs=_parent_inputs(_UNKNOWN_RECORD))
    bundle = LineageBundle.from_records(source_spans=(), records=(record,))

    with pytest.raises(InvalidLineageGraphError) as captured:
        validate_lineage_bundle(bundle)

    assert _UNKNOWN_RECORD.as_str() not in str(captured.value)
