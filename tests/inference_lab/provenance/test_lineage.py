"""Tests for lineage input references and directed edges."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.provenance.errors import InvalidLineageReferenceError
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.lineage import (
    InputRole,
    LineageEdge,
    LineageInputs,
    ParentRecordReference,
    SourceAnchor,
)

_DOCUMENT_VERSION_A = DocumentVersionId.from_str("11111111-1111-4111-8111-111111111111")
_DOCUMENT_VERSION_B = DocumentVersionId.from_str("22222222-2222-4222-8222-222222222222")
_SOURCE_SPAN_A = SourceSpanId.from_str("33333333-3333-4333-8333-333333333333")
_SOURCE_SPAN_B = SourceSpanId.from_str("44444444-4444-4444-8444-444444444444")
_PARENT_A = RecordId.from_str("55555555-5555-4555-8555-555555555555")
_PARENT_B = RecordId.from_str("66666666-6666-4666-8666-666666666666")
_CHILD = RecordId.from_str("77777777-7777-4777-8777-777777777777")


def _source_a(**annotations: object) -> SourceAnchor:
    return SourceAnchor(
        document_version_id=_DOCUMENT_VERSION_A,
        source_span_id=_SOURCE_SPAN_A,
        **annotations,  # type: ignore[arg-type]
    )


def test_source_anchor_preserves_typed_source_identity_and_annotations() -> None:
    anchor = _source_a(
        input_role=InputRole("supporting-passage"),
        input_ordinal=0,
    )

    assert anchor.document_version_id == _DOCUMENT_VERSION_A
    assert anchor.source_span_id == _SOURCE_SPAN_A
    assert anchor.input_role == InputRole.from_str("supporting-passage")
    assert anchor.input_ordinal == 0


def test_parent_reference_preserves_role_and_explicit_order() -> None:
    parent = ParentRecordReference(
        record_id=_PARENT_A,
        input_role=InputRole("document-summary"),
        input_ordinal=1,
    )

    assert parent.record_id == _PARENT_A
    assert parent.input_role is not None
    assert parent.input_role.as_str() == "document-summary"
    assert parent.input_ordinal == 1


def test_lineage_inputs_are_canonical_independent_of_caller_order() -> None:
    source_a = _source_a()
    source_b = SourceAnchor(
        document_version_id=_DOCUMENT_VERSION_B,
        source_span_id=_SOURCE_SPAN_B,
    )
    parent_a = ParentRecordReference(record_id=_PARENT_A)
    parent_b = ParentRecordReference(record_id=_PARENT_B)

    inputs = LineageInputs(
        source_anchors=(source_b, source_a),
        parent_records=(parent_b, parent_a),
    )

    assert inputs.source_anchors == (source_a, source_b)
    assert inputs.parent_records == (parent_a, parent_b)


def test_lineage_inputs_support_multiple_semantically_annotated_inputs() -> None:
    first = _source_a(input_role=InputRole("comparison-input"), input_ordinal=0)
    second = SourceAnchor(
        document_version_id=_DOCUMENT_VERSION_B,
        source_span_id=_SOURCE_SPAN_B,
        input_role=InputRole("comparison-input"),
        input_ordinal=1,
    )

    inputs = LineageInputs(source_anchors=(second, first))

    assert inputs.source_anchors == (first, second)


def test_edges_point_from_each_input_towards_the_derived_child() -> None:
    source = _source_a()
    parent = ParentRecordReference(record_id=_PARENT_A)
    inputs = LineageInputs(source_anchors=(source,), parent_records=(parent,))

    edges = inputs.edges_to(_CHILD)

    assert {edge.input_reference for edge in edges} == {source, parent}
    assert {edge.child_record_id for edge in edges} == {_CHILD}
    assert edges == tuple(sorted(edges, key=lambda edge: edge.key()))


def test_reference_and_edge_values_are_immutable() -> None:
    anchor = _source_a()
    edge = LineageEdge(input_reference=anchor, child_record_id=_CHILD)

    with pytest.raises(FrozenInstanceError):
        anchor.source_span_id = _SOURCE_SPAN_B  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        edge.child_record_id = _PARENT_A  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, True, 1, "supporting passage"])
def test_input_role_rejects_wrong_types_or_noncanonical_values(value: object) -> None:
    expected_error = (
        TypeError if not isinstance(value, str) else InvalidLineageReferenceError
    )

    with pytest.raises(expected_error):
        InputRole(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("ordinal", [True, 1.0, "1", (), []])
def test_input_ordinal_rejects_wrong_runtime_types(ordinal: object) -> None:
    with pytest.raises(TypeError, match="input_ordinal must be an integer"):
        _source_a(input_ordinal=ordinal)


def test_input_ordinal_rejects_negative_values() -> None:
    with pytest.raises(InvalidLineageReferenceError, match="greater than or equal"):
        _source_a(input_ordinal=-1)


def test_source_anchor_enforces_identifier_domains() -> None:
    with pytest.raises(TypeError, match="document_version_id"):
        SourceAnchor(
            document_version_id=_PARENT_A,  # type: ignore[arg-type]
            source_span_id=_SOURCE_SPAN_A,
        )
    with pytest.raises(TypeError, match="source_span_id"):
        SourceAnchor(
            document_version_id=_DOCUMENT_VERSION_A,
            source_span_id=_PARENT_A,  # type: ignore[arg-type]
        )


def test_parent_reference_enforces_record_identifier_domain() -> None:
    with pytest.raises(TypeError, match="record_id"):
        ParentRecordReference(record_id=_DOCUMENT_VERSION_A)  # type: ignore[arg-type]


def test_lineage_inputs_require_immutable_typed_collections() -> None:
    source = _source_a()
    parent = ParentRecordReference(record_id=_PARENT_A)

    with pytest.raises(TypeError, match="source_anchors must be a tuple"):
        LineageInputs(source_anchors=[source])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SourceAnchor"):
        LineageInputs(source_anchors=(parent,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parent_records must be a tuple"):
        LineageInputs(parent_records=[parent])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ParentRecordReference"):
        LineageInputs(parent_records=(source,))  # type: ignore[arg-type]


def test_lineage_inputs_reject_empty_or_duplicate_inputs() -> None:
    source = _source_a()
    parent = ParentRecordReference(record_id=_PARENT_A)

    with pytest.raises(InvalidLineageReferenceError, match="at least one"):
        LineageInputs()
    with pytest.raises(InvalidLineageReferenceError, match="duplicates"):
        LineageInputs(source_anchors=(source, source))
    with pytest.raises(InvalidLineageReferenceError, match="duplicates"):
        LineageInputs(parent_records=(parent, parent))


def test_annotations_do_not_disguise_duplicate_input_identity() -> None:
    first = _source_a(input_role=InputRole("first-input"), input_ordinal=0)
    second = _source_a(input_role=InputRole("second-input"), input_ordinal=1)

    with pytest.raises(InvalidLineageReferenceError, match="duplicates"):
        LineageInputs(source_anchors=(first, second))


def test_lineage_edge_enforces_reference_and_child_domains() -> None:
    source = _source_a()

    with pytest.raises(TypeError, match="input_reference"):
        LineageEdge(input_reference=_PARENT_A, child_record_id=_CHILD)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="child_record_id"):
        LineageEdge(
            input_reference=source,
            child_record_id=_DOCUMENT_VERSION_A,  # type: ignore[arg-type]
        )


def test_lineage_reference_errors_do_not_echo_rejected_values() -> None:
    private_marker = "private input role"

    with pytest.raises(InvalidLineageReferenceError) as captured:
        InputRole(private_marker)

    assert private_marker not in str(captured.value)
