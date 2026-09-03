"""Tests for canonical lineage serialisation and strict deserialisation."""

import json
from copy import deepcopy

import pytest

from inference_lab.provenance.derived import (
    DerivedRecordMetadata,
)
from inference_lab.provenance.errors import (
    InvalidDerivedRecordError,
    InvalidLineageGraphError,
    InvalidLineageSerialisationError,
    UnsupportedLineageSchemaVersionError,
)
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.lineage import (
    InputRole,
)
from inference_lab.provenance.operations import (
    ComponentRole,
    ProducingComponent,
)
from inference_lab.provenance.serialisation import (
    derived_record_payload,
    deserialise_derived_record,
    deserialise_lineage_bundle,
    lineage_bundle_payload,
    serialise_derived_record,
    serialise_lineage_bundle,
)
from inference_lab.provenance.validation import KnownSourceSpan, LineageBundle

from .fixtures import (  # noqa: TID252
    derived_record,
    parent_inputs,
    producing_operation,
    source_inputs,
)

_DOCUMENT_A = DocumentVersionId.from_str("11111111-1111-4111-8111-111111111111")
_DOCUMENT_B = DocumentVersionId.from_str("22222222-2222-4222-8222-222222222222")
_SPAN_A = SourceSpanId.from_str("33333333-3333-4333-8333-333333333333")
_SPAN_B = SourceSpanId.from_str("44444444-4444-4444-8444-444444444444")
_RECORD_A = RecordId.from_str("55555555-5555-4555-8555-555555555555")
_RECORD_B = RecordId.from_str("66666666-6666-4666-8666-666666666666")
_RECORD_C = RecordId.from_str("77777777-7777-4777-8777-777777777777")
_RECORD_D = RecordId.from_str("88888888-8888-4888-8888-888888888888")


_OPERATION = producing_operation(
    kind="summarisation",
    implementation="traceable-summariser",
    contract_version="1.2.0",
    configuration_digest="a" * 64,
    components=(
        ProducingComponent(
            role=ComponentRole("generator-model"),
            implementation="local-generator-adapter",
            contract_version="2.0.0",
            artefact_identity="organisation/model",
            artefact_revision="revision-a",
        ),
        ProducingComponent(
            role=ComponentRole("prompt"),
            implementation="summary-prompt",
            contract_version="1.0.0",
            artefact_identity="topic-summary",
            artefact_revision="v2",
        ),
    ),
)


def _records() -> tuple[DerivedRecordMetadata, ...]:
    first = derived_record(
        _RECORD_A,
        record_kind="document-summary",
        inputs=source_inputs(
            _DOCUMENT_A,
            _SPAN_A,
            input_role=InputRole("supporting-passage"),
            input_ordinal=0,
        ),
        operation=_OPERATION,
        output_digest="b" * 64,
    )
    second = derived_record(
        _RECORD_B,
        record_kind="document-summary",
        inputs=source_inputs(
            _DOCUMENT_B,
            _SPAN_B,
            input_role=InputRole("supporting-passage"),
            input_ordinal=0,
        ),
        operation=_OPERATION,
        output_digest="c" * 64,
    )
    topic = derived_record(
        _RECORD_C,
        record_kind="topic-summary",
        inputs=parent_inputs(
            _RECORD_A,
            _RECORD_B,
            input_role=InputRole("summary-input"),
            include_ordinals=True,
        ),
        operation=_OPERATION,
        output_digest="d" * 64,
    )
    replacement = derived_record(
        _RECORD_D,
        record_kind="topic-summary",
        inputs=parent_inputs(
            _RECORD_A,
            _RECORD_B,
            input_role=InputRole("summary-input"),
            include_ordinals=True,
        ),
        operation=_OPERATION,
        output_digest="e" * 64,
        supersedes_record_id=_RECORD_C,
    )
    return first, second, topic, replacement


def _bundle() -> LineageBundle:
    return LineageBundle.from_records(
        source_spans=(
            KnownSourceSpan(
                document_version_id=_DOCUMENT_B,
                source_span_id=_SPAN_B,
            ),
            KnownSourceSpan(
                document_version_id=_DOCUMENT_A,
                source_span_id=_SPAN_A,
            ),
        ),
        records=tuple(reversed(_records())),
    )


def _encode(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _record_json() -> dict[str, object]:
    return json.loads(serialise_derived_record(_records()[0]))


def _bundle_json() -> dict[str, object]:
    return json.loads(serialise_lineage_bundle(_bundle()))


def test_derived_record_payload_uses_canonical_full_field_names() -> None:
    payload = derived_record_payload(_records()[0])

    assert set(payload) == {
        "record_id",
        "record_kind",
        "lineage_schema_version",
        "inputs",
        "operation",
        "output_digest",
        "derivation_fingerprint",
        "supersedes_record_id",
    }
    assert payload["record_id"] == _RECORD_A.as_str()
    assert "doc_id" not in payload
    assert "corpus_version_id" not in payload


def test_derived_record_round_trip_rebuilds_typed_identifiers() -> None:
    record = _records()[0]

    restored = deserialise_derived_record(serialise_derived_record(record))

    assert restored == record
    assert isinstance(restored.record_id, RecordId)
    assert isinstance(
        restored.inputs.source_anchors[0].document_version_id,
        DocumentVersionId,
    )
    assert isinstance(restored.inputs.source_anchors[0].source_span_id, SourceSpanId)


def test_complete_bundle_round_trip_is_exact_and_validated() -> None:
    bundle = _bundle()

    restored = deserialise_lineage_bundle(serialise_lineage_bundle(bundle))

    assert restored == bundle
    assert isinstance(restored.records[2].inputs.parent_records[0].record_id, RecordId)
    assert restored.records[-1].supersedes_record_id == _RECORD_C


def test_serialisation_is_byte_identical_and_newline_terminated() -> None:
    first = serialise_lineage_bundle(_bundle())
    second = serialise_lineage_bundle(_bundle())

    assert first == second
    assert first.endswith(b"\n")
    assert first.startswith(b'{"edges":')
    assert first == _encode(lineage_bundle_payload(_bundle()))


def test_bundle_serialisation_is_independent_of_caller_collection_order() -> None:
    canonical = _bundle()
    reversed_bundle = LineageBundle(
        source_spans=tuple(reversed(canonical.source_spans)),
        records=tuple(reversed(canonical.records)),
        edges=tuple(reversed(canonical.edges)),
    )

    assert serialise_lineage_bundle(reversed_bundle) == serialise_lineage_bundle(
        canonical
    )


def test_deserialisation_canonicalises_semantically_identical_input_order() -> None:
    payload = _bundle_json()
    records = payload["records"]
    assert isinstance(records, list)
    topic = records[2]
    assert isinstance(topic, dict)
    inputs = topic["inputs"]
    assert isinstance(inputs, dict)
    parents = inputs["parent_records"]
    assert isinstance(parents, list)
    parents.reverse()

    restored = deserialise_lineage_bundle(_encode(payload))

    assert serialise_lineage_bundle(restored) == serialise_lineage_bundle(_bundle())


@pytest.mark.parametrize(
    "data",
    [
        b"\xff",
        b"{",
        b"[]",
        b'"lineage"',
        b'{"lineage_schema_version":"1.0.0","value":NaN}',
        b'{"lineage_schema_version":"1.0.0","lineage_schema_version":"1.0.0"}',
    ],
    ids=["utf8", "json", "array", "string", "non-finite", "duplicate-key"],
)
def test_malformed_json_boundaries_fail_closed(data: bytes) -> None:
    with pytest.raises(InvalidLineageSerialisationError):
        deserialise_lineage_bundle(data)


def test_missing_and_unknown_record_fields_are_rejected() -> None:
    missing = _record_json()
    missing.pop("inputs")
    unknown = _record_json()
    unknown["payload"] = "private chunk text"

    with pytest.raises(InvalidLineageSerialisationError, match="canonical fields"):
        deserialise_derived_record(_encode(missing))
    with pytest.raises(InvalidLineageSerialisationError, match="canonical fields"):
        deserialise_derived_record(_encode(unknown))


def test_missing_and_unknown_bundle_fields_are_rejected() -> None:
    missing = _bundle_json()
    missing.pop("edges")
    unknown = _bundle_json()
    unknown["storage_row_id"] = 42

    with pytest.raises(InvalidLineageSerialisationError, match="canonical fields"):
        deserialise_lineage_bundle(_encode(missing))
    with pytest.raises(InvalidLineageSerialisationError, match="canonical fields"):
        deserialise_lineage_bundle(_encode(unknown))


def test_nested_unknown_fields_are_rejected() -> None:
    payload = _record_json()
    operation = payload["operation"]
    assert isinstance(operation, dict)
    operation["model_path"] = "/private/model"

    with pytest.raises(InvalidLineageSerialisationError, match="operation"):
        deserialise_derived_record(_encode(payload))


@pytest.mark.parametrize("version", ["", "2.0.0", "1.0.0+local"])
def test_unsupported_record_schema_is_rejected_before_deeper_fields(
    version: str,
) -> None:
    payload = _record_json()
    payload["lineage_schema_version"] = version
    payload["record_id"] = "private-invalid-id"
    payload.pop("inputs")

    with pytest.raises(UnsupportedLineageSchemaVersionError) as captured:
        deserialise_derived_record(_encode(payload))

    assert "private-invalid-id" not in str(captured.value)


def test_unsupported_bundle_schema_is_rejected_before_records() -> None:
    payload = _bundle_json()
    payload["lineage_schema_version"] = "2.0.0"
    payload["records"] = "private-invalid-records"

    with pytest.raises(UnsupportedLineageSchemaVersionError) as captured:
        deserialise_lineage_bundle(_encode(payload))

    assert "private-invalid-records" not in str(captured.value)


def test_unsupported_nested_record_schema_is_rejected_before_record_shape() -> None:
    payload = _bundle_json()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    record["lineage_schema_version"] = "2.0.0"
    record.pop("inputs")

    with pytest.raises(UnsupportedLineageSchemaVersionError):
        deserialise_lineage_bundle(_encode(payload))


def test_missing_schema_version_is_rejected_explicitly() -> None:
    record = _record_json()
    record.pop("lineage_schema_version")
    bundle = _bundle_json()
    bundle.pop("lineage_schema_version")

    with pytest.raises(InvalidLineageSerialisationError, match="requires"):
        deserialise_derived_record(_encode(record))
    with pytest.raises(InvalidLineageSerialisationError, match="requires"):
        deserialise_lineage_bundle(_encode(bundle))


@pytest.mark.parametrize("ordinal", [True, 1.5, "1", []])
def test_input_ordinals_reject_implicit_coercion(ordinal: object) -> None:
    payload = _record_json()
    inputs = payload["inputs"]
    assert isinstance(inputs, dict)
    anchors = inputs["source_anchors"]
    assert isinstance(anchors, list)
    anchor = anchors[0]
    assert isinstance(anchor, dict)
    anchor["input_ordinal"] = ordinal

    with pytest.raises(InvalidLineageSerialisationError, match="input_ordinal"):
        deserialise_derived_record(_encode(payload))


def test_identifier_values_reject_implicit_coercion_and_noncanonical_text() -> None:
    numeric = _record_json()
    numeric["record_id"] = 5
    noncanonical = _record_json()
    noncanonical["record_id"] = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"

    with pytest.raises(InvalidLineageSerialisationError, match="record_id"):
        deserialise_derived_record(_encode(numeric))
    with pytest.raises(InvalidLineageSerialisationError, match="record_id"):
        deserialise_derived_record(_encode(noncanonical))


def test_mismatched_fingerprint_is_rejected_without_payload_echo() -> None:
    payload = _record_json()
    fingerprint = payload["derivation_fingerprint"]
    assert isinstance(fingerprint, dict)
    fingerprint["digest"] = "f" * 64
    payload["payload_marker"] = "private-output-marker"
    payload.pop("payload_marker")

    with pytest.raises(InvalidDerivedRecordError) as captured:
        deserialise_derived_record(_encode(payload))

    assert "private-output-marker" not in str(captured.value)


def test_edge_union_requires_matching_discriminator_and_null_branch() -> None:
    payload = _bundle_json()
    edges = payload["edges"]
    assert isinstance(edges, list)
    source_edge = next(
        edge
        for edge in edges
        if isinstance(edge, dict) and edge.get("input_type") == "source_anchor"
    )
    source_edge["parent_record"] = {
        "record_id": _RECORD_A.as_str(),
        "input_role": None,
        "input_ordinal": None,
    }

    with pytest.raises(InvalidLineageSerialisationError, match="null parent_record"):
        deserialise_lineage_bundle(_encode(payload))


def test_unknown_edge_input_type_is_rejected() -> None:
    payload = _bundle_json()
    edges = payload["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["input_type"] = "storage-row"

    with pytest.raises(InvalidLineageSerialisationError, match="input_type"):
        deserialise_lineage_bundle(_encode(payload))


def test_complete_bundle_graph_is_validated_after_decoding() -> None:
    payload = _bundle_json()
    edges = payload["edges"]
    assert isinstance(edges, list)
    edges.pop()

    with pytest.raises(InvalidLineageGraphError, match="exactly match"):
        deserialise_lineage_bundle(_encode(payload))


def test_invalid_graph_cannot_be_serialised() -> None:
    bundle = _bundle()
    invalid = LineageBundle(
        source_spans=bundle.source_spans,
        records=bundle.records,
        edges=bundle.edges[:-1],
    )

    with pytest.raises(InvalidLineageGraphError):
        serialise_lineage_bundle(invalid)


def test_serialisation_apis_reject_wrong_runtime_types() -> None:
    with pytest.raises(TypeError, match="record"):
        derived_record_payload(_bundle())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bundle"):
        lineage_bundle_payload(_records()[0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="data must be bytes"):
        deserialise_derived_record("{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="data must be bytes"):
        deserialise_lineage_bundle(bytearray(b"{}"))  # type: ignore[arg-type]


def test_serialisation_errors_do_not_echo_rejected_private_values() -> None:
    private_marker = "private record kind marker"
    payload = deepcopy(_record_json())
    payload["record_kind"] = private_marker

    with pytest.raises(InvalidLineageSerialisationError) as captured:
        deserialise_derived_record(_encode(payload))

    assert private_marker not in str(captured.value)
