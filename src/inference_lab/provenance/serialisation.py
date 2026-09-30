"""Canonical JSON encoding for derived-record lineage contracts."""

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.provenance.derived import (
    DerivationFingerprint,
    DerivedRecordMetadata,
    OutputDigest,
)
from inference_lab.provenance.lineage import (
    InputRole,
    LineageEdge,
    ParentRecordReference,
    SourceAnchor,
)
from inference_lab.provenance.operations import ProducingComponent, ProducingOperation
from inference_lab.provenance.validation import (
    LineageBundle,
    validate_lineage_bundle,
)
from inference_lab.provenance.versioning import CURRENT_LINEAGE_SCHEMA_VERSION


def _input_annotation_payload(
    input_role: InputRole | None,
    input_ordinal: int | None,
) -> JsonObject:
    """Return canonical JSON fields for semantic input annotations."""
    return {
        "input_role": input_role.as_str() if input_role is not None else None,
        "input_ordinal": input_ordinal,
    }


def _source_anchor_payload(anchor: SourceAnchor) -> JsonObject:
    """Return canonical JSON data for one source input."""
    return {
        "document_version_id": anchor.document_version_id.as_str(),
        "source_span_id": anchor.source_span_id.as_str(),
        **_input_annotation_payload(anchor.input_role, anchor.input_ordinal),
    }


def _parent_record_payload(parent: ParentRecordReference) -> JsonObject:
    """Return canonical JSON data for one parent-record input."""
    return {
        "record_id": parent.record_id.as_str(),
        **_input_annotation_payload(parent.input_role, parent.input_ordinal),
    }


def _component_payload(component: ProducingComponent) -> JsonObject:
    """Return canonical JSON data for one producing component."""
    return {
        "role": component.role.as_str(),
        "implementation": component.implementation,
        "contract_version": component.contract_version,
        "artefact_identity": component.artefact_identity,
        "artefact_revision": component.artefact_revision,
    }


def _operation_payload(operation: ProducingOperation) -> JsonObject:
    """Return canonical JSON data for one producing operation."""
    return {
        "kind": operation.kind.as_str(),
        "implementation": operation.implementation,
        "contract_version": operation.contract_version,
        "configuration_identity": {
            "algorithm": operation.configuration_identity.algorithm.value,
            "digest": operation.configuration_identity.digest,
        },
        "components": [
            _component_payload(component) for component in operation.components
        ],
    }


def _digest_payload(
    digest: OutputDigest | DerivationFingerprint,
) -> JsonObject:
    """Return canonical JSON data for one typed lineage digest."""
    return {"algorithm": digest.algorithm.value, "digest": digest.digest}


def derived_record_payload(record: DerivedRecordMetadata) -> JsonObject:
    """Return canonical JSON-compatible data for one derived record."""
    if not isinstance(record, DerivedRecordMetadata):
        raise TypeError("record must be a DerivedRecordMetadata.")
    return {
        "record_id": record.record_id.as_str(),
        "record_kind": record.record_kind.as_str(),
        "lineage_schema_version": record.lineage_schema_version,
        "inputs": {
            "source_anchors": [
                _source_anchor_payload(anchor)
                for anchor in record.inputs.source_anchors
            ],
            "parent_records": [
                _parent_record_payload(parent)
                for parent in record.inputs.parent_records
            ],
        },
        "operation": _operation_payload(record.operation),
        "output_digest": _digest_payload(record.output_digest),
        "derivation_fingerprint": _digest_payload(record.derivation_fingerprint),
        "supersedes_record_id": (
            record.supersedes_record_id.as_str()
            if record.supersedes_record_id is not None
            else None
        ),
    }


def _edge_payload(edge: LineageEdge) -> JsonObject:
    """Return canonical JSON data for one directed lineage edge."""
    reference = edge.input_reference
    if isinstance(reference, SourceAnchor):
        return {
            "input_type": "source_anchor",
            "child_record_id": edge.child_record_id.as_str(),
            "source_anchor": _source_anchor_payload(reference),
            "parent_record": None,
        }
    return {
        "input_type": "parent_record",
        "child_record_id": edge.child_record_id.as_str(),
        "source_anchor": None,
        "parent_record": _parent_record_payload(reference),
    }


def lineage_bundle_payload(bundle: LineageBundle) -> JsonObject:
    """Return canonical JSON-compatible data for a validated lineage bundle."""
    if not isinstance(bundle, LineageBundle):
        raise TypeError("bundle must be a LineageBundle.")
    validate_lineage_bundle(bundle)
    return {
        "lineage_schema_version": CURRENT_LINEAGE_SCHEMA_VERSION,
        "source_spans": [
            {
                "document_version_id": source.document_version_id.as_str(),
                "source_span_id": source.source_span_id.as_str(),
            }
            for source in bundle.source_spans
        ],
        "records": [derived_record_payload(record) for record in bundle.records],
        "edges": [_edge_payload(edge) for edge in bundle.edges],
    }


def serialise_derived_record(record: DerivedRecordMetadata) -> bytes:
    """Serialise one derived record as canonical UTF-8 JSON."""
    return canonical_json_bytes(derived_record_payload(record))


def serialise_lineage_bundle(bundle: LineageBundle) -> bytes:
    """Serialise one validated lineage bundle as canonical UTF-8 JSON."""
    return canonical_json_bytes(lineage_bundle_payload(bundle))
