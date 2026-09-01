"""Tests for the published derived-record lineage API boundary."""

import pytest
from structlog.testing import capture_logs

import inference_lab.provenance as provenance
from inference_lab.provenance import (
    InvalidLineageError,
    InvalidLineageSerialisationError,
    LineageBundle,
    deserialise_lineage_bundle,
    serialise_lineage_bundle,
    validate_lineage_bundle,
)

_LINEAGE_EXPORTS = {
    "CURRENT_LINEAGE_SCHEMA_VERSION",
    "SUPPORTED_LINEAGE_SCHEMA_VERSIONS",
    "ComponentRole",
    "ConfigurationIdentity",
    "DerivationFingerprint",
    "DerivedRecordKind",
    "DerivedRecordMetadata",
    "DigestAlgorithm",
    "InputRole",
    "InvalidDerivedRecordError",
    "InvalidLineageError",
    "InvalidLineageGraphError",
    "InvalidLineageMetadataError",
    "InvalidLineageReferenceError",
    "InvalidLineageSerialisationError",
    "KnownSourceSpan",
    "LineageBundle",
    "LineageEdge",
    "LineageInput",
    "LineageInputs",
    "OperationKind",
    "OutputDigest",
    "ParentRecordReference",
    "ProducingComponent",
    "ProducingOperation",
    "SourceAnchor",
    "UnsupportedLineageSchemaVersionError",
    "build_derivation_fingerprint",
    "deserialise_derived_record",
    "deserialise_lineage_bundle",
    "serialise_derived_record",
    "serialise_lineage_bundle",
    "validate_lineage_bundle",
    "validate_lineage_schema_version",
}


def test_lineage_contract_is_published_through_provenance() -> None:
    """Expose every supported lineage value and operation from the package."""
    assert _LINEAGE_EXPORTS <= set(provenance.__all__)
    assert all(hasattr(provenance, name) for name in _LINEAGE_EXPORTS)


def test_mutable_payload_builders_are_not_top_level_exports() -> None:
    """Keep the package API focused on immutable values and canonical bytes."""
    assert "derived_record_payload" not in provenance.__all__
    assert "lineage_bundle_payload" not in provenance.__all__


def test_all_published_lineage_failures_are_value_errors() -> None:
    """Keep correctly typed invariant failures under one catchable base."""
    error_types = (
        provenance.InvalidDerivedRecordError,
        provenance.InvalidLineageGraphError,
        provenance.InvalidLineageMetadataError,
        provenance.InvalidLineageReferenceError,
        provenance.InvalidLineageSerialisationError,
        provenance.UnsupportedLineageSchemaVersionError,
    )

    assert issubclass(InvalidLineageError, ValueError)
    assert all(
        issubclass(error_type, InvalidLineageError) for error_type in error_types
    )


def test_public_lineage_value_operations_do_not_emit_logs() -> None:
    """Leave operational logging to persistence and transaction boundaries."""
    bundle = LineageBundle()

    with capture_logs() as logs:
        assert validate_lineage_bundle(bundle) is bundle
        encoded = serialise_lineage_bundle(bundle)
        assert deserialise_lineage_bundle(encoded) == bundle

    assert logs == []


def test_rejected_private_payload_is_not_logged_or_echoed() -> None:
    """Reject private serialised content without exposing it diagnostically."""
    private_marker = "private-document-text-marker"
    data = f'{{"payload":"{private_marker}"}}'.encode()

    with capture_logs() as logs:
        with pytest.raises(InvalidLineageSerialisationError) as captured:
            deserialise_lineage_bundle(data)

    assert private_marker not in str(captured.value)
    assert logs == []
