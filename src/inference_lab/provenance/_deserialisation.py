"""Domain reconstruction from strict lineage JSON payloads."""

from inference_lab._json import JsonValue
from inference_lab.provenance._json_validation import (
    load_json,
    parse_identifier,
    require_array,
    require_object,
    require_optional_integer,
    require_optional_string,
    require_schema_version,
    require_string,
)
from inference_lab.provenance.derived import (
    DerivationFingerprint,
    DerivedRecordKind,
    DerivedRecordMetadata,
    OutputDigest,
)
from inference_lab.provenance.errors import InvalidLineageSerialisationError
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
from inference_lab.provenance.operations import (
    ComponentRole,
    ConfigurationIdentity,
    DigestAlgorithm,
    OperationKind,
    ProducingComponent,
    ProducingOperation,
)
from inference_lab.provenance.validation import (
    KnownSourceSpan,
    LineageBundle,
    validate_lineage_bundle,
)


def _parse_role(value: JsonValue) -> InputRole | None:
    """Parse one optional input-role annotation."""
    role = require_optional_string(value, field_name="input_role")
    if role is None:
        return None
    try:
        return InputRole.from_str(role)
    except ValueError:
        raise InvalidLineageSerialisationError(
            "input_role must be a canonical role."
        ) from None


def _parse_source_anchor(value: JsonValue) -> SourceAnchor:
    """Parse one strict source-anchor object."""
    payload = require_object(
        value,
        field_name="source_anchor",
        keys=frozenset(
            {"document_version_id", "source_span_id", "input_role", "input_ordinal"}
        ),
    )
    return SourceAnchor(
        document_version_id=parse_identifier(
            payload["document_version_id"],
            field_name="document_version_id",
            identifier_type=DocumentVersionId.from_str,
        ),
        source_span_id=parse_identifier(
            payload["source_span_id"],
            field_name="source_span_id",
            identifier_type=SourceSpanId.from_str,
        ),
        input_role=_parse_role(payload["input_role"]),
        input_ordinal=require_optional_integer(
            payload["input_ordinal"], field_name="input_ordinal"
        ),
    )


def _parse_parent_record(value: JsonValue) -> ParentRecordReference:
    """Parse one strict parent-record reference object."""
    payload = require_object(
        value,
        field_name="parent_record",
        keys=frozenset({"record_id", "input_role", "input_ordinal"}),
    )
    return ParentRecordReference(
        record_id=parse_identifier(
            payload["record_id"],
            field_name="record_id",
            identifier_type=RecordId.from_str,
        ),
        input_role=_parse_role(payload["input_role"]),
        input_ordinal=require_optional_integer(
            payload["input_ordinal"], field_name="input_ordinal"
        ),
    )


def _parse_digest_algorithm(value: JsonValue, *, field_name: str) -> DigestAlgorithm:
    """Parse one supported lineage digest algorithm."""
    algorithm = require_string(value, field_name=field_name)
    try:
        return DigestAlgorithm(algorithm)
    except ValueError:
        raise InvalidLineageSerialisationError(
            f"{field_name} is not a supported digest algorithm."
        ) from None


def _parse_configuration(value: JsonValue) -> ConfigurationIdentity:
    """Parse one canonical configuration identity."""
    payload = require_object(
        value,
        field_name="configuration_identity",
        keys=frozenset({"algorithm", "digest"}),
    )
    return ConfigurationIdentity(
        algorithm=_parse_digest_algorithm(
            payload["algorithm"], field_name="configuration_identity.algorithm"
        ),
        digest=require_string(
            payload["digest"], field_name="configuration_identity.digest"
        ),
    )


def _parse_component(value: JsonValue) -> ProducingComponent:
    """Parse one producing-component object."""
    payload = require_object(
        value,
        field_name="component",
        keys=frozenset(
            {
                "role",
                "implementation",
                "contract_version",
                "artefact_identity",
                "artefact_revision",
            }
        ),
    )
    try:
        role = ComponentRole.from_str(
            require_string(payload["role"], field_name="component.role")
        )
    except ValueError:
        raise InvalidLineageSerialisationError(
            "component.role must be canonical."
        ) from None
    return ProducingComponent(
        role=role,
        implementation=require_string(
            payload["implementation"], field_name="component.implementation"
        ),
        contract_version=require_string(
            payload["contract_version"], field_name="component.contract_version"
        ),
        artefact_identity=require_optional_string(
            payload["artefact_identity"], field_name="component.artefact_identity"
        ),
        artefact_revision=require_optional_string(
            payload["artefact_revision"], field_name="component.artefact_revision"
        ),
    )


def _parse_operation(value: JsonValue) -> ProducingOperation:
    """Parse one producing-operation object."""
    payload = require_object(
        value,
        field_name="operation",
        keys=frozenset(
            {
                "kind",
                "implementation",
                "contract_version",
                "configuration_identity",
                "components",
            }
        ),
    )
    try:
        kind = OperationKind.from_str(
            require_string(payload["kind"], field_name="operation.kind")
        )
    except ValueError:
        raise InvalidLineageSerialisationError(
            "operation.kind must be canonical."
        ) from None
    components = tuple(
        _parse_component(component)
        for component in require_array(
            payload["components"], field_name="operation.components"
        )
    )
    return ProducingOperation(
        kind=kind,
        implementation=require_string(
            payload["implementation"], field_name="operation.implementation"
        ),
        contract_version=require_string(
            payload["contract_version"], field_name="operation.contract_version"
        ),
        configuration_identity=_parse_configuration(payload["configuration_identity"]),
        components=components,
    )


def _parse_output_digest(value: JsonValue) -> OutputDigest:
    """Parse one output digest object."""
    payload = require_object(
        value,
        field_name="output_digest",
        keys=frozenset({"algorithm", "digest"}),
    )
    return OutputDigest(
        algorithm=_parse_digest_algorithm(
            payload["algorithm"], field_name="output_digest.algorithm"
        ),
        digest=require_string(payload["digest"], field_name="output_digest.digest"),
    )


def _parse_fingerprint(value: JsonValue) -> DerivationFingerprint:
    """Parse one derivation fingerprint object."""
    payload = require_object(
        value,
        field_name="derivation_fingerprint",
        keys=frozenset({"algorithm", "digest"}),
    )
    return DerivationFingerprint(
        algorithm=_parse_digest_algorithm(
            payload["algorithm"], field_name="derivation_fingerprint.algorithm"
        ),
        digest=require_string(
            payload["digest"], field_name="derivation_fingerprint.digest"
        ),
    )


def _parse_inputs(value: JsonValue) -> LineageInputs:
    """Parse canonical direct lineage inputs."""
    payload = require_object(
        value,
        field_name="inputs",
        keys=frozenset({"source_anchors", "parent_records"}),
    )
    return LineageInputs(
        source_anchors=tuple(
            _parse_source_anchor(anchor)
            for anchor in require_array(
                payload["source_anchors"], field_name="inputs.source_anchors"
            )
        ),
        parent_records=tuple(
            _parse_parent_record(parent)
            for parent in require_array(
                payload["parent_records"], field_name="inputs.parent_records"
            )
        ),
    )


_RECORD_KEYS = frozenset(
    {
        "record_id",
        "record_kind",
        "lineage_schema_version",
        "inputs",
        "operation",
        "output_digest",
        "derivation_fingerprint",
        "supersedes_record_id",
    }
)


def _parse_record_payload(value: JsonValue) -> DerivedRecordMetadata:
    """Parse one complete derived-record payload."""
    unvalidated_payload = require_object(value, field_name="record")
    schema_version = require_schema_version(unvalidated_payload)
    payload = require_object(value, field_name="record", keys=_RECORD_KEYS)
    try:
        record_kind = DerivedRecordKind.from_str(
            require_string(payload["record_kind"], field_name="record_kind")
        )
    except ValueError:
        raise InvalidLineageSerialisationError(
            "record_kind must be canonical."
        ) from None
    supersedes_value = require_optional_string(
        payload["supersedes_record_id"], field_name="supersedes_record_id"
    )
    return DerivedRecordMetadata(
        record_id=parse_identifier(
            payload["record_id"],
            field_name="record_id",
            identifier_type=RecordId.from_str,
        ),
        record_kind=record_kind,
        lineage_schema_version=schema_version,
        inputs=_parse_inputs(payload["inputs"]),
        operation=_parse_operation(payload["operation"]),
        output_digest=_parse_output_digest(payload["output_digest"]),
        derivation_fingerprint=_parse_fingerprint(payload["derivation_fingerprint"]),
        supersedes_record_id=(
            parse_identifier(
                supersedes_value,
                field_name="supersedes_record_id",
                identifier_type=RecordId.from_str,
            )
            if supersedes_value is not None
            else None
        ),
    )


def deserialise_derived_record(data: bytes) -> DerivedRecordMetadata:
    """Deserialise one strict derived-record JSON payload."""
    payload = load_json(data)
    require_schema_version(payload)
    return _parse_record_payload(payload)


def _parse_known_source(value: JsonValue) -> KnownSourceSpan:
    """Parse one source-span catalogue entry."""
    payload = require_object(
        value,
        field_name="known_source_span",
        keys=frozenset({"document_version_id", "source_span_id"}),
    )
    return KnownSourceSpan(
        document_version_id=parse_identifier(
            payload["document_version_id"],
            field_name="document_version_id",
            identifier_type=DocumentVersionId.from_str,
        ),
        source_span_id=parse_identifier(
            payload["source_span_id"],
            field_name="source_span_id",
            identifier_type=SourceSpanId.from_str,
        ),
    )


def _parse_edge(value: JsonValue) -> LineageEdge:
    """Parse one directed lineage edge with an explicit input type."""
    payload = require_object(
        value,
        field_name="edge",
        keys=frozenset(
            {"input_type", "child_record_id", "source_anchor", "parent_record"}
        ),
    )
    input_type = require_string(payload["input_type"], field_name="input_type")
    if input_type == "source_anchor":
        if payload["parent_record"] is not None:
            raise InvalidLineageSerialisationError(
                "source edge requires a null parent_record."
            )
        input_reference = _parse_source_anchor(payload["source_anchor"])
    elif input_type == "parent_record":
        if payload["source_anchor"] is not None:
            raise InvalidLineageSerialisationError(
                "parent edge requires a null source_anchor."
            )
        input_reference = _parse_parent_record(payload["parent_record"])
    else:
        raise InvalidLineageSerialisationError(
            "input_type must identify a supported lineage input."
        )
    return LineageEdge(
        input_reference=input_reference,
        child_record_id=parse_identifier(
            payload["child_record_id"],
            field_name="child_record_id",
            identifier_type=RecordId.from_str,
        ),
    )


_BUNDLE_KEYS = frozenset({"lineage_schema_version", "source_spans", "records", "edges"})


def deserialise_lineage_bundle(data: bytes) -> LineageBundle:
    """Deserialise and validate one complete lineage bundle."""
    payload = load_json(data)
    schema_version = require_schema_version(payload)
    payload = require_object(payload, field_name="bundle", keys=_BUNDLE_KEYS)
    records = tuple(
        _parse_record_payload(record)
        for record in require_array(payload["records"], field_name="records")
    )
    if any(record.lineage_schema_version != schema_version for record in records):
        raise InvalidLineageSerialisationError(
            "record lineage_schema_version must match the bundle version."
        )
    bundle = LineageBundle(
        source_spans=tuple(
            _parse_known_source(source)
            for source in require_array(
                payload["source_spans"], field_name="source_spans"
            )
        ),
        records=records,
        edges=tuple(
            _parse_edge(edge)
            for edge in require_array(payload["edges"], field_name="edges")
        ),
    )
    return validate_lineage_bundle(bundle)
