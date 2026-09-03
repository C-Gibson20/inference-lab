"""Reusable builders for provenance contract tests."""

from inference_lab.provenance.derived import (
    DerivedRecordKind,
    DerivedRecordMetadata,
    OutputDigest,
    build_derivation_fingerprint,
)
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.lineage import (
    InputRole,
    LineageInputs,
    ParentRecordReference,
    SourceAnchor,
)
from inference_lab.provenance.operations import (
    ConfigurationIdentity,
    DigestAlgorithm,
    OperationKind,
    ProducingComponent,
    ProducingOperation,
)
from inference_lab.provenance.versioning import CURRENT_LINEAGE_SCHEMA_VERSION


def source_inputs(
    document_version_id: DocumentVersionId,
    source_span_id: SourceSpanId,
    *,
    input_role: InputRole | None = None,
    input_ordinal: int | None = None,
) -> LineageInputs:
    """Build lineage inputs containing one source anchor."""
    return LineageInputs(
        source_anchors=(
            SourceAnchor(
                document_version_id=document_version_id,
                source_span_id=source_span_id,
                input_role=input_role,
                input_ordinal=input_ordinal,
            ),
        )
    )


def parent_inputs(
    *record_ids: RecordId,
    input_role: InputRole | None = None,
    include_ordinals: bool = False,
) -> LineageInputs:
    """Build lineage inputs containing canonically ordered parent references."""
    return LineageInputs(
        parent_records=tuple(
            ParentRecordReference(
                record_id=record_id,
                input_role=input_role,
                input_ordinal=ordinal if include_ordinals else None,
            )
            for ordinal, record_id in enumerate(record_ids)
        )
    )


def producing_operation(
    *,
    kind: str,
    implementation: str,
    contract_version: str,
    configuration_digest: str,
    components: tuple[ProducingComponent, ...] = (),
) -> ProducingOperation:
    """Build producing-operation metadata with a SHA-256 configuration identity."""
    return ProducingOperation(
        kind=OperationKind(kind),
        implementation=implementation,
        contract_version=contract_version,
        configuration_identity=ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest=configuration_digest,
        ),
        components=components,
    )


def derived_record(
    record_id: RecordId,
    *,
    record_kind: str,
    inputs: LineageInputs,
    operation: ProducingOperation,
    output_digest: str,
    supersedes_record_id: RecordId | None = None,
) -> DerivedRecordMetadata:
    """Build derived metadata with its matching deterministic fingerprint."""
    kind = DerivedRecordKind(record_kind)
    digest = OutputDigest(DigestAlgorithm.SHA256, output_digest)
    return DerivedRecordMetadata(
        record_id=record_id,
        record_kind=kind,
        lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
        inputs=inputs,
        operation=operation,
        output_digest=digest,
        derivation_fingerprint=build_derivation_fingerprint(
            record_kind=kind,
            lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
            inputs=inputs,
            operation=operation,
            output_digest=digest,
        ),
        supersedes_record_id=supersedes_record_id,
    )
