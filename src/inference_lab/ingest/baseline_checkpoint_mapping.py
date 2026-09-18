"""Explicit mappings between ingest checkpoint contracts and storage records."""

from inference_lab.ingest.baseline_checkpoints import (
    BaselineCheckpoint,
    BaselineCheckpointIdentity,
    BaselineCheckpointStatus,
    BaselineIngestManifest,
    BaselineManifestState,
    BaselineWorkKey,
)
from inference_lab.ingest.baseline_contracts import (
    BaselineFailureCategory,
    BaselineStage,
)
from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceObservationId,
    SourceSpanId,
)
from inference_lab.storage.baseline_checkpoint_records import (
    BaselineCheckpointIdentityRecord,
    BaselineCheckpointRecord,
    BaselineCheckpointRecordFailureCategory,
    BaselineCheckpointRecordIdentityKind,
    BaselineCheckpointRecordStage,
    BaselineCheckpointRecordStatus,
    BaselineManifestRecord,
    BaselineManifestRecordState,
)


class BaselineCheckpointMappingError(ValueError):
    """Raised when a persisted baseline record cannot be reconstructed."""


def baseline_manifest_to_record(
    manifest: BaselineIngestManifest,
) -> BaselineManifestRecord:
    """Map one validated ingest manifest to its storage-owned record."""
    if not isinstance(manifest, BaselineIngestManifest):
        raise TypeError("manifest must be a BaselineIngestManifest.")
    return BaselineManifestRecord(
        corpus_id=manifest.corpus_id,
        proposed_corpus_version=manifest.proposed_corpus_version,
        expected_corpus_version=manifest.expected_corpus_version,
        configuration_identity=manifest.configuration_identity,
        required_component_kinds=manifest.required_component_kinds,
        discovery_summary_identity=manifest.discovery_summary_identity,
        state=BaselineManifestRecordState(manifest.state.value),
    )


def baseline_manifest_from_record(
    record: BaselineManifestRecord,
) -> BaselineIngestManifest:
    """Map one validated storage record back to an ingest manifest."""
    if not isinstance(record, BaselineManifestRecord):
        raise TypeError("record must be a BaselineManifestRecord.")
    try:
        state_value = record.state.value
        state = BaselineManifestState(state_value)
    except (AttributeError, TypeError, ValueError) as error:
        raise BaselineCheckpointMappingError(
            "the persisted baseline manifest state is unsupported."
        ) from error
    return BaselineIngestManifest(
        corpus_id=record.corpus_id,
        proposed_corpus_version=record.proposed_corpus_version,
        expected_corpus_version=record.expected_corpus_version,
        configuration_identity=record.configuration_identity,
        required_component_kinds=record.required_component_kinds,
        discovery_summary_identity=record.discovery_summary_identity,
        state=state,
    )


def _identity_to_record(
    identity: BaselineCheckpointIdentity,
) -> BaselineCheckpointIdentityRecord:
    """Map one typed ingest identity to a storage identity record."""
    if isinstance(identity, DocumentId):
        kind = BaselineCheckpointRecordIdentityKind.DOCUMENT
    elif isinstance(identity, DocumentVersionId):
        kind = BaselineCheckpointRecordIdentityKind.DOCUMENT_VERSION
    elif isinstance(identity, RecordId):
        kind = BaselineCheckpointRecordIdentityKind.RECORD
    elif isinstance(identity, SourceObservationId):
        kind = BaselineCheckpointRecordIdentityKind.SOURCE_OBSERVATION
    elif isinstance(identity, SourceSpanId):
        kind = BaselineCheckpointRecordIdentityKind.SOURCE_SPAN
    else:
        raise TypeError("identity must be a supported checkpoint identity.")

    return BaselineCheckpointIdentityRecord(
        kind=kind,
        value=identity.as_str(),
    )


def _identity_from_record(
    identity: BaselineCheckpointIdentityRecord,
) -> BaselineCheckpointIdentity:
    """Map one storage identity record back to a typed ingest identity."""
    try:
        if identity.kind is BaselineCheckpointRecordIdentityKind.DOCUMENT:
            return DocumentId.from_str(identity.value)
        if identity.kind is BaselineCheckpointRecordIdentityKind.DOCUMENT_VERSION:
            return DocumentVersionId.from_str(identity.value)
        if identity.kind is BaselineCheckpointRecordIdentityKind.RECORD:
            return RecordId.from_str(identity.value)
        if identity.kind is BaselineCheckpointRecordIdentityKind.SOURCE_OBSERVATION:
            return SourceObservationId.from_str(identity.value)
        if identity.kind is BaselineCheckpointRecordIdentityKind.SOURCE_SPAN:
            return SourceSpanId.from_str(identity.value)
    except (AttributeError, TypeError, ValueError) as error:
        raise BaselineCheckpointMappingError(
            "the persisted checkpoint identity is invalid."
        ) from error

    raise BaselineCheckpointMappingError(
        "the persisted checkpoint identity kind is unsupported."
    )


def baseline_checkpoint_to_record(
    checkpoint: BaselineCheckpoint,
    *,
    corpus_id: CorpusId,
) -> BaselineCheckpointRecord:
    """Map one validated ingest checkpoint to a storage record."""
    if not isinstance(checkpoint, BaselineCheckpoint):
        raise TypeError("checkpoint must be a BaselineCheckpoint.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")

    return BaselineCheckpointRecord(
        corpus_id=corpus_id,
        proposed_corpus_version=checkpoint.work_key.proposed_corpus_version,
        document_version_id=checkpoint.work_key.document_version_id,
        stage=BaselineCheckpointRecordStage(checkpoint.work_key.stage.value),
        configuration_identity=checkpoint.work_key.configuration_identity,
        document_id=checkpoint.document_id,
        input_identities=tuple(
            _identity_to_record(identity) for identity in checkpoint.input_identities
        ),
        output_identities=tuple(
            _identity_to_record(identity) for identity in checkpoint.output_identities
        ),
        status=BaselineCheckpointRecordStatus(checkpoint.status.value),
        failure_category=(
            BaselineCheckpointRecordFailureCategory(checkpoint.failure_category.value)
            if checkpoint.failure_category is not None
            else None
        ),
    )


def baseline_checkpoint_from_record(
    record: BaselineCheckpointRecord,
    *,
    expected_corpus_id: CorpusId | None = None,
) -> BaselineCheckpoint:
    """Map one storage record back to an ingest checkpoint."""
    if not isinstance(record, BaselineCheckpointRecord):
        raise TypeError("record must be a BaselineCheckpointRecord.")
    if expected_corpus_id is not None and not isinstance(
        expected_corpus_id,
        CorpusId,
    ):
        raise TypeError("expected_corpus_id must be a CorpusId or None.")
    if expected_corpus_id is not None and record.corpus_id != expected_corpus_id:
        raise BaselineCheckpointMappingError(
            "the checkpoint belongs to a different corpus."
        )

    try:
        stage = BaselineStage(record.stage.value)
        status = BaselineCheckpointStatus(record.status.value)
        failure_category = (
            BaselineFailureCategory(record.failure_category.value)
            if record.failure_category is not None
            else None
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise BaselineCheckpointMappingError(
            "the persisted checkpoint contains unsupported values."
        ) from error

    try:
        return BaselineCheckpoint(
            work_key=BaselineWorkKey(
                proposed_corpus_version=record.proposed_corpus_version,
                document_version_id=record.document_version_id,
                stage=stage,
                configuration_identity=record.configuration_identity,
            ),
            document_id=record.document_id,
            input_identities=tuple(
                _identity_from_record(identity) for identity in record.input_identities
            ),
            output_identities=tuple(
                _identity_from_record(identity) for identity in record.output_identities
            ),
            status=status,
            failure_category=failure_category,
        )
    except BaselineCheckpointMappingError:
        raise
    except (TypeError, ValueError) as error:
        raise BaselineCheckpointMappingError(
            "the persisted checkpoint is invalid."
        ) from error
