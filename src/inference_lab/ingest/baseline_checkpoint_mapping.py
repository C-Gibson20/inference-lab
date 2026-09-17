"""Explicit mappings between ingest checkpoint contracts and storage records."""

from inference_lab.ingest.baseline_checkpoints import (
    BaselineIngestManifest,
    BaselineManifestState,
)
from inference_lab.storage.baseline_checkpoint_records import (
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
