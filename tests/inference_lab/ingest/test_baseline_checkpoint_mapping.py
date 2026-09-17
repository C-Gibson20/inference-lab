"""Tests for the ingest/storage baseline checkpoint mapping boundary."""

import pytest

from inference_lab.ingest.baseline_checkpoint_mapping import (
    baseline_manifest_from_record,
    baseline_manifest_to_record,
)
from inference_lab.ingest.baseline_checkpoints import (
    BaselineIngestManifest,
    BaselineManifestState,
)
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    OutputDigest,
)
from inference_lab.storage.baseline_checkpoint_records import (
    BaselineManifestRecord,
    BaselineManifestRecordState,
)
from inference_lab.storage.versioning import ComponentKind

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_PROPOSED_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_CONFIGURATION = ConfigurationIdentity(
    algorithm=DigestAlgorithm.SHA256,
    digest="0" * 64,
)
_DISCOVERY_DIGEST = OutputDigest(
    algorithm=DigestAlgorithm.SHA256,
    digest="1" * 64,
)


def _manifest() -> BaselineIngestManifest:
    return BaselineIngestManifest(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=None,
        configuration_identity=_CONFIGURATION,
        required_component_kinds=(ComponentKind("chunks"),),
        discovery_summary_identity=_DISCOVERY_DIGEST,
        state=BaselineManifestState.PREPARING,
    )


def test_manifest_mapping_round_trips_without_payloads() -> None:
    manifest = _manifest()

    record = baseline_manifest_to_record(manifest)
    restored = baseline_manifest_from_record(record)

    assert restored == manifest
    assert "source_text" not in repr(record)
    assert "chunk_text" not in repr(record)


def test_manifest_mapping_converts_lifecycle_state_by_value() -> None:
    manifest = _manifest()
    record = baseline_manifest_to_record(manifest)

    assert record.state is BaselineManifestRecordState.PREPARING


@pytest.mark.parametrize(
    "value",
    [object(), None],
)
def test_manifest_mapping_rejects_wrong_input_types(value: object) -> None:
    with pytest.raises(TypeError):
        baseline_manifest_to_record(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        baseline_manifest_from_record(value)  # type: ignore[arg-type]


def test_manifest_mapping_rejects_unsupported_persisted_state() -> None:
    record = _manifest_record_without_validation()

    with pytest.raises(ValueError, match="unsupported"):
        baseline_manifest_from_record(record)


def _manifest_record_without_validation() -> BaselineManifestRecord:
    """Build a record-shaped object to exercise corrupt-state handling."""
    record = object.__new__(BaselineManifestRecord)
    object.__setattr__(record, "corpus_id", _CORPUS_ID)
    object.__setattr__(record, "proposed_corpus_version", _PROPOSED_VERSION)
    object.__setattr__(record, "expected_corpus_version", None)
    object.__setattr__(record, "configuration_identity", _CONFIGURATION)
    object.__setattr__(record, "required_component_kinds", (ComponentKind("chunks"),))
    object.__setattr__(record, "discovery_summary_identity", _DISCOVERY_DIGEST)
    object.__setattr__(record, "state", "unsupported")
    return record
