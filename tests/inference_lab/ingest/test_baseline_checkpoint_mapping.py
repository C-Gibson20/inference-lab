"""Tests for the ingest/storage baseline checkpoint mapping boundary."""

from collections.abc import Callable

import pytest

from inference_lab.ingest.baseline_checkpoint_mapping import (
    BaselineCheckpointMappingError,
    baseline_checkpoint_from_record,
    baseline_checkpoint_to_record,
    baseline_manifest_from_record,
    baseline_manifest_to_record,
)
from inference_lab.ingest.baseline_checkpoints import (
    BaselineCheckpoint,
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
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
    RecordId,
    SourceObservationId,
    SourceSpanId,
)
from inference_lab.storage.baseline_checkpoint_records import (
    BaselineCheckpointIdentityRecord,
    BaselineCheckpointRecord,
    BaselineCheckpointRecordIdentityKind,
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
_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_DOCUMENT_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")
_RECORD_ID = RecordId("77777777-7777-4777-8777-777777777777")
_SOURCE_OBSERVATION_ID = SourceObservationId("88888888-8888-4888-8888-888888888888")
_SOURCE_SPAN_ID = SourceSpanId("99999999-9999-4999-8999-999999999999")


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


def _completed_checkpoint() -> BaselineCheckpoint:
    return BaselineCheckpoint(
        work_key=BaselineWorkKey(
            proposed_corpus_version=_PROPOSED_VERSION,
            document_version_id=_DOCUMENT_VERSION_ID,
            stage=BaselineStage.PARSE,
            configuration_identity=_CONFIGURATION,
        ),
        document_id=_DOCUMENT_ID,
        input_identities=(
            _DOCUMENT_ID,
            _DOCUMENT_VERSION_ID,
            _SOURCE_OBSERVATION_ID,
            _SOURCE_SPAN_ID,
        ),
        output_identities=(_RECORD_ID,),
        status=BaselineCheckpointStatus.COMPLETED,
    )


def _started_checkpoint() -> BaselineCheckpoint:
    checkpoint = _completed_checkpoint()
    return BaselineCheckpoint(
        work_key=checkpoint.work_key,
        document_id=checkpoint.document_id,
        input_identities=checkpoint.input_identities,
        output_identities=(),
        status=BaselineCheckpointStatus.STARTED,
    )


def _failed_checkpoint() -> BaselineCheckpoint:
    checkpoint = _completed_checkpoint()
    return BaselineCheckpoint(
        work_key=checkpoint.work_key,
        document_id=checkpoint.document_id,
        input_identities=checkpoint.input_identities,
        output_identities=(),
        status=BaselineCheckpointStatus.FAILED,
        failure_category=BaselineFailureCategory.PARSE_FAILED,
    )


@pytest.mark.parametrize(
    "checkpoint_factory",
    [_completed_checkpoint, _started_checkpoint, _failed_checkpoint],
)
def test_checkpoint_mapping_round_trips(
    checkpoint_factory: Callable[[], BaselineCheckpoint],
) -> None:
    checkpoint = checkpoint_factory()

    record = baseline_checkpoint_to_record(
        checkpoint,
        corpus_id=_CORPUS_ID,
    )
    restored = baseline_checkpoint_from_record(
        record,
        expected_corpus_id=_CORPUS_ID,
    )

    assert restored == checkpoint


def test_checkpoint_mapping_preserves_all_identity_kinds() -> None:
    record = baseline_checkpoint_to_record(
        _completed_checkpoint(),
        corpus_id=_CORPUS_ID,
    )

    assert tuple(identity.kind for identity in record.input_identities) == (
        BaselineCheckpointRecordIdentityKind.DOCUMENT,
        BaselineCheckpointRecordIdentityKind.DOCUMENT_VERSION,
        BaselineCheckpointRecordIdentityKind.SOURCE_OBSERVATION,
        BaselineCheckpointRecordIdentityKind.SOURCE_SPAN,
    )


def test_checkpoint_mapping_rejects_wrong_input_types() -> None:
    checkpoint = _completed_checkpoint()
    record = baseline_checkpoint_to_record(
        checkpoint,
        corpus_id=_CORPUS_ID,
    )

    with pytest.raises(TypeError):
        baseline_checkpoint_to_record(object(), corpus_id=_CORPUS_ID)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        baseline_checkpoint_to_record(checkpoint, corpus_id=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        baseline_checkpoint_from_record(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        baseline_checkpoint_from_record(record, expected_corpus_id=object())  # type: ignore[arg-type]


def test_checkpoint_mapping_rejects_corpus_mismatch() -> None:
    record = baseline_checkpoint_to_record(
        _completed_checkpoint(),
        corpus_id=_CORPUS_ID,
    )

    with pytest.raises(BaselineCheckpointMappingError, match="different corpus"):
        baseline_checkpoint_from_record(
            record,
            expected_corpus_id=CorpusId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        )


def _corrupt_checkpoint_record(**changes: object) -> BaselineCheckpointRecord:
    """Create a record-shaped value with one deliberately corrupt field."""
    record = baseline_checkpoint_to_record(
        _completed_checkpoint(),
        corpus_id=_CORPUS_ID,
    )
    for field_name, value in changes.items():
        object.__setattr__(record, field_name, value)
    return record


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "unknown"},
        {"status": "unknown"},
        {"failure_category": "unknown"},
        {"status": "failed", "failure_category": None},
    ],
)
def test_checkpoint_mapping_rejects_corrupt_persisted_values(
    changes: dict[str, object],
) -> None:
    record = _corrupt_checkpoint_record(**changes)

    with pytest.raises(BaselineCheckpointMappingError):
        baseline_checkpoint_from_record(record)


def test_checkpoint_mapping_rejects_corrupt_identity_kind() -> None:
    identity = object.__new__(BaselineCheckpointIdentityRecord)
    object.__setattr__(identity, "kind", "unknown")
    object.__setattr__(identity, "value", _RECORD_ID.as_str())
    record = _corrupt_checkpoint_record(input_identities=(identity,))

    with pytest.raises(BaselineCheckpointMappingError, match="identity kind"):
        baseline_checkpoint_from_record(record)


def test_checkpoint_mapping_exposes_no_payloads_or_raw_errors() -> None:
    record = _corrupt_checkpoint_record(stage="/private/source_text")

    with pytest.raises(BaselineCheckpointMappingError) as error:
        baseline_checkpoint_from_record(record)

    rendered = str(error.value)
    assert "/private" not in rendered
    assert "source_text" not in rendered


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
