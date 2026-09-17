"""Tests for durable baseline-ingest checkpoint contracts."""

import pytest

from inference_lab.ingest.baseline_checkpoints import (
    BaselineCheckpoint,
    BaselineCheckpointStatus,
    BaselineIngestManifest,
    BaselineManifestState,
    BaselineWorkKey,
)
from inference_lab.ingest.baseline_contracts import BaselineStage
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
    RecordId,
)
from inference_lab.storage.versioning import ComponentKind

_PROPOSED_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_DOCUMENT_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")
_RECORD_ID = RecordId("77777777-7777-4777-8777-777777777777")
_CONFIGURATION = ConfigurationIdentity(
    algorithm=DigestAlgorithm.SHA256,
    digest="0" * 64,
)


def _work_key() -> BaselineWorkKey:
    return BaselineWorkKey(
        proposed_corpus_version=_PROPOSED_VERSION,
        document_version_id=_DOCUMENT_VERSION_ID,
        stage=BaselineStage.PARSE,
        configuration_identity=_CONFIGURATION,
    )


def test_work_key_is_canonically_orderable() -> None:
    work_key = _work_key()

    assert work_key.key() == (
        _PROPOSED_VERSION.as_str(),
        _DOCUMENT_VERSION_ID.as_str(),
        "parse",
        "sha256:" + ("0" * 64),
    )


def test_checkpoint_lifecycle_values_are_closed() -> None:
    assert BaselineManifestState.PREPARING.value == "preparing"
    assert BaselineManifestState.READY.value == "ready"
    assert BaselineManifestState.COMMITTED.value == "committed"
    assert BaselineManifestState.ABORTED.value == "aborted"

    assert BaselineCheckpointStatus.STARTED.value == "started"
    assert BaselineCheckpointStatus.COMPLETED.value == "completed"
    assert BaselineCheckpointStatus.FAILED.value == "failed"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("proposed_corpus_version", "not-an-id"),
        ("document_version_id", "not-an-id"),
        ("stage", "parse"),
        ("configuration_identity", "sha256:" + ("0" * 64)),
    ],
)
def test_work_key_rejects_wrong_field_types(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "proposed_corpus_version": _PROPOSED_VERSION,
        "document_version_id": _DOCUMENT_VERSION_ID,
        "stage": BaselineStage.PARSE,
        "configuration_identity": _CONFIGURATION,
    }
    values[field_name] = value

    with pytest.raises(TypeError):
        BaselineWorkKey(**values)  # type: ignore[arg-type]


def test_work_key_contains_no_private_path_or_payload() -> None:
    work_key = BaselineWorkKey(
        proposed_corpus_version=_PROPOSED_VERSION,
        document_version_id=_DOCUMENT_VERSION_ID,
        stage=BaselineStage.CHUNK,
        configuration_identity=_CONFIGURATION,
    )

    rendered = repr(work_key)

    assert "/private" not in rendered
    assert "source_text" not in rendered
    assert "chunk_text" not in rendered


def test_manifest_key_and_component_kinds_are_canonical() -> None:
    manifest = BaselineIngestManifest(
        corpus_id=CorpusId.from_str("00000000-0000-0000-0000-000000000001"),
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=None,
        configuration_identity=_CONFIGURATION,
        required_component_kinds=(
            ComponentKind("parser"),
            ComponentKind("chunks"),
        ),
        discovery_summary_identity=OutputDigest(
            DigestAlgorithm.SHA256,
            "d" * 64,
        ),
        state=BaselineManifestState.PREPARING,
    )

    assert manifest.key() == (
        "00000000-0000-0000-0000-000000000001",
        _PROPOSED_VERSION.as_str(),
    )
    assert tuple(kind.as_str() for kind in manifest.required_component_kinds) == (
        "chunks",
        "parser",
    )


def test_manifest_rejects_same_proposed_and_expected_version() -> None:
    with pytest.raises(ValueError, match="must differ"):
        BaselineIngestManifest(
            corpus_id=CorpusId.from_str("00000000-0000-0000-0000-000000000001"),
            proposed_corpus_version=_PROPOSED_VERSION,
            expected_corpus_version=_PROPOSED_VERSION,
            configuration_identity=_CONFIGURATION,
            required_component_kinds=(),
            discovery_summary_identity=OutputDigest(
                DigestAlgorithm.SHA256,
                "d" * 64,
            ),
            state=BaselineManifestState.PREPARING,
        )


def test_manifest_rejects_duplicate_component_kinds() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        BaselineIngestManifest(
            corpus_id=CorpusId.from_str("00000000-0000-0000-0000-000000000001"),
            proposed_corpus_version=_PROPOSED_VERSION,
            expected_corpus_version=None,
            configuration_identity=_CONFIGURATION,
            required_component_kinds=(
                ComponentKind("chunks"),
                ComponentKind("chunks"),
            ),
            discovery_summary_identity=OutputDigest(
                DigestAlgorithm.SHA256,
                "d" * 64,
            ),
            state=BaselineManifestState.PREPARING,
        )


def test_completed_checkpoint_preserves_typed_identities() -> None:
    checkpoint = BaselineCheckpoint(
        work_key=BaselineWorkKey(
            proposed_corpus_version=_PROPOSED_VERSION,
            document_version_id=_DOCUMENT_VERSION_ID,
            stage=BaselineStage.PARSE,
            configuration_identity=_CONFIGURATION,
        ),
        document_id=DocumentId.from_str("00000000-0000-0000-0000-000000000003"),
        input_identities=(_DOCUMENT_VERSION_ID,),
        output_identities=(RecordId.from_str("00000000-0000-0000-0000-000000000004"),),
        status=BaselineCheckpointStatus.COMPLETED,
    )

    assert checkpoint.status is BaselineCheckpointStatus.COMPLETED
    assert checkpoint.key()[0] == _PROPOSED_VERSION.as_str()


def test_started_checkpoint_rejects_output_identity() -> None:
    with pytest.raises(ValueError, match="started checkpoints"):
        BaselineCheckpoint(
            work_key=_work_key(),
            document_id=_DOCUMENT_ID,
            input_identities=(_DOCUMENT_VERSION_ID,),
            output_identities=(_RECORD_ID,),
            status=BaselineCheckpointStatus.STARTED,
        )


def test_failed_checkpoint_requires_bounded_failure_category() -> None:
    with pytest.raises(ValueError, match="require a failure category"):
        BaselineCheckpoint(
            work_key=_work_key(),
            document_id=_DOCUMENT_ID,
            input_identities=(_DOCUMENT_VERSION_ID,),
            output_identities=(),
            status=BaselineCheckpointStatus.FAILED,
        )


def test_checkpoint_rejects_duplicate_input_identity() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        BaselineCheckpoint(
            work_key=_work_key(),
            document_id=_DOCUMENT_ID,
            input_identities=(
                _DOCUMENT_VERSION_ID,
                _DOCUMENT_VERSION_ID,
            ),
            output_identities=(),
            status=BaselineCheckpointStatus.STARTED,
        )
