"""Compact value contracts for durable baseline checkpoints."""

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

_CORPUS_ID = CorpusId("00000000-0000-0000-0000-000000000001")
_PROPOSED_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_DOCUMENT_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")
_RECORD_ID = RecordId("77777777-7777-4777-8777-777777777777")
_CONFIGURATION = ConfigurationIdentity(DigestAlgorithm.SHA256, "0" * 64)


def _work_key() -> BaselineWorkKey:
    return BaselineWorkKey(
        proposed_corpus_version=_PROPOSED_VERSION,
        document_version_id=_DOCUMENT_VERSION_ID,
        stage=BaselineStage.PARSE,
        configuration_identity=_CONFIGURATION,
    )


def _manifest(
    *,
    expected_corpus_version: CorpusVersionId | None = None,
    component_kinds: tuple[ComponentKind, ...] = (
        ComponentKind("parser"),
        ComponentKind("chunks"),
    ),
) -> BaselineIngestManifest:
    return BaselineIngestManifest(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=expected_corpus_version,
        configuration_identity=_CONFIGURATION,
        required_component_kinds=component_kinds,
        discovery_summary_identity=OutputDigest(
            DigestAlgorithm.SHA256,
            "d" * 64,
        ),
        state=BaselineManifestState.PREPARING,
    )


def test_work_key_is_canonically_orderable() -> None:
    assert _work_key().key() == (
        _PROPOSED_VERSION.as_str(),
        _DOCUMENT_VERSION_ID.as_str(),
        "parse",
        "sha256:" + ("0" * 64),
    )


def test_manifest_and_checkpoint_lifecycle_states_are_closed() -> None:
    assert {state.value for state in BaselineManifestState} == {
        "preparing",
        "ready",
        "committed",
        "aborted",
    }
    assert {status.value for status in BaselineCheckpointStatus} == {
        "started",
        "completed",
        "failed",
    }


def test_manifest_identity_and_component_contracts() -> None:
    manifest = _manifest()
    assert manifest.key() == (_CORPUS_ID.as_str(), _PROPOSED_VERSION.as_str())
    assert tuple(kind.as_str() for kind in manifest.required_component_kinds) == (
        "chunks",
        "parser",
    )

    with pytest.raises(ValueError, match="must differ"):
        _manifest(expected_corpus_version=_PROPOSED_VERSION)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        _manifest(component_kinds=(ComponentKind("chunks"), ComponentKind("chunks")))


def test_checkpoint_identity_order_and_lifecycle_constraints() -> None:
    checkpoint = BaselineCheckpoint(
        work_key=_work_key(),
        document_id=_DOCUMENT_ID,
        input_identities=(_RECORD_ID, _DOCUMENT_VERSION_ID),
        output_identities=(),
        status=BaselineCheckpointStatus.COMPLETED,
    )
    assert checkpoint.input_identities == (_DOCUMENT_VERSION_ID, _RECORD_ID)
    assert checkpoint.key()[0] == _PROPOSED_VERSION.as_str()

    with pytest.raises(ValueError, match="started checkpoints"):
        BaselineCheckpoint(
            work_key=_work_key(),
            document_id=_DOCUMENT_ID,
            input_identities=(_DOCUMENT_VERSION_ID,),
            output_identities=(_RECORD_ID,),
            status=BaselineCheckpointStatus.STARTED,
        )
    with pytest.raises(ValueError, match="require a failure category"):
        BaselineCheckpoint(
            work_key=_work_key(),
            document_id=_DOCUMENT_ID,
            input_identities=(_DOCUMENT_VERSION_ID,),
            output_identities=(),
            status=BaselineCheckpointStatus.FAILED,
        )
