"""Tests for replay-aware baseline checkpoint storage outcomes."""

import pytest

from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
)
from inference_lab.storage import (
    BaselineCheckpointRecord,
    BaselineCheckpointRecordStage,
    BaselineCheckpointRecordStatus,
    BaselineManifestRecord,
    BaselineManifestRecordState,
    BeginBaselineManifestOutcome,
    TransitionBaselineManifestOutcome,
    WriteBaselineCheckpointOutcome,
)

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_PROPOSED_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_DOCUMENT_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")
_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_CONFIGURATION = ConfigurationIdentity(
    algorithm=DigestAlgorithm.SHA256,
    digest="0" * 64,
)


def _manifest(
    state: BaselineManifestRecordState = BaselineManifestRecordState.PREPARING,
) -> BaselineManifestRecord:
    return BaselineManifestRecord(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=None,
        configuration_identity=_CONFIGURATION,
        required_component_kinds=(),
        discovery_summary_identity=OutputDigest(
            DigestAlgorithm.SHA256,
            "1" * 64,
        ),
        state=state,
    )


def _checkpoint() -> BaselineCheckpointRecord:
    return BaselineCheckpointRecord(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        document_version_id=_DOCUMENT_VERSION_ID,
        stage=BaselineCheckpointRecordStage.PARSE,
        configuration_identity=_CONFIGURATION,
        document_id=_DOCUMENT_ID,
        input_identities=(),
        output_identities=(),
        status=BaselineCheckpointRecordStatus.STARTED,
    )


def test_begin_manifest_outcome_requires_preparing_for_new_write() -> None:
    outcome = BeginBaselineManifestOutcome(_manifest())

    assert outcome.replayed is False

    with pytest.raises(ValueError, match="preparing"):
        BeginBaselineManifestOutcome(_manifest(BaselineManifestRecordState.READY))


def test_begin_manifest_outcome_allows_terminal_exact_replay() -> None:
    outcome = BeginBaselineManifestOutcome(
        _manifest(BaselineManifestRecordState.COMMITTED),
        replayed=True,
    )

    assert outcome.replayed is True


def test_transition_manifest_outcome_allows_non_preparing_state() -> None:
    outcome = TransitionBaselineManifestOutcome(
        _manifest(BaselineManifestRecordState.READY)
    )

    assert outcome.manifest.state is BaselineManifestRecordState.READY


@pytest.mark.parametrize(
    "outcome_type",
    [
        BeginBaselineManifestOutcome,
        TransitionBaselineManifestOutcome,
    ],
)
def test_manifest_outcomes_reject_wrong_types(outcome_type: type[object]) -> None:
    with pytest.raises(TypeError):
        outcome_type(object())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        outcome_type(_manifest(), replayed=1)  # type: ignore[call-arg]


def test_checkpoint_outcome_validates_record_and_replayed_flag() -> None:
    outcome = WriteBaselineCheckpointOutcome(_checkpoint(), replayed=True)

    assert outcome.replayed is True

    with pytest.raises(TypeError):
        WriteBaselineCheckpointOutcome(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        WriteBaselineCheckpointOutcome(_checkpoint(), replayed=1)  # type: ignore[arg-type]
