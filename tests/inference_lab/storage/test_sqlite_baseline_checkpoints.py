"""Acceptance tests for SQLite baseline checkpoint persistence."""

from pathlib import Path

import pytest

from inference_lab.provenance import (
    ConfigurationIdentity,
    ContentHash,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    OutputDigest,
    RecordId,
)
from inference_lab.storage import (
    BaselineCheckpointIdentityRecord,
    BaselineCheckpointReader,
    BaselineCheckpointRecord,
    BaselineCheckpointRecordIdentityKind,
    BaselineCheckpointRecordStage,
    BaselineCheckpointRecordStatus,
    BaselineCheckpointRepository,
    BaselineManifestRecord,
    BaselineManifestRecordState,
    BeginBaselineManifestOutcome,
    ComponentKind,
    StorageConflictError,
    TransitionBaselineManifestOutcome,
    WriteBaselineCheckpointOutcome,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import open_repository as _open_repository  # noqa: TID252

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_PROPOSED_VERSION = CorpusVersionId("22222222-2222-4222-8222-222222222222")
_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_DOCUMENT_VERSION_ID = DocumentVersionId("44444444-4444-4444-8444-444444444444")
_RECORD_ID = RecordId("55555555-5555-4555-8555-555555555555")
_CONFIGURATION = ConfigurationIdentity(DigestAlgorithm.SHA256, "a" * 64)
_DISCOVERY_DIGEST = OutputDigest(DigestAlgorithm.SHA256, "b" * 64)
_DOCUMENT_VERSION = DocumentVersion(
    _DOCUMENT_VERSION_ID,
    _DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "c" * 64),
)


def _manifest(**overrides: object) -> BaselineManifestRecord:
    values: dict[str, object] = {
        "corpus_id": _CORPUS_ID,
        "proposed_corpus_version": _PROPOSED_VERSION,
        "expected_corpus_version": None,
        "configuration_identity": _CONFIGURATION,
        "required_component_kinds": (ComponentKind("chunks"),),
        "discovery_summary_identity": _DISCOVERY_DIGEST,
        "state": BaselineManifestRecordState.PREPARING,
    }
    values.update(overrides)
    return BaselineManifestRecord(**values)  # type: ignore[arg-type]


def _checkpoint(**overrides: object) -> BaselineCheckpointRecord:
    values: dict[str, object] = {
        "corpus_id": _CORPUS_ID,
        "proposed_corpus_version": _PROPOSED_VERSION,
        "document_version_id": _DOCUMENT_VERSION_ID,
        "document_id": _DOCUMENT_ID,
        "stage": BaselineCheckpointRecordStage.PARSE,
        "configuration_identity": _CONFIGURATION,
        "input_identities": (
            BaselineCheckpointIdentityRecord(
                kind=BaselineCheckpointRecordIdentityKind.DOCUMENT_VERSION,
                value=_DOCUMENT_VERSION_ID.as_str(),
            ),
        ),
        "output_identities": (),
        "status": BaselineCheckpointRecordStatus.STARTED,
        "failure_category": None,
    }
    values.update(overrides)
    return BaselineCheckpointRecord(**values)  # type: ignore[arg-type]


def _prepare_repository(path: Path) -> SQLiteMetadataRepository:
    repository = _open_repository(path)
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Checkpoint corpus"))
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(_DOCUMENT_VERSION)
        transaction.begin_corpus_update(
            _CORPUS_ID,
            _PROPOSED_VERSION,
            expected_corpus_version=None,
            required_component_kinds=(ComponentKind("chunks"),),
        )
        transaction.begin_baseline_manifest(_manifest())
    return repository


def test_repository_exposes_checkpoint_reader_and_transaction_writer(
    tmp_path: Path,
) -> None:
    repository = _prepare_repository(tmp_path / "metadata.sqlite3")
    try:
        assert isinstance(repository, BaselineCheckpointReader)
        with repository.transaction() as transaction:
            assert isinstance(transaction, BaselineCheckpointRepository)
    finally:
        repository.close()


def test_manifest_and_checkpoint_round_trip_with_exact_replay(
    tmp_path: Path,
) -> None:
    repository = _prepare_repository(tmp_path / "metadata.sqlite3")
    checkpoint = _checkpoint()
    manifest_replay: BeginBaselineManifestOutcome | None = None
    checkpoint_outcome: WriteBaselineCheckpointOutcome | None = None
    checkpoint_replay: WriteBaselineCheckpointOutcome | None = None
    try:
        with repository.transaction() as transaction:
            manifest_replay = transaction.begin_baseline_manifest(_manifest())
            checkpoint_outcome = transaction.write_baseline_checkpoint(
                checkpoint,
                expected_status=None,
            )
            checkpoint_replay = transaction.write_baseline_checkpoint(
                checkpoint,
                expected_status=None,
            )

        assert manifest_replay is not None
        assert checkpoint_outcome is not None
        assert checkpoint_replay is not None
        assert manifest_replay.replayed is True
        assert checkpoint_outcome.replayed is False
        assert checkpoint_replay.replayed is True
        assert repository.get_baseline_manifest(_CORPUS_ID, _PROPOSED_VERSION) == (
            _manifest()
        )
        assert (
            repository.get_baseline_checkpoint(
                _CORPUS_ID,
                _PROPOSED_VERSION,
                _DOCUMENT_VERSION_ID,
                BaselineCheckpointRecordStage.PARSE,
                _CONFIGURATION,
            )
            == checkpoint
        )
        assert repository.list_baseline_checkpoints(
            _CORPUS_ID,
            _PROPOSED_VERSION,
        ) == (checkpoint,)
    finally:
        repository.close()


def test_checkpoint_compare_and_set_and_manifest_transition_are_idempotent(
    tmp_path: Path,
) -> None:
    repository = _prepare_repository(tmp_path / "metadata.sqlite3")
    started = _checkpoint()
    completed = _checkpoint(
        status=BaselineCheckpointRecordStatus.COMPLETED,
        output_identities=(
            BaselineCheckpointIdentityRecord(
                kind=BaselineCheckpointRecordIdentityKind.RECORD,
                value=_RECORD_ID.as_str(),
            ),
        ),
    )
    transitioned: WriteBaselineCheckpointOutcome | None = None
    replayed: WriteBaselineCheckpointOutcome | None = None
    manifest_transition: TransitionBaselineManifestOutcome | None = None
    manifest_replay: TransitionBaselineManifestOutcome | None = None
    try:
        with repository.transaction() as transaction:
            transaction.write_baseline_checkpoint(
                started,
                expected_status=None,
            )
            transitioned = transaction.write_baseline_checkpoint(
                completed,
                expected_status=BaselineCheckpointRecordStatus.STARTED,
            )
            replayed = transaction.write_baseline_checkpoint(
                completed,
                expected_status=BaselineCheckpointRecordStatus.STARTED,
            )
            manifest_transition = transaction.transition_baseline_manifest(
                _CORPUS_ID,
                _PROPOSED_VERSION,
                expected_state=BaselineManifestRecordState.PREPARING,
                next_state=BaselineManifestRecordState.READY,
            )

        assert transitioned is not None
        assert replayed is not None
        assert manifest_transition is not None
        assert transitioned.replayed is False
        assert replayed.replayed is True
        assert manifest_transition.manifest.state is BaselineManifestRecordState.READY

        with repository.transaction() as transaction:
            manifest_replay = transaction.transition_baseline_manifest(
                _CORPUS_ID,
                _PROPOSED_VERSION,
                expected_state=BaselineManifestRecordState.PREPARING,
                next_state=BaselineManifestRecordState.READY,
            )
        assert manifest_replay is not None
        assert manifest_replay.replayed is True
    finally:
        repository.close()


def test_checkpoint_and_manifest_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    repository = _prepare_repository(tmp_path / "metadata.sqlite3")
    try:
        with repository.transaction() as transaction:
            transaction.begin_baseline_manifest(_manifest())
            transaction.write_baseline_checkpoint(
                _checkpoint(),
                expected_status=None,
            )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.begin_baseline_manifest(
                    _manifest(
                        discovery_summary_identity=OutputDigest(
                            DigestAlgorithm.SHA256,
                            "d" * 64,
                        )
                    )
                )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.transition_baseline_manifest(
                    _CORPUS_ID,
                    _PROPOSED_VERSION,
                    expected_state=BaselineManifestRecordState.READY,
                    next_state=BaselineManifestRecordState.COMMITTED,
                )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.write_baseline_checkpoint(
                    _checkpoint(status=BaselineCheckpointRecordStatus.COMPLETED),
                    expected_status=BaselineCheckpointRecordStatus.COMPLETED,
                )
    finally:
        repository.close()
