"""ING-05/STOR-03/PROF-02 acceptance for baseline generation publication."""

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    BaselineChunkGenerationEntry,
    BaselineChunkGenerationManifest,
    BaselineProfilingContext,
    begin_baseline_corpus_update,
    publish_baseline_chunk_generation,
)
from inference_lab.profiling import FailureEvent, StageTiming
from inference_lab.provenance import (
    ConfigurationIdentity,
    ContentHash,
    CorpusDocumentVersionMembership,
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
)
from inference_lab.storage import (
    ChunkerVersionRecord,
    ChunkingPolicyRecord,
    ChunkOversizeUnitPolicy,
    ChunkParserVersionRecord,
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    ChunkStructuralBoundaryPolicy,
    ChunkTokeniserIdentityRecord,
    CommitCorpusUpdateOutcome,
    ComponentGenerationState,
    NoCorpusChangeOutcome,
    StorageConcurrencyError,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CORPUS_ID = CorpusId.from_str("11111111-1111-4111-8111-111111111111")
_DOCUMENT_ID = DocumentId.from_str("22222222-2222-4222-8222-222222222222")
_DOCUMENT_VERSION_ID = DocumentVersionId.from_str(
    "33333333-3333-4333-8333-333333333333"
)
_CONFIGURATION = ConfigurationIdentity(DigestAlgorithm.SHA256, "a" * 64)
_CREATED_AT = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
_PRIVATE_FAILURE = "private-baseline-publication-failure"


class _UncertainCommitConnection(sqlite3.Connection):
    """Commit durably before reporting an ambiguous acknowledgement."""

    def commit(self) -> None:
        super().commit()
        raise sqlite3.OperationalError("private-uncertain-acknowledgement")


class _CollectingSink:
    """Collect shared profiling records without rendering them."""

    def __init__(self) -> None:
        self.records: list[object] = []

    def emit(self, record: object) -> None:
        self.records.append(record)


def _open(path: Path) -> SQLiteMetadataRepository:
    """Open a deterministic test repository."""
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        created_at_factory=lambda: _CREATED_AT,
    )


def _install_trigger(database_path: Path, timing: str) -> None:
    """Install one private SQLite fault trigger for a durable boundary."""
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            f"""CREATE TRIGGER reject_baseline_publication
            {timing}
            BEGIN
                SELECT RAISE(ABORT, '{_PRIVATE_FAILURE}');
            END"""
        )
        connection.commit()
    finally:
        connection.close()


def _remove_trigger(database_path: Path) -> None:
    """Remove the private publication fault trigger."""
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP TRIGGER reject_baseline_publication")
        connection.commit()
    finally:
        connection.close()


def _selection(
    *,
    document_id: DocumentId = _DOCUMENT_ID,
    document_version_id: DocumentVersionId = _DOCUMENT_VERSION_ID,
    payload: bytes = b"canonical-chunk-set",
) -> ChunkSetSelectionRecord:
    """Build a storage-valid exact chunk selection."""
    return ChunkSetSelectionRecord(
        document_id=document_id,
        document_version_id=document_version_id,
        parser_version=ChunkParserVersionRecord(
            implementation="text-native",
            contract_version="3.0.0",
            normalisation_version="1.0.0",
        ),
        chunker_version=ChunkerVersionRecord(
            implementation="structure-aware-greedy",
            contract_version="1.0.1",
        ),
        tokeniser_identity=ChunkTokeniserIdentityRecord(
            implementation="deterministic-tokeniser",
            contract_version="1.0.0",
            artefact_identity="fixture-tokeniser",
            artefact_revision="2026-01",
        ),
        policy=ChunkingPolicyRecord(
            min_tokens=1,
            max_tokens=10,
            overlap_numerator=0,
            overlap_denominator=1,
            structural_boundary_policy=ChunkStructuralBoundaryPolicy.GREEDY,
            oversize_unit_policy=ChunkOversizeUnitPolicy.TOKEN_WINDOW,
        ),
        configuration_identity=_CONFIGURATION,
        payload_digest=OutputDigest(
            DigestAlgorithm.SHA256,
            hashlib.sha256(payload).hexdigest(),
        ),
        canonical_payload=payload,
    )


def _membership(
    *,
    corpus_id: CorpusId = _CORPUS_ID,
    document_id: DocumentId = _DOCUMENT_ID,
    document_version_id: DocumentVersionId = _DOCUMENT_VERSION_ID,
) -> CorpusDocumentVersionMembership:
    """Build the single-corpus membership used by the fixture."""
    return CorpusDocumentVersionMembership(
        corpus_id=corpus_id,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def _manifest(
    corpus_version: CorpusVersionId,
    membership: CorpusDocumentVersionMembership,
    selection: ChunkSetSelectionRecord,
    *,
    corpus_id: CorpusId = _CORPUS_ID,
    generation_id: str,
) -> BaselineChunkGenerationManifest:
    """Build a complete one-entry chunks manifest."""
    return BaselineChunkGenerationManifest(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        configuration_identity=_CONFIGURATION,
        generation_id=generation_id,
        entries=(
            BaselineChunkGenerationEntry(
                membership=membership,
                selection=selection,
            ),
        ),
    )


def _seed_catalogue(
    repository: SQLiteMetadataRepository,
) -> ChunkSetSelectionRecord:
    """Create one corpus-visible document and its immutable chunk set."""
    membership = _membership()
    selection = _selection()
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Acceptance corpus"))
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(
            DocumentVersion(
                _DOCUMENT_VERSION_ID,
                _DOCUMENT_ID,
                ContentHash(HashAlgorithm.SHA256, "b" * 64),
            )
        )
        transaction.add_corpus_document_version_membership(membership)
        transaction.write_chunk_set(
            ChunkSetWriteRequest(corpus_id=_CORPUS_ID, selection=selection)
        )
    return selection


def _publish_first(
    repository: SQLiteMetadataRepository,
    selection: ChunkSetSelectionRecord,
) -> tuple[CorpusVersionId, CorpusDocumentVersionMembership]:
    """Publish the initial version through the ingest publication API."""
    corpus_version = CorpusVersionId.generate()
    membership = _membership()
    manifest = _manifest(
        corpus_version,
        membership,
        selection,
        generation_id="generation-first",
    )
    with repository.transaction() as transaction:
        begin_baseline_corpus_update(
            transaction,
            _CORPUS_ID,
            corpus_version,
            expected_corpus_version=None,
        )
        transaction.add_corpus_update_document_membership(
            _CORPUS_ID,
            corpus_version,
            membership,
        )
        result = publish_baseline_chunk_generation(transaction, repository, manifest)
        assert isinstance(result, CommitCorpusUpdateOutcome)
        assert (
            result.manifest.component_generations[0].state
            is ComponentGenerationState.READY
        )
    return corpus_version, membership


def test_sqlite_publication_profiles_commit_and_no_change_counts(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    try:
        selection = _seed_catalogue(repository)
        sink = _CollectingSink()
        clock_values = iter((0, 10_000_000, 20_000_000, 30_000_000))
        profiling = BaselineProfilingContext.create(
            run_id="run-publication",
            sink=sink,
            clock=lambda: next(clock_values),
            observed_at_factory=lambda: datetime(2026, 9, 18, tzinfo=UTC),
        )

        first_version = CorpusVersionId.generate()
        membership = _membership()
        first_manifest = _manifest(
            first_version,
            membership,
            selection,
            generation_id="generation-first",
        )
        first_result: NoCorpusChangeOutcome | CommitCorpusUpdateOutcome | None = None
        with repository.transaction() as transaction:
            begin_baseline_corpus_update(
                transaction,
                _CORPUS_ID,
                first_version,
                expected_corpus_version=None,
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS_ID,
                first_version,
                membership,
            )
            first_result = publish_baseline_chunk_generation(
                transaction,
                repository,
                first_manifest,
                profiling=profiling,
                parent_stage_id="run-publication:membership:request",
                profiling_identity="first",
            )

        assert isinstance(first_result, CommitCorpusUpdateOutcome)

        second_version = CorpusVersionId.generate()
        second_manifest = _manifest(
            second_version,
            membership,
            selection,
            generation_id="generation-replay",
        )
        second_result: NoCorpusChangeOutcome | CommitCorpusUpdateOutcome | None = None
        with repository.transaction() as transaction:
            begin_baseline_corpus_update(
                transaction,
                _CORPUS_ID,
                second_version,
                expected_corpus_version=first_version,
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS_ID,
                second_version,
                membership,
            )
            second_result = publish_baseline_chunk_generation(
                transaction,
                repository,
                second_manifest,
                profiling=profiling,
                profiling_identity="replay",
            )

        assert isinstance(second_result, NoCorpusChangeOutcome)
        completions = [
            record
            for record in sink.records
            if isinstance(record, StageTiming) and record.event == "stage_completed"
        ]
        assert [
            (
                record.pipeline_stage,
                record.status,
                record.items_processed,
                record.records_emitted,
                record.parent_stage_id,
            )
            for record in completions
        ] == [
            ("publication", "completed", 1, 1, "run-publication:membership:request"),
            ("publication", "completed", 1, 0, None),
        ]
        assert len({record.stage_id for record in completions}) == 2
        assert not [
            record for record in sink.records if isinstance(record, FailureEvent)
        ]
    finally:
        repository.close()


def test_sqlite_publication_replays_exactly_after_cold_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    selection = _seed_catalogue(repository)
    corpus_version, membership = _publish_first(repository, selection)
    repository.close()

    reopened = _open(database_path)
    try:
        manifest = _manifest(
            corpus_version,
            membership,
            selection,
            generation_id="generation-first",
        )
        result: NoCorpusChangeOutcome | CommitCorpusUpdateOutcome | None = None
        with reopened.transaction() as transaction:
            begun = begin_baseline_corpus_update(
                transaction,
                _CORPUS_ID,
                corpus_version,
                expected_corpus_version=None,
            )
            assert begun.replayed is True
            result = publish_baseline_chunk_generation(
                transaction,
                reopened,
                manifest,
            )

        assert isinstance(result, CommitCorpusUpdateOutcome)
        assert result.replayed is True
        assert result.version.sequence == 1
        assert len(reopened.list_corpus_versions(_CORPUS_ID)) == 1
    finally:
        reopened.close()


def test_sqlite_publication_faults_roll_back_and_retry(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    try:
        selection = _seed_catalogue(repository)
        corpus_version = CorpusVersionId.generate()
        membership = _membership()
        manifest = _manifest(
            corpus_version,
            membership,
            selection,
            generation_id="generation-fault-retry",
        )
        with repository.transaction() as transaction:
            begin_baseline_corpus_update(
                transaction,
                _CORPUS_ID,
                corpus_version,
                expected_corpus_version=None,
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS_ID,
                corpus_version,
                membership,
            )

        _install_trigger(database_path, "BEFORE INSERT ON corpus_versions")
        with pytest.raises(StorageConflictError) as raised:
            with repository.transaction() as transaction:
                publish_baseline_chunk_generation(
                    transaction,
                    repository,
                    manifest,
                )
        assert _PRIVATE_FAILURE not in str(raised.value)
        update = repository.get_corpus_update(_CORPUS_ID, corpus_version)
        assert update is not None
        assert update.state.value == "preparing"
        assert repository.get_active_corpus_version(_CORPUS_ID) is None

        _remove_trigger(database_path)
        result: NoCorpusChangeOutcome | CommitCorpusUpdateOutcome | None = None
        with repository.transaction() as transaction:
            result = publish_baseline_chunk_generation(
                transaction,
                repository,
                manifest,
            )
        assert isinstance(result, CommitCorpusUpdateOutcome)
        assert result.replayed is False
    finally:
        repository.close()


def test_sqlite_publication_replays_after_ambiguous_commit_acknowledgement(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    selection = _seed_catalogue(repository)
    parent_version, membership = _publish_first(repository, selection)
    changed_document_id = _DOCUMENT_ID
    changed_document_version_id = DocumentVersionId.generate()
    changed_membership = _membership(
        document_id=changed_document_id,
        document_version_id=changed_document_version_id,
    )
    changed_selection = _selection(
        document_id=changed_document_id,
        document_version_id=changed_document_version_id,
        payload=b"uncertain-acknowledgement-chunk-set",
    )
    with repository.transaction() as transaction:
        transaction.add_document(DocumentIdentity(changed_document_id))
        transaction.add_document_version(
            DocumentVersion(
                changed_document_version_id,
                changed_document_id,
                ContentHash(HashAlgorithm.SHA256, "d" * 64),
            )
        )
        transaction.add_corpus_document_version_membership(changed_membership)
        transaction.write_chunk_set(
            ChunkSetWriteRequest(
                corpus_id=_CORPUS_ID,
                selection=changed_selection,
            )
        )
    proposed_version = CorpusVersionId.generate()
    manifest = _manifest(
        proposed_version,
        changed_membership,
        changed_selection,
        generation_id="generation-uncertain-ack",
    )
    with repository.transaction() as transaction:
        begin_baseline_corpus_update(
            transaction,
            _CORPUS_ID,
            proposed_version,
            expected_corpus_version=parent_version,
        )
        transaction.replace_corpus_update_document_membership(
            _CORPUS_ID,
            proposed_version,
            membership,
            changed_membership,
        )
    repository.close()

    connection = sqlite3.connect(
        database_path,
        isolation_level=None,
        factory=_UncertainCommitConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    uncertain = SQLiteMetadataRepository(
        connection,
        MIGRATIONS,
        created_at_factory=lambda: _CREATED_AT,
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageTransactionError) as raised:
                with uncertain.transaction() as transaction:
                    publish_baseline_chunk_generation(
                        transaction,
                        uncertain,
                        manifest,
                    )

        assert "private-uncertain-acknowledgement" not in str(raised.value)
        assert [log for log in logs if log["event"] == "transaction_failed"] == [
            {
                "event": "transaction_failed",
                "pipeline_stage": "storage",
                "error_category": "transaction",
                "retryable": False,
                "log_level": "error",
            }
        ]
        committed = uncertain.get_active_corpus_version(_CORPUS_ID)
        assert committed is not None
        assert committed.corpus_version == proposed_version
        assert committed.sequence == 2
    finally:
        uncertain.close()

    reopened = _open(database_path)
    try:
        replayed: NoCorpusChangeOutcome | CommitCorpusUpdateOutcome | None = None
        with reopened.transaction() as transaction:
            begun = begin_baseline_corpus_update(
                transaction,
                _CORPUS_ID,
                proposed_version,
                expected_corpus_version=parent_version,
            )
            assert begun.replayed is True
            replayed = publish_baseline_chunk_generation(
                transaction,
                reopened,
                manifest,
            )

        assert isinstance(replayed, CommitCorpusUpdateOutcome)
        assert replayed.replayed is True
        assert replayed.version.sequence == 2
        assert (
            replayed.manifest.component_generations[0].generation_id
            == manifest.generation_id
        )
        versions = reopened.list_corpus_versions(_CORPUS_ID)
        assert len(versions) == 2
        assert versions[0].corpus_version == parent_version
        assert versions[1] == replayed.version
    finally:
        reopened.close()


def test_sqlite_baseline_publication_serialises_same_parent_race(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    selection = _seed_catalogue(repository)
    parent_version, membership = _publish_first(repository, selection)
    winning_document_id = _DOCUMENT_ID
    winning_document_version_id = DocumentVersionId.generate()
    winning_membership = _membership(
        document_id=winning_document_id,
        document_version_id=winning_document_version_id,
    )
    winning_selection = _selection(
        document_id=winning_document_id,
        document_version_id=winning_document_version_id,
        payload=b"winning-race-chunk-set",
    )
    losing_document_id = _DOCUMENT_ID
    losing_document_version_id = DocumentVersionId.generate()
    losing_membership = _membership(
        document_id=losing_document_id,
        document_version_id=losing_document_version_id,
    )
    losing_selection = _selection(
        document_id=losing_document_id,
        document_version_id=losing_document_version_id,
        payload=b"losing-race-chunk-set",
    )
    with repository.transaction() as transaction:
        for (
            document_id,
            document_version_id,
            content_hash,
            new_membership,
            new_selection,
        ) in (
            (
                winning_document_id,
                winning_document_version_id,
                "e" * 64,
                winning_membership,
                winning_selection,
            ),
            (
                losing_document_id,
                losing_document_version_id,
                "f" * 64,
                losing_membership,
                losing_selection,
            ),
        ):
            transaction.add_document(DocumentIdentity(document_id))
            transaction.add_document_version(
                DocumentVersion(
                    document_version_id,
                    document_id,
                    ContentHash(HashAlgorithm.SHA256, content_hash),
                )
            )
            transaction.add_corpus_document_version_membership(new_membership)
            transaction.write_chunk_set(
                ChunkSetWriteRequest(
                    corpus_id=_CORPUS_ID,
                    selection=new_selection,
                )
            )
    winning_version = CorpusVersionId.generate()
    losing_version = CorpusVersionId.generate()
    winning_manifest = _manifest(
        winning_version,
        winning_membership,
        winning_selection,
        generation_id="generation-winning-race",
    )
    losing_manifest = _manifest(
        losing_version,
        losing_membership,
        losing_selection,
        generation_id="generation-losing-race",
    )
    for proposed_version in (winning_version, losing_version):
        with repository.transaction() as transaction:
            begin_baseline_corpus_update(
                transaction,
                _CORPUS_ID,
                proposed_version,
                expected_corpus_version=parent_version,
            )
            transaction.replace_corpus_update_document_membership(
                _CORPUS_ID,
                proposed_version,
                membership,
                winning_membership
                if proposed_version == winning_version
                else losing_membership,
            )
    repository.close()

    winner = _open(database_path)
    contender_ready = Event()
    start_contender = Event()
    first_attempt_finished = Event()
    winner_committed = Event()

    def contend() -> tuple[StorageConcurrencyError, StorageConflictError]:
        contender = SQLiteMetadataRepository.open(
            database_path,
            busy_timeout_ms=25,
            created_at_factory=lambda: _CREATED_AT,
        )
        lock_error: StorageConcurrencyError | None = None
        stale_error: StorageConflictError | None = None
        try:
            contender_ready.set()
            if not start_contender.wait(timeout=5):
                raise AssertionError("winner did not start the contender")
            try:
                with contender.transaction() as transaction:
                    publish_baseline_chunk_generation(
                        transaction,
                        contender,
                        losing_manifest,
                    )
            except StorageConcurrencyError as error:
                lock_error = error
            finally:
                first_attempt_finished.set()

            if not winner_committed.wait(timeout=5):
                raise AssertionError("winner did not finish its publication")
            try:
                with contender.transaction() as transaction:
                    publish_baseline_chunk_generation(
                        transaction,
                        contender,
                        losing_manifest,
                    )
            except StorageConflictError as error:
                stale_error = error
        finally:
            contender.close()

        assert lock_error is not None
        assert stale_error is not None
        return lock_error, stale_error

    winning_result: NoCorpusChangeOutcome | CommitCorpusUpdateOutcome | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(contend)
            assert contender_ready.wait(timeout=5)
            try:
                with winner.transaction() as transaction:
                    start_contender.set()
                    assert first_attempt_finished.wait(timeout=5)
                    winning_result = publish_baseline_chunk_generation(
                        transaction,
                        winner,
                        winning_manifest,
                    )
            finally:
                winner_committed.set()
            lock_error, stale_error = future.result(timeout=5)

        assert winning_result is not None
        assert isinstance(winning_result, CommitCorpusUpdateOutcome)
        assert winning_result.replayed is False
        assert winning_result.version.sequence == 2
        assert winning_result.version.parent_corpus_version == parent_version
        assert lock_error.retryable is True
        assert stale_error.retryable is False
        active = winner.get_active_corpus_version(_CORPUS_ID)
        assert active == winning_result.version
        versions = winner.list_corpus_versions(_CORPUS_ID)
        assert len(versions) == 2
        assert versions[0].corpus_version == parent_version
        assert versions[1] == winning_result.version
    finally:
        winner.close()


def test_sqlite_baseline_publication_isolates_second_corpus_and_private_canaries(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    selection = _seed_catalogue(repository)
    first_version, first_membership = _publish_first(repository, selection)

    second_corpus_id = CorpusId.from_str("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    second_document_id = DocumentId.generate()
    second_document_version_id = DocumentVersionId.generate()
    second_membership = _membership(
        corpus_id=second_corpus_id,
        document_id=second_document_id,
        document_version_id=second_document_version_id,
    )
    private_payload = b"private-second-corpus-canary"
    second_selection = _selection(
        document_id=second_document_id,
        document_version_id=second_document_version_id,
        payload=private_payload,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(
            CorpusMetadata(second_corpus_id, "Second private corpus")
        )
        transaction.add_document(DocumentIdentity(second_document_id))
        transaction.add_document_version(
            DocumentVersion(
                second_document_version_id,
                second_document_id,
                ContentHash(HashAlgorithm.SHA256, "e" * 64),
            )
        )
        transaction.add_corpus_document_version_membership(second_membership)
        transaction.write_chunk_set(
            ChunkSetWriteRequest(
                corpus_id=second_corpus_id,
                selection=second_selection,
            )
        )

    second_version = CorpusVersionId.generate()
    second_manifest = _manifest(
        second_version,
        second_membership,
        second_selection,
        corpus_id=second_corpus_id,
        generation_id="generation-second-corpus",
    )
    result: NoCorpusChangeOutcome | CommitCorpusUpdateOutcome | None = None
    with capture_logs() as logs:
        with repository.transaction() as transaction:
            begin_baseline_corpus_update(
                transaction,
                second_corpus_id,
                second_version,
                expected_corpus_version=None,
            )
            transaction.add_corpus_update_document_membership(
                second_corpus_id,
                second_version,
                second_membership,
            )
            result = publish_baseline_chunk_generation(
                transaction,
                repository,
                second_manifest,
            )

    assert isinstance(result, CommitCorpusUpdateOutcome)
    assert result.version.sequence == 1
    assert "Second private corpus" not in repr(logs)
    assert private_payload.decode() not in repr(logs)

    first_active = repository.get_active_corpus_version(_CORPUS_ID)
    assert first_active is not None
    assert first_active.corpus_version == first_version
    assert repository.get_active_corpus_version(second_corpus_id) == result.version
    assert (
        repository.get_chunk_set_selection(
            _CORPUS_ID,
            second_document_version_id,
            _CONFIGURATION,
        )
        is None
    )
    assert (
        repository.get_chunk_set_selection(
            second_corpus_id,
            _DOCUMENT_VERSION_ID,
            _CONFIGURATION,
        )
        is None
    )

    with repository.open_active_corpus_snapshot(_CORPUS_ID) as first_snapshot:
        assert first_snapshot.manifest.document_memberships == (first_membership,)
        assert (
            first_snapshot.get_chunk_set_selection(
                second_document_version_id,
                _CONFIGURATION,
            )
            is None
        )
    with repository.open_active_corpus_snapshot(second_corpus_id) as second_snapshot:
        assert second_snapshot.manifest.document_memberships == (second_membership,)
        assert (
            second_snapshot.get_chunk_set_selection(
                second_document_version_id,
                _CONFIGURATION,
            )
            == second_selection
        )
        assert (
            second_snapshot.get_chunk_set_selection(
                _DOCUMENT_VERSION_ID,
                _CONFIGURATION,
            )
            is None
        )
    repository.close()


def test_sqlite_open_snapshot_remains_pinned_across_next_publication(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    try:
        selection = _seed_catalogue(repository)
        first_version, membership = _publish_first(repository, selection)
        changed_document_version = DocumentVersionId.generate()
        changed_membership = _membership(
            document_version_id=changed_document_version,
        )
        changed_selection = _selection(
            document_version_id=changed_document_version,
            payload=b"next-canonical-chunk-set",
        )
        with repository.transaction() as transaction:
            transaction.add_document_version(
                DocumentVersion(
                    changed_document_version,
                    _DOCUMENT_ID,
                    ContentHash(HashAlgorithm.SHA256, "c" * 64),
                )
            )
            transaction.add_corpus_document_version_membership(changed_membership)
            transaction.write_chunk_set(
                ChunkSetWriteRequest(
                    corpus_id=_CORPUS_ID,
                    selection=changed_selection,
                )
            )

        next_version = CorpusVersionId.generate()
        next_manifest = _manifest(
            next_version,
            changed_membership,
            changed_selection,
            generation_id="generation-next",
        )
        with repository.open_active_corpus_snapshot(_CORPUS_ID) as snapshot:
            assert snapshot.version.corpus_version == first_version
            with repository.transaction() as transaction:
                begin_baseline_corpus_update(
                    transaction,
                    _CORPUS_ID,
                    next_version,
                    expected_corpus_version=first_version,
                )
                transaction.replace_corpus_update_document_membership(
                    _CORPUS_ID,
                    next_version,
                    membership,
                    changed_membership,
                )
                result = publish_baseline_chunk_generation(
                    transaction,
                    repository,
                    next_manifest,
                )
                assert isinstance(result, CommitCorpusUpdateOutcome)

            assert snapshot.version.corpus_version == first_version
            assert snapshot.manifest.document_memberships == (membership,)

        active = repository.get_active_corpus_version(_CORPUS_ID)
        assert active is not None
        assert active.corpus_version == next_version
    finally:
        repository.close()
