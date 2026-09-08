"""Tests for per-source identity resolution and persistence coordination."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

import inference_lab.ingest.identity_persistence as persistence_module
from inference_lab.ingest import (
    ClassifiedSourceCandidate,
    DiscoveredSource,
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
    DiscoveryIdentityIdAllocator,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    IdentityApplicationResult,
    IdentityPersistenceUnit,
    TraversalInventory,
    TraversalRootScope,
    UuidDiscoveryIdentityIdAllocator,
    VerifiedSourceSnapshot,
    persist_discovered_source_identities,
    persist_discovered_source_identity,
)
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
    IdentityIdCollisionError,
    ResolutionKind,
    SourceObservationState,
)
from inference_lab.storage import (
    MetadataTransaction,
    RepositoryUnavailableError,
    SourceLocatorRevision,
    StorageConcurrencyError,
    StorageConflictError,
)
from inference_lab.storage import _sqlite_transaction as sqlite_transaction_module
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.ingest.identity_persistence_fixtures import (
    _CORPUS_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
    _HASH,
    _OTHER_HASH,
    _SECOND_DOCUMENT_ID,
    _SECOND_SOURCE_ID,
    _SECOND_VERSION_ID,
    _SOURCE_ID,
    _THIRD_HASH,
    _AlternateCountingAllocator,
    _case,
    _CollisionThenUniqueAllocator,
    _CountingAllocator,
    _persist,
    _persistence_unit,
    _RecordingAllocator,
    _Repository,
    _resolved,
    _Transaction,
)


def test_conflict_retry_rebuilds_resolution_in_a_new_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    first = _resolved(source, snapshot, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID)
    rebuilt = _resolved(source, snapshot, _SECOND_DOCUMENT_ID, _SECOND_VERSION_ID)
    expected = IdentityApplicationResult(rebuilt, created_document_version=True)
    transactions = (_Transaction(), _Transaction())
    repository = _Repository(*transactions)
    contexts: list[dict[str, object]] = []

    def resolve_again(*args: object, **kwargs: object) -> DiscoveredSource:
        del args, kwargs
        contexts.append(get_contextvars())
        return (first, rebuilt)[len(contexts) - 1]

    resolver = Mock(side_effect=resolve_again)
    primary = StorageConflictError()
    application = Mock(side_effect=(primary, expected))
    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        resolver,
    )
    monkeypatch.setattr(
        persistence_module,
        "apply_discovery_identity_outcome",
        application,
    )
    with capture_logs() as logs:
        result = _persist(source, snapshot, request, inventory, repository)

    assert result == expected
    assert repository.calls == 2
    assert resolver.call_count == 2
    assert application.call_args_list[0].args == (first,)
    assert application.call_args_list[1].args == (rebuilt,)
    assert transactions[0].exit_exception is primary
    assert transactions[1].exit_exception is None
    assert (
        contexts
        == [
            {
                "corpus_id": _CORPUS_ID.as_str(),
                "corpus_version": None,
                "document_id": None,
            }
        ]
        * 2
    )
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("identity_persistence_retry", "warning"),
        ("identity_persistence_completed", "info"),
    ]
    assert logs[0]["error_category"] == "conflict"
    assert logs[0]["retry_number"] == 1
    assert logs[1]["attempt_count"] == 2


def test_allocator_collision_retries_resolution_before_any_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    rebuilt = _resolved(source, snapshot, _SECOND_DOCUMENT_ID, _SECOND_VERSION_ID)
    expected = IdentityApplicationResult(rebuilt, created_document_version=True)
    collision = IdentityIdCollisionError("private-allocator-payload")
    transactions = (_Transaction(), _Transaction())
    repository = _Repository(*transactions)
    resolver = Mock(side_effect=(collision, rebuilt))
    application = Mock(return_value=expected)
    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        resolver,
    )
    monkeypatch.setattr(
        persistence_module,
        "apply_discovery_identity_outcome",
        application,
    )
    with capture_logs() as logs:
        result = _persist(source, snapshot, request, inventory, repository)

    assert result == expected
    assert repository.calls == resolver.call_count == 2
    application.assert_called_once_with(
        rebuilt,
        corpus_id=_CORPUS_ID,
        transaction=transactions[1],
    )
    assert transactions[0].exit_exception is collision
    assert transactions[1].exit_exception is None
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("identity_persistence_retry", "warning"),
        ("identity_persistence_completed", "info"),
    ]
    assert logs[0]["error_category"] == "identity_write_failed"
    assert logs[1]["attempt_count"] == 2
    assert "private-allocator-payload" not in repr(logs)


def test_real_observation_id_collision_recovers_with_fresh_allocation(
    tmp_path: Path,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    allocator = _CollisionThenUniqueAllocator()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    prior_version = DocumentVersion(
        _FIRST_VERSION_ID,
        _FIRST_DOCUMENT_ID,
        _OTHER_HASH,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Test corpus"))
        transaction.add_document(DocumentIdentity(_FIRST_DOCUMENT_ID))
        transaction.add_document_version(prior_version)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS_ID,
                _FIRST_DOCUMENT_ID,
                _FIRST_VERSION_ID,
            )
        )
        transaction.add_source_observation(
            SourceObservationState(
                _CORPUS_ID,
                _SOURCE_ID,
                1,
                _FIRST_DOCUMENT_ID,
                _FIRST_VERSION_ID,
            )
        )
    try:
        result = persist_discovered_source_identity(
            source,
            snapshot,
            request=request,
            inventory=inventory,
            repository=repository,
            allocator=allocator,
        )

        assert isinstance(result, IdentityApplicationResult)
        assert result.created_document_version
        assert result.source.source_observation_id == _SECOND_SOURCE_ID
        assert result.source.identity_outcome.document == DocumentIdentity(
            _SECOND_DOCUMENT_ID
        )
        assert result.source.identity_outcome.version == DocumentVersion(
            _SECOND_VERSION_ID,
            _SECOND_DOCUMENT_ID,
            _HASH,
        )
        assert allocator.counts() == (2, 1, 1)
        assert repository.get_source_observation(
            _CORPUS_ID,
            _SOURCE_ID,
        ) == SourceObservationState(
            _CORPUS_ID,
            _SOURCE_ID,
            1,
            _FIRST_DOCUMENT_ID,
            _FIRST_VERSION_ID,
        )
        assert (
            repository.get_source_observation(
                _CORPUS_ID,
                _SECOND_SOURCE_ID,
            )
            is not None
        )
    finally:
        repository.close()


def test_repeated_allocator_collision_returns_bounded_identity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    collisions = (IdentityIdCollisionError(), IdentityIdCollisionError())
    transactions = (_Transaction(), _Transaction())
    repository = _Repository(*transactions)
    resolver = Mock(side_effect=collisions)
    application = Mock()
    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        resolver,
    )
    monkeypatch.setattr(
        persistence_module,
        "apply_discovery_identity_outcome",
        application,
    )

    result = _persist(source, snapshot, request, inventory, repository)

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.IDENTITY_WRITE_FAILED
    assert not result.retryable
    assert repository.calls == resolver.call_count == 2
    application.assert_not_called()
    assert transactions[0].exit_exception is collisions[0]
    assert transactions[1].exit_exception is collisions[1]


def test_concurrent_discovery_rebuilds_to_one_unchanged_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    database_path = tmp_path / "metadata.sqlite3"
    initial = SQLiteMetadataRepository.open(database_path, apply_migrations=True)
    with initial.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Test corpus"))
    initial.close()

    holder_allocator = _CountingAllocator()
    contender_allocator = _AlternateCountingAllocator()
    holder_entered = Event()
    release_holder = Event()
    holder_done = Event()
    retry_calls = 0
    original_resolver = persistence_module.resolve_discovered_source_identity
    original_log_retry = persistence_module._log_retry

    def gated_resolver(
        candidate: ClassifiedSourceCandidate,
        verified: VerifiedSourceSnapshot,
        *,
        request: DiscoveryRequest,
        inventory: TraversalInventory,
        transaction: MetadataTransaction,
        allocator: DiscoveryIdentityIdAllocator,
    ) -> DiscoveredSource | DiscoveryFailure:
        if allocator is holder_allocator:
            holder_entered.set()
            if not release_holder.wait(timeout=5):
                raise AssertionError("concurrent discovery holder was not released")
        return original_resolver(
            candidate,
            verified,
            request=request,
            inventory=inventory,
            transaction=transaction,
            allocator=allocator,
        )

    def release_after_retry(
        error: IdentityIdCollisionError
        | StorageConcurrencyError
        | StorageConflictError,
    ) -> None:
        nonlocal retry_calls
        assert isinstance(error, StorageConcurrencyError)
        retry_calls += 1
        original_log_retry(error)
        release_holder.set()
        if not holder_done.wait(timeout=5):
            raise AssertionError("concurrent discovery holder did not complete")

    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        gated_resolver,
    )
    monkeypatch.setattr(persistence_module, "_log_retry", release_after_retry)

    def discover(
        allocator: DiscoveryIdentityIdAllocator,
        *,
        holder: bool,
    ) -> IdentityApplicationResult | DiscoveryFailure:
        repository = SQLiteMetadataRepository.open(
            database_path,
            busy_timeout_ms=25,
        )
        try:
            return persist_discovered_source_identity(
                source,
                snapshot,
                request=request,
                inventory=inventory,
                repository=repository,
                allocator=allocator,
            )
        finally:
            repository.close()
            if holder:
                holder_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        holder_future = executor.submit(discover, holder_allocator, holder=True)
        assert holder_entered.wait(timeout=5)
        contender_future = executor.submit(
            discover,
            contender_allocator,
            holder=False,
        )
        holder_result = holder_future.result(timeout=5)
        contender_result = contender_future.result(timeout=5)

    assert isinstance(holder_result, IdentityApplicationResult)
    assert holder_result.created_document_version
    assert holder_result.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
    assert isinstance(contender_result, IdentityApplicationResult)
    assert not contender_result.created_document_version
    assert contender_result.source.identity_outcome.kind is ResolutionKind.UNCHANGED
    assert contender_result.source.source_observation_id == _SOURCE_ID
    assert (
        contender_result.source.identity_outcome.document
        == holder_result.source.identity_outcome.document
    )
    assert (
        contender_result.source.identity_outcome.version
        == holder_result.source.identity_outcome.version
    )
    assert retry_calls == 1
    assert holder_allocator.counts() == (1, 1, 1)
    assert contender_allocator.counts() == (0, 0, 0)

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert len(reopened.list_document_versions(_FIRST_DOCUMENT_ID)) == 1
        assert len(reopened.list_corpus_document_version_memberships(_CORPUS_ID)) == 1
        assert (
            len(reopened.list_source_observation_history(_CORPUS_ID, _SOURCE_ID)) == 1
        )
        assert len(reopened.list_source_locator_history(_CORPUS_ID, _SOURCE_ID)) == 1
        assert (
            reopened.list_duplicate_relationships(
                _CORPUS_ID,
                _FIRST_VERSION_ID,
            )
            == ()
        )
    finally:
        reopened.close()


def test_identity_batch_rolls_back_one_failure_and_continues_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "sources").resolve()
    units = (
        _persistence_unit(root, ordinal=0, filename="first.txt", content_hash=_HASH),
        _persistence_unit(
            root,
            ordinal=1,
            filename="failing.txt",
            content_hash=_OTHER_HASH,
        ),
        _persistence_unit(
            root,
            ordinal=2,
            filename="third.txt",
            content_hash=_THIRD_HASH,
        ),
    )
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(DiscoveryInput(root, DiscoveryInputKind.DIRECTORY),),
    )
    inventory = TraversalInventory(
        corpus_id=request.corpus_id,
        root_count=1,
        candidates=tuple(unit.source.candidate for unit in units),
        failures=(),
        complete_root_ordinals=(0,),
        root_scopes=(TraversalRootScope(0, root, DiscoveryInputKind.DIRECTORY),),
    )
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Test corpus"))

    allocator = _RecordingAllocator()
    original_add_locator = sqlite_transaction_module.add_source_locator

    def fail_middle_locator(
        connection: sqlite3.Connection,
        revision: SourceLocatorRevision,
    ) -> None:
        if revision.fingerprint.content_hash == _OTHER_HASH:
            raise StorageConflictError()
        original_add_locator(connection, revision)

    monkeypatch.setattr(
        sqlite_transaction_module,
        "add_source_locator",
        fail_middle_locator,
    )
    try:
        results = persist_discovered_source_identities(
            units,
            request=request,
            inventory=inventory,
            repository=repository,
            allocator=allocator,
        )

        assert len(results) == 3
        assert isinstance(results[0], IdentityApplicationResult)
        assert results[0].created_document_version
        assert isinstance(results[1], DiscoveryFailure)
        assert results[1].category is DiscoveryFailureCategory.IDENTITY_WRITE_FAILED
        assert isinstance(results[2], IdentityApplicationResult)
        assert results[2].created_document_version
        assert tuple(
            result.source.ordinal
            if isinstance(result, IdentityApplicationResult)
            else result.ordinal
            for result in results
        ) == (0, 1, 2)

        assert len(allocator.source_ids) == 4
        assert len(allocator.document_ids) == 4
        assert len(allocator.version_ids) == 4
        assert results[0].source.source_observation_id == allocator.source_ids[0]
        assert results[2].source.source_observation_id == allocator.source_ids[3]
        for failed_attempt in (1, 2):
            assert (
                repository.get_document(allocator.document_ids[failed_attempt]) is None
            )
            assert (
                repository.get_document_version(allocator.version_ids[failed_attempt])
                is None
            )
            assert (
                repository.get_corpus_document_version_membership(
                    _CORPUS_ID,
                    allocator.version_ids[failed_attempt],
                )
                is None
            )
            assert (
                repository.get_source_observation(
                    _CORPUS_ID,
                    allocator.source_ids[failed_attempt],
                )
                is None
            )
            assert (
                repository.get_source_locator(
                    _CORPUS_ID,
                    allocator.source_ids[failed_attempt],
                )
                is None
            )

        assert len(repository.list_corpus_document_version_memberships(_CORPUS_ID)) == 2
        assert (
            repository.find_document_versions_by_content_hash(
                _CORPUS_ID,
                _OTHER_HASH,
            )
            == ()
        )
    finally:
        repository.close()


def test_identity_batch_requires_unique_discovery_order(tmp_path: Path) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    unit = IdentityPersistenceUnit(source, snapshot)

    with pytest.raises(ValueError, match="unique ordered ordinals"):
        persist_discovered_source_identities(
            (unit, unit),
            request=request,
            inventory=inventory,
            repository=_Repository(),  # type: ignore[arg-type]
            allocator=UuidDiscoveryIdentityIdAllocator(),
        )


@pytest.mark.parametrize(
    ("error", "expected_category"),
    [
        (
            StorageConcurrencyError(),
            DiscoveryFailureCategory.CONCURRENT_WRITE_CONFLICT,
        ),
        (StorageConflictError(), DiscoveryFailureCategory.IDENTITY_WRITE_FAILED),
    ],
)
def test_retry_exhaustion_returns_bounded_persistence_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: StorageConcurrencyError | StorageConflictError,
    expected_category: DiscoveryFailureCategory,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    resolved = _resolved(source, snapshot, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID)
    transactions = (_Transaction(), _Transaction())
    repository = _Repository(*transactions)
    resolver = Mock(return_value=resolved)
    application = Mock(side_effect=(error, error))
    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        resolver,
    )
    monkeypatch.setattr(
        persistence_module,
        "apply_discovery_identity_outcome",
        application,
    )
    with capture_logs() as logs:
        result = _persist(source, snapshot, request, inventory, repository)

    assert isinstance(result, DiscoveryFailure)
    assert result.category is expected_category
    assert result.file_kind is DiscoveryFileKind.REGULAR_FILE
    assert result.media_type == "text/plain"
    assert repository.calls == resolver.call_count == application.call_count == 2
    assert [record["event"] for record in logs] == [
        "identity_persistence_retry",
        "identity_persistence_failed",
    ]
    assert logs[-1]["error_category"] == expected_category.value
    assert logs[-1]["failure_scope"] == "source"
    assert logs[-1]["failure_stage"] == "persistence"
    assert logs[-1]["retryable"] is (
        expected_category is DiscoveryFailureCategory.CONCURRENT_WRITE_CONFLICT
    )
    assert "source.txt" not in repr(logs)
    assert _HASH.digest not in repr(logs)


def test_request_fatal_storage_failure_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    resolved = _resolved(source, snapshot, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID)
    transaction = _Transaction()
    repository = _Repository(transaction)
    resolver = Mock(return_value=resolved)
    application = Mock(side_effect=RepositoryUnavailableError())
    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        resolver,
    )
    monkeypatch.setattr(
        persistence_module,
        "apply_discovery_identity_outcome",
        application,
    )

    with pytest.raises(RepositoryUnavailableError):
        _persist(source, snapshot, request, inventory, repository)

    assert repository.calls == resolver.call_count == application.call_count == 1


def test_resolution_failure_commits_no_application_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    failure = DiscoveryFailure.for_category(
        ordinal=0,
        category=DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS,
        locator=source.candidate.locator,
        file_kind=DiscoveryFileKind.REGULAR_FILE,
        media_type="text/plain",
    )
    transaction = _Transaction()
    repository = _Repository(transaction)
    resolver = Mock(return_value=failure)
    application = Mock()
    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        resolver,
    )
    monkeypatch.setattr(
        persistence_module,
        "apply_discovery_identity_outcome",
        application,
    )

    result = _persist(source, snapshot, request, inventory, repository)

    assert result == failure
    application.assert_not_called()
    assert transaction.exit_exception is None
