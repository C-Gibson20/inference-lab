"""Tests for identity-persistence retry policy and recovery."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

import inference_lab.ingest.identity_persistence as persistence_module
from inference_lab.ingest import (
    DiscoveredSource,
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
    IdentityApplicationResult,
    persist_discovered_source_identity,
)
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
    IdentityIdCollisionError,
    SourceObservationState,
)
from inference_lab.storage import (
    RepositoryUnavailableError,
    StorageConcurrencyError,
    StorageConflictError,
)
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
    _case,
    _CollisionThenUniqueAllocator,
    _persist,
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
