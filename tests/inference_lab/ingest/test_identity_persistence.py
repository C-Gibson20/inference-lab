"""Tests for per-source identity resolution and persistence coordination."""

from pathlib import Path
from types import TracebackType
from unittest.mock import Mock

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

import inference_lab.ingest.identity_persistence as persistence_module
from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    ClassifiedSourceCandidate,
    DiscoveredSource,
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    FileIdentityHint,
    FileStatEvidence,
    IdentityApplicationResult,
    SourceClassification,
    SourceLocator,
    SourceSupport,
    StableFileEvidence,
    TraversalCandidate,
    TraversalInventory,
    TraversalRootScope,
    UuidDiscoveryIdentityIdAllocator,
    VerifiedSourceSnapshot,
    persist_discovered_source_identity,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    CorpusMetadata,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    IdentityResolutionOutcome,
    ResolutionKind,
    ResolutionReason,
    SourceObservationId,
)
from inference_lab.storage import (
    RepositoryUnavailableError,
    StorageConcurrencyError,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_FIRST_DOCUMENT_ID = DocumentId("22222222-2222-4222-8222-222222222222")
_SECOND_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_FIRST_VERSION_ID = DocumentVersionId("44444444-4444-4444-8444-444444444444")
_SECOND_VERSION_ID = DocumentVersionId("55555555-5555-4555-8555-555555555555")
_SOURCE_ID = SourceObservationId("66666666-6666-4666-8666-666666666666")
_HASH = ContentHash(HashAlgorithm.SHA256, "a" * 64)


class _Transaction:
    def __init__(self) -> None:
        self.active = False
        self.exit_exception: BaseException | None = None

    def __enter__(self) -> "_Transaction":
        assert not self.active
        self.active = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception_type, traceback
        assert self.active
        self.active = False
        self.exit_exception = exception
        return False


class _Repository:
    def __init__(self, *transactions: _Transaction) -> None:
        self._transactions = transactions
        self.calls = 0

    def transaction(self) -> _Transaction:
        transaction = self._transactions[self.calls]
        self.calls += 1
        return transaction


def _case(
    tmp_path: Path,
) -> tuple[
    ClassifiedSourceCandidate,
    VerifiedSourceSnapshot,
    DiscoveryRequest,
    TraversalInventory,
]:
    root = (tmp_path / "sources").resolve()
    path = root / "source.txt"
    locator = SourceLocator(path, 0, Path("source.txt"))
    evidence = FileStatEvidence(42, 123_456_789, FileIdentityHint(7, 11))
    candidate = TraversalCandidate(0, locator, evidence)
    classification = SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
        classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
        parser_capability="text-native",
        encoding="utf-8",
    )
    source = ClassifiedSourceCandidate(candidate, classification)
    snapshot = VerifiedSourceSnapshot(
        locator=locator,
        media_type="text/plain",
        encoding="utf-8",
        byte_count=42,
        content_hash=_HASH,
        file_evidence=StableFileEvidence(evidence, evidence, evidence, evidence),
    )
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(DiscoveryInput(root, DiscoveryInputKind.DIRECTORY),),
    )
    inventory = TraversalInventory(
        root_count=1,
        candidates=(candidate,),
        failures=(),
        complete_root_ordinals=(0,),
        root_scopes=(TraversalRootScope(0, root, DiscoveryInputKind.DIRECTORY),),
    )
    return source, snapshot, request, inventory


def _resolved(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    document_id: DocumentId,
    version_id: DocumentVersionId,
) -> DiscoveredSource:
    version = DocumentVersion(version_id, document_id, _HASH)
    return DiscoveredSource(
        ordinal=source.candidate.ordinal,
        source_observation_id=_SOURCE_ID,
        snapshot=snapshot,
        external_identifiers=(),
        family_candidates=(),
        classification=source.classification,
        identity_outcome=IdentityResolutionOutcome(
            kind=ResolutionKind.NEW_DOCUMENT,
            reason=ResolutionReason.NO_MATCH,
            document=DocumentIdentity(document_id),
            version=version,
        ),
    )


def _persist(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    request: DiscoveryRequest,
    inventory: TraversalInventory,
    repository: _Repository,
) -> IdentityApplicationResult | DiscoveryFailure:
    return persist_discovered_source_identity(
        source,
        snapshot,
        request=request,
        inventory=inventory,
        repository=repository,  # type: ignore[arg-type]
        allocator=UuidDiscoveryIdentityIdAllocator(),
    )


def test_real_repository_resolves_and_persists_one_atomic_source(
    tmp_path: Path,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Test corpus"))
    try:
        result = persist_discovered_source_identity(
            source,
            snapshot,
            request=request,
            inventory=inventory,
            repository=repository,
            allocator=UuidDiscoveryIdentityIdAllocator(),
        )

        assert isinstance(result, IdentityApplicationResult)
        assert result.parse_eligible
        outcome = result.source.identity_outcome
        assert outcome.document is not None
        assert outcome.version is not None
        assert repository.get_document(outcome.document.document_id) == outcome.document
        assert (
            repository.get_document_version(outcome.version.document_version_id)
            == outcome.version
        )
        assert (
            repository.get_source_observation(
                _CORPUS_ID,
                result.source.source_observation_id,
            )
            is not None
        )
        assert (
            repository.get_source_locator(
                _CORPUS_ID,
                result.source.source_observation_id,
            )
            is not None
        )
    finally:
        repository.close()


def test_resolution_and_application_share_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    resolved = _resolved(source, snapshot, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID)
    expected = IdentityApplicationResult(resolved, parse_eligible=True)
    transaction = _Transaction()
    repository = _Repository(transaction)
    resolver = Mock(return_value=resolved)
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

    result = _persist(source, snapshot, request, inventory, repository)

    assert result == expected
    assert resolver.call_args.kwargs["transaction"] is transaction
    assert application.call_args.kwargs["transaction"] is transaction
    assert transaction.exit_exception is None


def test_conflict_retry_rebuilds_resolution_in_a_new_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    first = _resolved(source, snapshot, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID)
    rebuilt = _resolved(source, snapshot, _SECOND_DOCUMENT_ID, _SECOND_VERSION_ID)
    expected = IdentityApplicationResult(rebuilt, parse_eligible=True)
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
    assert contexts == [{"corpus_id": _CORPUS_ID.as_str()}] * 2
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("identity_persistence_retry", "warning")
    ]
    assert logs[0]["error_category"] == "conflict"
    assert logs[0]["retry_number"] == 1


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
