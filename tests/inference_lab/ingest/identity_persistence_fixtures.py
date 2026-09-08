"""Shared doubles and builders for identity-persistence tests."""

from pathlib import Path
from types import TracebackType

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    ClassifiedSourceCandidate,
    DiscoveredSource,
    DiscoveryFailure,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    FileIdentityHint,
    FileStatEvidence,
    IdentityApplicationResult,
    IdentityPersistenceUnit,
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

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_FIRST_DOCUMENT_ID = DocumentId("22222222-2222-4222-8222-222222222222")
_SECOND_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_FIRST_VERSION_ID = DocumentVersionId("44444444-4444-4444-8444-444444444444")
_SECOND_VERSION_ID = DocumentVersionId("55555555-5555-4555-8555-555555555555")
_SOURCE_ID = SourceObservationId("66666666-6666-4666-8666-666666666666")
_SECOND_SOURCE_ID = SourceObservationId("77777777-7777-4777-8777-777777777777")
_HASH = ContentHash(HashAlgorithm.SHA256, "a" * 64)
_OTHER_HASH = ContentHash(HashAlgorithm.SHA256, "b" * 64)
_THIRD_HASH = ContentHash(HashAlgorithm.SHA256, "c" * 64)


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


class _CountingAllocator:
    def __init__(self) -> None:
        self.source_calls = 0
        self.document_calls = 0
        self.version_calls = 0

    def new_source_observation_id(self) -> SourceObservationId:
        self.source_calls += 1
        return _SOURCE_ID

    def new_document_id(self) -> DocumentId:
        self.document_calls += 1
        return _FIRST_DOCUMENT_ID

    def new_document_version_id(self) -> DocumentVersionId:
        self.version_calls += 1
        return _FIRST_VERSION_ID

    def counts(self) -> tuple[int, int, int]:
        return self.source_calls, self.document_calls, self.version_calls


class _CollisionThenUniqueAllocator(_CountingAllocator):
    def new_source_observation_id(self) -> SourceObservationId:
        self.source_calls += 1
        return (_SOURCE_ID, _SECOND_SOURCE_ID)[self.source_calls - 1]

    def new_document_id(self) -> DocumentId:
        self.document_calls += 1
        return _SECOND_DOCUMENT_ID

    def new_document_version_id(self) -> DocumentVersionId:
        self.version_calls += 1
        return _SECOND_VERSION_ID


class _AlternateCountingAllocator(_CountingAllocator):
    def new_source_observation_id(self) -> SourceObservationId:
        self.source_calls += 1
        return _SECOND_SOURCE_ID

    def new_document_id(self) -> DocumentId:
        self.document_calls += 1
        return _SECOND_DOCUMENT_ID

    def new_document_version_id(self) -> DocumentVersionId:
        self.version_calls += 1
        return _SECOND_VERSION_ID


class _RecordingAllocator:
    def __init__(self) -> None:
        self.source_ids: list[SourceObservationId] = []
        self.document_ids: list[DocumentId] = []
        self.version_ids: list[DocumentVersionId] = []

    def new_source_observation_id(self) -> SourceObservationId:
        value = SourceObservationId.generate()
        self.source_ids.append(value)
        return value

    def new_document_id(self) -> DocumentId:
        value = DocumentId.generate()
        self.document_ids.append(value)
        return value

    def new_document_version_id(self) -> DocumentVersionId:
        value = DocumentVersionId.generate()
        self.version_ids.append(value)
        return value


def _persistence_unit(
    root: Path,
    *,
    ordinal: int,
    filename: str,
    content_hash: ContentHash,
) -> IdentityPersistenceUnit:
    path = root / filename
    locator = SourceLocator(path, 0, Path(filename))
    evidence = FileStatEvidence(
        42,
        123_456_789 + ordinal,
        FileIdentityHint(7, 11 + ordinal),
    )
    candidate = TraversalCandidate(ordinal, locator, evidence)
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
        content_hash=content_hash,
        file_evidence=StableFileEvidence(evidence, evidence, evidence, evidence),
    )
    return IdentityPersistenceUnit(source, snapshot)


def _case(
    tmp_path: Path,
) -> tuple[
    ClassifiedSourceCandidate,
    VerifiedSourceSnapshot,
    DiscoveryRequest,
    TraversalInventory,
]:
    root = (tmp_path / "sources").resolve()
    unit = _persistence_unit(root, ordinal=0, filename="source.txt", content_hash=_HASH)
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(DiscoveryInput(root, DiscoveryInputKind.DIRECTORY),),
    )
    inventory = TraversalInventory(
        corpus_id=request.corpus_id,
        root_count=1,
        candidates=(unit.source.candidate,),
        failures=(),
        complete_root_ordinals=(0,),
        root_scopes=(TraversalRootScope(0, root, DiscoveryInputKind.DIRECTORY),),
    )
    return unit.source, unit.snapshot, request, inventory


def _resolved(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    document_id: DocumentId,
    version_id: DocumentVersionId,
) -> DiscoveredSource:
    version = DocumentVersion(version_id, document_id, _HASH)
    return DiscoveredSource(
        ordinal=source.candidate.ordinal,
        corpus_id=_CORPUS_ID,
        policy=CURRENT_DISCOVERY_POLICY,
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
