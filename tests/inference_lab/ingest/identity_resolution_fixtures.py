"""Shared doubles and builders for identity-resolution tests."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    SourceClassification,
    SourceLocator,
    SourceSupport,
    StableFileEvidence,
    TraversalCandidate,
    TraversalInventory,
    TraversalRootScope,
    VerifiedSourceSnapshot,
    resolve_discovered_source_identity,
)
from inference_lab.provenance import (
    CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY,
    ContentHash,
    CorpusId,
    DocumentId,
    DocumentVersion,
    DocumentVersionId,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    ExternalIdentifierResolutionPolicy,
    HashAlgorithm,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage import (
    SourceFileIdentityHint,
    SourceFingerprint,
    SourceLocatorRevision,
)

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_OTHER_CORPUS_ID = CorpusId("22222222-2222-4222-8222-222222222222")
_FIRST_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_SECOND_DOCUMENT_ID = DocumentId("44444444-4444-4444-8444-444444444444")
_ALLOCATED_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_FIRST_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")
_SECOND_VERSION_ID = DocumentVersionId("77777777-7777-4777-8777-777777777777")
_ALLOCATED_VERSION_ID = DocumentVersionId("88888888-8888-4888-8888-888888888888")
_SOURCE_ID = SourceObservationId("99999999-9999-4999-8999-999999999999")
_SECOND_SOURCE_ID = SourceObservationId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_ALLOCATED_SOURCE_ID = SourceObservationId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_HASH_A = ContentHash(HashAlgorithm.SHA256, "a" * 64)
_HASH_B = ContentHash(HashAlgorithm.SHA256, "b" * 64)
_DOI = ExternalIdentifier("doi", "10.1000/example")
_OTHER_DOI = ExternalIdentifier("doi", "10.1000/different")
_PMID = ExternalIdentifier("pmid", "12345678")
_AUTHORITATIVE_DOI_POLICY = ExternalIdentifierResolutionPolicy(
    policy_name="fixture-doi",
    policy_version="1.0.0",
    authoritative_namespaces=("doi",),
)


@dataclass
class _Allocator:
    source_id: SourceObservationId = _ALLOCATED_SOURCE_ID
    document_id: DocumentId = _ALLOCATED_DOCUMENT_ID
    version_id: DocumentVersionId = _ALLOCATED_VERSION_ID
    source_calls: int = 0
    document_calls: int = 0
    version_calls: int = 0

    def new_source_observation_id(self) -> SourceObservationId:
        self.source_calls += 1
        return self.source_id

    def new_document_id(self) -> DocumentId:
        self.document_calls += 1
        return self.document_id

    def new_document_version_id(self) -> DocumentVersionId:
        self.version_calls += 1
        return self.version_id


@dataclass
class _Transaction:
    versions: tuple[DocumentVersion, ...] = ()
    memberships: dict[CorpusId, frozenset[DocumentVersionId]] = field(
        default_factory=dict
    )
    observations: tuple[SourceObservationState, ...] = ()
    locators: tuple[SourceLocatorRevision, ...] = ()
    assertions: tuple[ExternalIdentifierAssertion, ...] = ()
    corpus_calls: list[CorpusId] = field(default_factory=list)
    active: bool = False

    def __enter__(self) -> "_Transaction":
        assert not self.active
        self.active = True
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> Literal[False]:
        del exception_type, exception, traceback
        self.active = False
        return False

    def _check(self, corpus_id: CorpusId | None = None) -> None:
        assert self.active
        if corpus_id is not None:
            self.corpus_calls.append(corpus_id)

    def _is_member(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> bool:
        return document_version_id in self.memberships.get(corpus_id, frozenset())

    def get_source_observation(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> SourceObservationState | None:
        self._check(corpus_id)
        return next(
            (
                observation
                for observation in self.observations
                if observation.corpus_id == corpus_id
                and observation.source_observation_id == source_observation_id
            ),
            None,
        )

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        self._check()
        return next(
            (
                version
                for version in self.versions
                if version.document_version_id == document_version_id
            ),
            None,
        )

    def find_source_locator_by_path(
        self,
        corpus_id: CorpusId,
        path: Path,
    ) -> SourceLocatorRevision | None:
        self._check(corpus_id)
        return next(
            (
                locator
                for locator in self.locators
                if locator.corpus_id == corpus_id and locator.path == path
            ),
            None,
        )

    def find_source_locators_by_identity_hint(
        self,
        corpus_id: CorpusId,
        identity_hint: SourceFileIdentityHint,
    ) -> tuple[SourceLocatorRevision, ...]:
        self._check(corpus_id)
        return tuple(
            locator
            for locator in self.locators
            if locator.corpus_id == corpus_id and locator.identity_hint == identity_hint
        )

    def find_document_versions_by_content_hash(
        self,
        corpus_id: CorpusId,
        content_hash: ContentHash,
    ) -> tuple[DocumentVersion, ...]:
        self._check(corpus_id)
        return tuple(
            version
            for version in self.versions
            if self._is_member(corpus_id, version.document_version_id)
            and version.content_hash == content_hash
        )

    def find_document_versions_by_external_identifiers(
        self,
        corpus_id: CorpusId,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        self._check(corpus_id)
        supplied = {identifier.as_tuple() for identifier in external_identifiers}
        matches: list[DocumentVersion] = []
        for version in self.versions:
            if not self._is_member(corpus_id, version.document_version_id):
                continue
            effective = {item.as_tuple() for item in version.external_identifiers}
            effective.update(
                assertion.external_identifier.as_tuple()
                for assertion in self.assertions
                if assertion.document_version_id == version.document_version_id
                and any(
                    observation.corpus_id == corpus_id
                    and observation.source_observation_id
                    == assertion.source_observation_id
                    for observation in self.observations
                )
            )
            if supplied & effective:
                matches.append(version)
        return tuple(
            sorted(
                matches,
                key=lambda item: (
                    item.document_id.as_str(),
                    item.document_version_id.as_str(),
                ),
            )
        )

    def list_external_identifier_assertions(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        self._check(corpus_id)
        return tuple(
            assertion
            for assertion in self.assertions
            if assertion.document_version_id == document_version_id
            and any(
                observation.corpus_id == corpus_id
                and observation.source_observation_id == assertion.source_observation_id
                for observation in self.observations
            )
        )

    def find_source_observations_by_document_version(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[SourceObservationState, ...]:
        self._check(corpus_id)
        return tuple(
            observation
            for observation in self.observations
            if observation.corpus_id == corpus_id
            and observation.document_version_id == document_version_id
        )

    def get_source_locator(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> SourceLocatorRevision | None:
        self._check(corpus_id)
        return next(
            (
                locator
                for locator in self.locators
                if locator.corpus_id == corpus_id
                and locator.source_observation_id == source_observation_id
            ),
            None,
        )


def _version(
    version_id: DocumentVersionId = _FIRST_VERSION_ID,
    document_id: DocumentId = _FIRST_DOCUMENT_ID,
    content_hash: ContentHash = _HASH_A,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
) -> DocumentVersion:
    return DocumentVersion(
        version_id,
        document_id,
        content_hash,
        external_identifiers,
    )


def _observation(
    *,
    corpus_id: CorpusId = _CORPUS_ID,
    source_id: SourceObservationId = _SOURCE_ID,
    version: DocumentVersion,
) -> SourceObservationState:
    return SourceObservationState(
        corpus_id,
        source_id,
        1,
        version.document_id,
        version.document_version_id,
    )


def _locator(
    path: Path,
    *,
    corpus_id: CorpusId = _CORPUS_ID,
    source_id: SourceObservationId = _SOURCE_ID,
    content_hash: ContentHash = _HASH_A,
    device_id: int = 7,
    file_id: int = 11,
) -> SourceLocatorRevision:
    return SourceLocatorRevision(
        corpus_id=corpus_id,
        source_observation_id=source_id,
        locator_revision=1,
        path=path,
        identity_hint=SourceFileIdentityHint(device_id, file_id),
        fingerprint=SourceFingerprint(content_hash, 42, 123_456_789),
    )


def _source_case(
    tmp_path: Path,
    *,
    content_hash: ContentHash = _HASH_A,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
    complete: bool = True,
    live_locators: tuple[SourceLocatorRevision, ...] = (),
) -> tuple[
    ClassifiedSourceCandidate,
    VerifiedSourceSnapshot,
    TraversalInventory,
]:
    root = (tmp_path / "sources").resolve()
    path = root / "current.txt"
    evidence = FileStatEvidence(
        byte_count=42,
        modified_time_ns=123_456_789,
        identity_hint=FileIdentityHint(7, 11),
    )
    candidate = TraversalCandidate(
        ordinal=0,
        locator=SourceLocator(path, 0, Path("current.txt")),
        stat_evidence=evidence,
        external_identifiers=external_identifiers,
    )
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
        locator=candidate.locator,
        media_type=classification.media_type,
        encoding=classification.encoding,
        byte_count=42,
        content_hash=content_hash,
        file_evidence=StableFileEvidence(evidence, evidence, evidence, evidence),
    )
    candidates = [candidate]
    for ordinal, locator in enumerate(live_locators, start=1):
        candidates.append(
            TraversalCandidate(
                ordinal=ordinal,
                locator=SourceLocator(
                    locator.path,
                    0,
                    locator.path.relative_to(root),
                ),
                stat_evidence=evidence,
            )
        )
    inventory = TraversalInventory(
        corpus_id=_CORPUS_ID,
        root_count=1,
        candidates=tuple(candidates),
        failures=(),
        complete_root_ordinals=(0,) if complete else (),
        root_scopes=(TraversalRootScope(0, root, DiscoveryInputKind.DIRECTORY),),
    )
    return source, snapshot, inventory


def _request(
    tmp_path: Path,
    *,
    corpus_id: CorpusId = _CORPUS_ID,
    policy: ExternalIdentifierResolutionPolicy = (
        CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY
    ),
) -> DiscoveryRequest:
    return DiscoveryRequest(
        corpus_id=corpus_id,
        inputs=(
            DiscoveryInput(
                root=(tmp_path / "sources").resolve(),
                kind=DiscoveryInputKind.DIRECTORY,
            ),
        ),
        external_identifier_policy=policy,
    )


def _resolve(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    inventory: TraversalInventory,
    transaction: _Transaction,
    allocator: _Allocator,
    request: DiscoveryRequest,
) -> DiscoveredSource | DiscoveryFailure:
    with transaction:
        result = resolve_discovered_source_identity(
            source,
            snapshot,
            request=request,
            inventory=inventory,
            transaction=transaction,  # type: ignore[arg-type]
            allocator=allocator,
        )
    return result
