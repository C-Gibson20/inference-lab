"""Tests for conservative source-observation continuity resolution."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
    DiscoveryInputKind,
    FileIdentityHint,
    FileStatEvidence,
    ObservationMatchKind,
    SourceLocator,
    SourceObservationAssignment,
    StableFileEvidence,
    TraversalCandidate,
    TraversalInventory,
    TraversalRootScope,
    VerifiedSourceSnapshot,
    resolve_source_observation,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    DocumentId,
    DocumentVersion,
    DocumentVersionId,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    HashAlgorithm,
    IdentityIdCollisionError,
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
_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_VERSION_ID = DocumentVersionId("44444444-4444-4444-8444-444444444444")
_SOURCE_ID = SourceObservationId("55555555-5555-4555-8555-555555555555")
_SECOND_SOURCE_ID = SourceObservationId("66666666-6666-4666-8666-666666666666")
_NEW_SOURCE_ID = SourceObservationId("77777777-7777-4777-8777-777777777777")
_HASH = ContentHash(HashAlgorithm.SHA256, "a" * 64)
_OTHER_HASH = ContentHash(HashAlgorithm.SHA256, "b" * 64)
_DOI = ExternalIdentifier("doi", "10.1000/example")


@dataclass
class _Reader:
    locators: tuple[SourceLocatorRevision, ...] = ()
    versions: tuple[DocumentVersion, ...] = ()
    observations: tuple[SourceObservationState, ...] = ()
    assertions: tuple[ExternalIdentifierAssertion, ...] = ()
    corpus_calls: list[CorpusId] = field(default_factory=list)

    def _record(self, corpus_id: CorpusId) -> None:
        self.corpus_calls.append(corpus_id)

    def find_source_locator_by_path(
        self,
        corpus_id: CorpusId,
        path: Path,
    ) -> SourceLocatorRevision | None:
        self._record(corpus_id)
        return next(
            (locator for locator in self.locators if locator.path == path), None
        )

    def find_source_locators_by_identity_hint(
        self,
        corpus_id: CorpusId,
        identity_hint: SourceFileIdentityHint,
    ) -> tuple[SourceLocatorRevision, ...]:
        self._record(corpus_id)
        return tuple(
            locator
            for locator in self.locators
            if locator.identity_hint == identity_hint
        )

    def find_document_versions_by_content_hash(
        self,
        corpus_id: CorpusId,
        content_hash: ContentHash,
    ) -> tuple[DocumentVersion, ...]:
        self._record(corpus_id)
        return tuple(
            version for version in self.versions if version.content_hash == content_hash
        )

    def list_external_identifier_assertions(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        self._record(corpus_id)
        return tuple(
            assertion
            for assertion in self.assertions
            if assertion.document_version_id == document_version_id
        )

    def find_source_observations_by_document_version(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[SourceObservationState, ...]:
        self._record(corpus_id)
        return tuple(
            observation
            for observation in self.observations
            if observation.document_version_id == document_version_id
        )

    def get_source_locator(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> SourceLocatorRevision | None:
        self._record(corpus_id)
        return next(
            (
                locator
                for locator in self.locators
                if locator.source_observation_id == source_observation_id
            ),
            None,
        )

    def get_source_observation(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> SourceObservationState | None:
        self._record(corpus_id)
        return next(
            (
                observation
                for observation in self.observations
                if observation.source_observation_id == source_observation_id
            ),
            None,
        )


@dataclass
class _Allocator:
    value: object = _NEW_SOURCE_ID
    calls: int = 0

    def new_source_observation_id(self) -> object:
        self.calls += 1
        return self.value


def _snapshot(
    path: Path,
    *,
    content_hash: ContentHash = _HASH,
    device_id: int = 7,
    file_id: int = 11,
) -> VerifiedSourceSnapshot:
    stat_evidence = FileStatEvidence(
        byte_count=42,
        modified_time_ns=123_456_789,
        identity_hint=FileIdentityHint(device_id=device_id, file_id=file_id),
    )
    return VerifiedSourceSnapshot(
        locator=SourceLocator(
            path=path,
            request_root_ordinal=0,
            relative_path=Path(path.name),
        ),
        media_type="text/plain",
        encoding="utf-8",
        byte_count=42,
        content_hash=content_hash,
        file_evidence=StableFileEvidence(
            stat_evidence,
            stat_evidence,
            stat_evidence,
            stat_evidence,
        ),
    )


def _locator(
    path: Path,
    *,
    source_id: SourceObservationId = _SOURCE_ID,
    content_hash: ContentHash = _HASH,
    device_id: int = 7,
    file_id: int = 11,
) -> SourceLocatorRevision:
    return SourceLocatorRevision(
        corpus_id=_CORPUS_ID,
        source_observation_id=source_id,
        locator_revision=1,
        path=path,
        identity_hint=SourceFileIdentityHint(device_id=device_id, file_id=file_id),
        fingerprint=SourceFingerprint(
            content_hash=content_hash,
            byte_count=42,
            modified_time_ns=123_456_789,
        ),
    )


def _version(
    *,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
) -> DocumentVersion:
    return DocumentVersion(
        document_version_id=_VERSION_ID,
        document_id=_DOCUMENT_ID,
        content_hash=_HASH,
        external_identifiers=external_identifiers,
    )


def _observation(
    source_id: SourceObservationId = _SOURCE_ID,
) -> SourceObservationState:
    return SourceObservationState(
        corpus_id=_CORPUS_ID,
        source_observation_id=source_id,
        observation_revision=1,
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
    )


def _inventory(
    root: Path,
    snapshot: VerifiedSourceSnapshot,
    *,
    complete: bool = True,
    live_prior: SourceLocatorRevision | None = None,
    failures: tuple[DiscoveryFailure, ...] = (),
) -> TraversalInventory:
    candidates = [
        TraversalCandidate(
            ordinal=0,
            locator=snapshot.locator,
            stat_evidence=snapshot.file_evidence.pre_open,
        )
    ]
    if live_prior is not None:
        candidates.append(
            TraversalCandidate(
                ordinal=1,
                locator=SourceLocator(
                    path=live_prior.path,
                    request_root_ordinal=0,
                    relative_path=live_prior.path.relative_to(root),
                ),
                stat_evidence=snapshot.file_evidence.pre_open,
            )
        )
    return TraversalInventory(
        root_count=1,
        candidates=tuple(candidates),
        failures=failures,
        complete_root_ordinals=(0,) if complete else (),
        root_scopes=(
            TraversalRootScope(
                ordinal=0,
                path=root,
                kind=DiscoveryInputKind.DIRECTORY,
            ),
        ),
    )


def _resolve(
    snapshot: VerifiedSourceSnapshot,
    inventory: TraversalInventory,
    reader: _Reader,
    allocator: _Allocator,
    *,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
) -> SourceObservationAssignment | DiscoveryFailure:
    return resolve_source_observation(
        snapshot,
        ordinal=0,
        corpus_id=_CORPUS_ID,
        external_identifiers=external_identifiers,
        inventory=inventory,
        reader=reader,  # type: ignore[arg-type]
        allocator=allocator,  # type: ignore[arg-type]
    )


def test_exact_locator_accepts_same_hash_after_filesystem_identity_change(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    current = root / "source.txt"
    snapshot = _snapshot(current, device_id=8, file_id=12)
    stored = _locator(current)
    allocator = _Allocator()

    assignment = _resolve(
        snapshot,
        _inventory(root, snapshot),
        _Reader(locators=(stored,)),
        allocator,
    )

    assert assignment == SourceObservationAssignment(
        _SOURCE_ID,
        ObservationMatchKind.EXACT_LOCATOR,
        stored,
    )
    assert allocator.calls == 0


def test_exact_locator_accepts_changed_bytes_for_same_filesystem_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    current = root / "source.txt"
    snapshot = _snapshot(current, content_hash=_OTHER_HASH)
    stored = _locator(current)

    assignment = _resolve(
        snapshot,
        _inventory(root, snapshot),
        _Reader(locators=(stored,)),
        _Allocator(),
    )

    assert isinstance(assignment, SourceObservationAssignment)
    assert assignment.match_kind is ObservationMatchKind.EXACT_LOCATOR
    assert assignment.source_observation_id == _SOURCE_ID


def test_unique_identity_move_requires_old_locator_absent_from_complete_scan(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    prior = _locator(root / "old.txt")
    snapshot = _snapshot(root / "new.txt", content_hash=_OTHER_HASH)

    assignment = _resolve(
        snapshot,
        _inventory(root, snapshot),
        _Reader(locators=(prior,)),
        _Allocator(),
    )

    assert assignment == SourceObservationAssignment(
        _SOURCE_ID,
        ObservationMatchKind.FILESYSTEM_IDENTITY,
        prior,
    )


def test_live_hard_link_remains_an_independent_new_observation(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    prior = _locator(root / "still-live.txt")
    snapshot = _snapshot(root / "new-link.txt", content_hash=_OTHER_HASH)
    allocator = _Allocator()

    assignment = _resolve(
        snapshot,
        _inventory(root, snapshot, live_prior=prior),
        _Reader(locators=(prior,)),
        allocator,
    )

    assert assignment == SourceObservationAssignment(
        _NEW_SOURCE_ID,
        ObservationMatchKind.NEW,
    )
    assert allocator.calls == 1


@pytest.mark.parametrize(
    ("category", "file_kind"),
    (
        (DiscoveryFailureCategory.SOURCE_SYMLINK, DiscoveryFileKind.SYMLINK),
        (DiscoveryFailureCategory.SOURCE_SPECIAL_FILE, DiscoveryFileKind.SPECIAL),
        (DiscoveryFailureCategory.NESTED_MOUNT, DiscoveryFileKind.DIRECTORY),
    ),
)
def test_policy_skipped_prior_path_remains_an_independent_observation(
    tmp_path: Path,
    category: DiscoveryFailureCategory,
    file_kind: DiscoveryFileKind,
) -> None:
    root = tmp_path.resolve()
    prior = _locator(root / "still-present.txt")
    snapshot = _snapshot(root / "new.txt", content_hash=_OTHER_HASH)
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=category,
        locator=SourceLocator(
            path=prior.path,
            request_root_ordinal=0,
            relative_path=prior.path.relative_to(root),
        ),
        file_kind=file_kind,
    )
    allocator = _Allocator()

    assignment = _resolve(
        snapshot,
        _inventory(root, snapshot, failures=(failure,)),
        _Reader(locators=(prior,)),
        allocator,
    )

    assert assignment == SourceObservationAssignment(
        _NEW_SOURCE_ID,
        ObservationMatchKind.NEW,
    )
    assert allocator.calls == 1


@pytest.mark.parametrize(
    ("category", "file_kind"),
    (
        (DiscoveryFailureCategory.SOURCE_SYMLINK, DiscoveryFileKind.SYMLINK),
        (DiscoveryFailureCategory.NESTED_MOUNT, DiscoveryFileKind.DIRECTORY),
        (DiscoveryFailureCategory.TRAVERSAL_FAILED, DiscoveryFileKind.DIRECTORY),
    ),
)
def test_untraversed_prior_path_reports_ambiguity(
    tmp_path: Path,
    category: DiscoveryFailureCategory,
    file_kind: DiscoveryFileKind,
) -> None:
    root = tmp_path.resolve()
    skipped_directory = root / "untraversed"
    prior = _locator(skipped_directory / "possibly-present.txt")
    snapshot = _snapshot(root / "new.txt", content_hash=_OTHER_HASH)
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=category,
        locator=SourceLocator(
            path=skipped_directory,
            request_root_ordinal=0,
            relative_path=skipped_directory.relative_to(root),
        ),
        file_kind=file_kind,
    )
    allocator = _Allocator()

    result = _resolve(
        snapshot,
        _inventory(root, snapshot, failures=(failure,)),
        _Reader(locators=(prior,)),
        allocator,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS
    assert allocator.calls == 0


def test_live_byte_identical_path_remains_an_independent_new_observation(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    prior = _locator(root / "still-live.txt", device_id=99, file_id=101)
    snapshot = _snapshot(root / "copied.txt")
    reader = _Reader(
        locators=(prior,),
        versions=(_version(),),
        observations=(_observation(),),
    )

    assignment = _resolve(
        snapshot,
        _inventory(root, snapshot, live_prior=prior),
        reader,
        _Allocator(),
    )

    assert assignment == SourceObservationAssignment(
        _NEW_SOURCE_ID,
        ObservationMatchKind.NEW,
    )


def test_traversal_gap_reports_ambiguity_without_allocating_or_reassigning(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    prior = _locator(root / "possibly-missing.txt")
    snapshot = _snapshot(root / "new.txt", content_hash=_OTHER_HASH)
    allocator = _Allocator()

    result = _resolve(
        snapshot,
        _inventory(root, snapshot, complete=False),
        _Reader(locators=(prior,)),
        allocator,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS
    assert allocator.calls == 0


def test_unique_missing_hash_match_uses_compatible_asserted_identifiers(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    prior = _locator(root / "old.txt", device_id=99, file_id=101)
    snapshot = _snapshot(root / "new.txt")
    version = _version()
    assertion = ExternalIdentifierAssertion(_SOURCE_ID, _VERSION_ID, _DOI)
    reader = _Reader(
        locators=(prior,),
        versions=(version,),
        observations=(_observation(),),
        assertions=(assertion,),
    )

    assignment = _resolve(
        snapshot,
        _inventory(root, snapshot),
        reader,
        _Allocator(),
        external_identifiers=(_DOI,),
    )

    assert assignment == SourceObservationAssignment(
        _SOURCE_ID,
        ObservationMatchKind.CONTENT_HASH,
        prior,
    )
    assert reader.corpus_calls
    assert set(reader.corpus_calls) == {_CORPUS_ID}
    assert _OTHER_CORPUS_ID not in reader.corpus_calls


def test_conflicting_identifier_prevents_hash_continuity_reassignment(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    prior = _locator(root / "old.txt", device_id=99, file_id=101)
    snapshot = _snapshot(root / "new.txt")
    reader = _Reader(
        locators=(prior,),
        versions=(_version(external_identifiers=(_DOI,)),),
        observations=(_observation(),),
    )

    assignment = _resolve(
        snapshot,
        _inventory(root, snapshot),
        reader,
        _Allocator(),
        external_identifiers=(ExternalIdentifier("doi", "10.1000/different"),),
    )

    assert assignment == SourceObservationAssignment(
        _NEW_SOURCE_ID,
        ObservationMatchKind.NEW,
    )


def test_multiple_missing_identity_candidates_report_ambiguity(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    first = _locator(root / "first.txt")
    second = _locator(root / "second.txt", source_id=_SECOND_SOURCE_ID)
    snapshot = _snapshot(root / "current.txt", content_hash=_OTHER_HASH)

    result = _resolve(
        snapshot,
        _inventory(root, snapshot),
        _Reader(locators=(first, second)),
        _Allocator(),
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS


def test_new_assignment_rejects_allocator_collision(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    snapshot = _snapshot(root / "new.txt")
    collision = SourceObservationState(
        corpus_id=_CORPUS_ID,
        source_observation_id=_NEW_SOURCE_ID,
        observation_revision=1,
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
    )

    with pytest.raises(IdentityIdCollisionError, match="existing SourceObservationId"):
        _resolve(
            snapshot,
            _inventory(root, snapshot),
            _Reader(observations=(collision,)),
            _Allocator(),
        )
