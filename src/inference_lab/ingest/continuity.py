"""Conservative source-observation continuity resolution."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from inference_lab._validation import is_strict_integer
from inference_lab.ingest.discovery import (
    DiscoveryInputKind,
    VerifiedSourceSnapshot,
)
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
)
from inference_lab.ingest.traversal import TraversalInventory, canonical_path_key
from inference_lab.provenance import (
    CorpusId,
    DocumentVersion,
    ExternalIdentifier,
    IdentityIdCollisionError,
    SourceObservationId,
)
from inference_lab.provenance.values import validate_external_identifiers
from inference_lab.storage import (
    MetadataReader,
    SourceFileIdentityHint,
    SourceLocatorRevision,
)


class ObservationMatchKind(StrEnum):
    """Evidence that selected one source observation."""

    EXACT_LOCATOR = "exact_locator"
    FILESYSTEM_IDENTITY = "filesystem_identity"
    CONTENT_HASH = "content_hash"
    NEW = "new"


@dataclass(frozen=True, slots=True)
class SourceObservationAssignment:
    """Assign a verified source to a new or existing observation."""

    source_observation_id: SourceObservationId
    match_kind: ObservationMatchKind
    previous_locator: SourceLocatorRevision | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate assignment identity and evidence semantics."""
        if not isinstance(self.source_observation_id, SourceObservationId):
            raise TypeError("source_observation_id must be a SourceObservationId.")
        if not isinstance(self.match_kind, ObservationMatchKind):
            raise TypeError("match_kind must be an ObservationMatchKind.")
        if self.previous_locator is not None and not isinstance(
            self.previous_locator, SourceLocatorRevision
        ):
            raise TypeError("previous_locator must be a SourceLocatorRevision or None.")
        if (self.match_kind is ObservationMatchKind.NEW) != (
            self.previous_locator is None
        ):
            raise ValueError(
                "new assignments must omit previous_locator and existing "
                "assignments must retain it."
            )
        if (
            self.previous_locator is not None
            and self.previous_locator.source_observation_id
            != self.source_observation_id
        ):
            raise ValueError("previous_locator must identify the assigned observation.")


class SourceObservationIdAllocator(Protocol):
    """Allocate opaque IDs for newly observed sources."""

    def new_source_observation_id(self) -> SourceObservationId:
        """Allocate one new source-observation ID."""
        ...


class UuidSourceObservationIdAllocator:
    """Allocate production UUID-backed source-observation identifiers."""

    def new_source_observation_id(self) -> SourceObservationId:
        """Allocate one new source-observation ID."""
        return SourceObservationId.generate()


class _PriorPathStatus(StrEnum):
    LIVE = "live"
    ABSENT = "absent"
    UNPROVEN = "unproven"


def _identity_hint(snapshot: VerifiedSourceSnapshot) -> SourceFileIdentityHint | None:
    hint = snapshot.file_evidence.opened_after_read.identity_hint
    if hint is None:
        return None
    return SourceFileIdentityHint(
        device_id=hint.device_id,
        file_id=hint.file_id,
    )


def _exact_evidence_is_compatible(
    locator: SourceLocatorRevision,
    snapshot: VerifiedSourceSnapshot,
) -> bool:
    current_hint = _identity_hint(snapshot)
    return locator.fingerprint.content_hash == snapshot.content_hash or (
        current_hint is not None and locator.identity_hint == current_hint
    )


def _prior_path_status(
    path: Path,
    inventory: TraversalInventory,
) -> _PriorPathStatus:
    path_key = canonical_path_key(path)
    if any(
        canonical_path_key(candidate.locator.path) == path_key
        for candidate in inventory.candidates
    ):
        return _PriorPathStatus.LIVE

    containing_scopes = tuple(
        scope
        for scope in inventory.root_scopes
        if path == scope.path
        or (
            scope.kind is DiscoveryInputKind.DIRECTORY
            and path.is_relative_to(scope.path)
        )
    )
    if any(
        scope.ordinal in inventory.complete_root_ordinals for scope in containing_scopes
    ):
        return _PriorPathStatus.ABSENT
    return _PriorPathStatus.UNPROVEN


def _continuity_failure(
    *,
    ordinal: int,
    snapshot: VerifiedSourceSnapshot,
) -> DiscoveryFailure:
    return DiscoveryFailure.for_category(
        ordinal=ordinal,
        category=DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS,
        locator=snapshot.locator,
        file_kind=DiscoveryFileKind.REGULAR_FILE,
        media_type=snapshot.media_type,
    )


def _select_missing_locator(
    locators: tuple[SourceLocatorRevision, ...],
    *,
    ordinal: int,
    snapshot: VerifiedSourceSnapshot,
    inventory: TraversalInventory,
    match_kind: ObservationMatchKind,
) -> SourceObservationAssignment | DiscoveryFailure | None:
    by_observation = {locator.source_observation_id: locator for locator in locators}
    missing: list[SourceLocatorRevision] = []
    has_unproven = False
    for locator in by_observation.values():
        status = _prior_path_status(locator.path, inventory)
        if status is _PriorPathStatus.ABSENT:
            missing.append(locator)
        elif status is _PriorPathStatus.UNPROVEN:
            has_unproven = True

    if has_unproven or len(missing) > 1:
        return _continuity_failure(ordinal=ordinal, snapshot=snapshot)
    if not missing:
        return None
    selected = missing[0]
    return SourceObservationAssignment(
        source_observation_id=selected.source_observation_id,
        match_kind=match_kind,
        previous_locator=selected,
    )


def _identifier_values(
    identifiers: tuple[ExternalIdentifier, ...],
) -> dict[str, frozenset[str]]:
    values: dict[str, set[str]] = {}
    for identifier in identifiers:
        values.setdefault(identifier.namespace, set()).add(identifier.value)
    return {namespace: frozenset(items) for namespace, items in values.items()}


def _identifiers_are_compatible(
    supplied: tuple[ExternalIdentifier, ...],
    candidate: DocumentVersion,
    reader: MetadataReader,
    corpus_id: CorpusId,
) -> bool:
    candidate_identifiers = list(candidate.external_identifiers)
    candidate_identifiers.extend(
        assertion.external_identifier
        for assertion in reader.list_external_identifier_assertions(
            corpus_id,
            candidate.document_version_id,
        )
    )
    supplied_values = _identifier_values(supplied)
    candidate_values = _identifier_values(tuple(candidate_identifiers))
    return all(
        supplied_values[namespace] == candidate_values[namespace]
        for namespace in supplied_values.keys() & candidate_values.keys()
    )


def _hash_match_locators(
    snapshot: VerifiedSourceSnapshot,
    *,
    corpus_id: CorpusId,
    external_identifiers: tuple[ExternalIdentifier, ...],
    reader: MetadataReader,
) -> tuple[SourceLocatorRevision, ...]:
    locators: dict[SourceObservationId, SourceLocatorRevision] = {}
    candidates = reader.find_document_versions_by_content_hash(
        corpus_id,
        snapshot.content_hash,
    )
    for candidate in candidates:
        if not _identifiers_are_compatible(
            external_identifiers,
            candidate,
            reader,
            corpus_id,
        ):
            continue
        observations = reader.find_source_observations_by_document_version(
            corpus_id,
            candidate.document_version_id,
        )
        for observation in observations:
            locator = reader.get_source_locator(
                corpus_id,
                observation.source_observation_id,
            )
            if locator is not None:
                locators[observation.source_observation_id] = locator
    return tuple(
        sorted(
            locators.values(),
            key=lambda locator: locator.source_observation_id.as_str(),
        )
    )


def _new_assignment(
    allocator: SourceObservationIdAllocator,
    *,
    corpus_id: CorpusId,
    reader: MetadataReader,
) -> SourceObservationAssignment:
    source_observation_id = allocator.new_source_observation_id()
    if not isinstance(source_observation_id, SourceObservationId):
        raise TypeError(
            "allocator.new_source_observation_id() must return a SourceObservationId."
        )
    if reader.get_source_observation(corpus_id, source_observation_id) is not None:
        raise IdentityIdCollisionError(
            "allocator returned an existing SourceObservationId."
        )
    return SourceObservationAssignment(
        source_observation_id=source_observation_id,
        match_kind=ObservationMatchKind.NEW,
    )


def resolve_source_observation(
    snapshot: VerifiedSourceSnapshot,
    *,
    ordinal: int,
    corpus_id: CorpusId,
    external_identifiers: tuple[ExternalIdentifier, ...],
    inventory: TraversalInventory,
    reader: MetadataReader,
    allocator: SourceObservationIdAllocator,
) -> SourceObservationAssignment | DiscoveryFailure:
    """Resolve one verified source using only target-corpus and scan evidence."""
    if not isinstance(snapshot, VerifiedSourceSnapshot):
        raise TypeError("snapshot must be a VerifiedSourceSnapshot.")
    if not is_strict_integer(ordinal):
        raise TypeError("ordinal must be an integer.")
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    validate_external_identifiers(external_identifiers)
    if not isinstance(inventory, TraversalInventory):
        raise TypeError("inventory must be a TraversalInventory.")
    if len(inventory.root_scopes) != inventory.root_count:
        raise ValueError("inventory must retain every resolved root scope.")
    if not callable(getattr(allocator, "new_source_observation_id", None)):
        raise TypeError("allocator must provide new_source_observation_id().")

    exact = reader.find_source_locator_by_path(corpus_id, snapshot.locator.path)
    if exact is not None and _exact_evidence_is_compatible(exact, snapshot):
        return SourceObservationAssignment(
            source_observation_id=exact.source_observation_id,
            match_kind=ObservationMatchKind.EXACT_LOCATOR,
            previous_locator=exact,
        )

    current_hint = _identity_hint(snapshot)
    if current_hint is not None:
        identity_match = _select_missing_locator(
            reader.find_source_locators_by_identity_hint(corpus_id, current_hint),
            ordinal=ordinal,
            snapshot=snapshot,
            inventory=inventory,
            match_kind=ObservationMatchKind.FILESYSTEM_IDENTITY,
        )
        if identity_match is not None:
            return identity_match

    hash_match = _select_missing_locator(
        _hash_match_locators(
            snapshot,
            corpus_id=corpus_id,
            external_identifiers=external_identifiers,
            reader=reader,
        ),
        ordinal=ordinal,
        snapshot=snapshot,
        inventory=inventory,
        match_kind=ObservationMatchKind.CONTENT_HASH,
    )
    if hash_match is not None:
        return hash_match
    return _new_assignment(allocator, corpus_id=corpus_id, reader=reader)
