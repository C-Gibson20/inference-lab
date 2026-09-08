"""Per-source identity resolution and persistence coordination."""

from dataclasses import dataclass
from types import TracebackType
from typing import Final, Protocol, Self

from structlog.contextvars import bound_contextvars

from inference_lab.ingest.classification import ClassifiedSourceCandidate
from inference_lab.ingest.discovery import DiscoveryRequest, VerifiedSourceSnapshot
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
)
from inference_lab.ingest.discovery_logging import (
    emit_discovery_event,
    failure_log_fields,
)
from inference_lab.ingest.identity_application import (
    IdentityApplicationResult,
    IdentityApplicationTransaction,
    apply_discovery_identity_outcome,
)
from inference_lab.ingest.identity_resolution import (
    DiscoveryIdentityIdAllocator,
    IdentityResolutionReader,
    resolve_discovered_source_identity,
)
from inference_lab.ingest.traversal import TraversalInventory
from inference_lab.provenance import IdentityIdCollisionError
from inference_lab.storage import (
    StorageConcurrencyError,
    StorageConflictError,
)

_MAX_IDENTITY_ATTEMPTS: Final = 2


class IdentityPersistenceTransaction(
    IdentityResolutionReader,
    IdentityApplicationTransaction,
    Protocol,
):
    """Provide the context-managed capabilities used by identity persistence."""

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...


class IdentityPersistenceRepository(Protocol):
    """Create atomic identity-persistence transactions."""

    def transaction(self) -> IdentityPersistenceTransaction:
        """Return a transaction that starts on context entry."""
        ...


@dataclass(frozen=True, slots=True)
class IdentityPersistenceUnit:
    """Pair one classified source with its verified bytes snapshot."""

    source: ClassifiedSourceCandidate
    snapshot: VerifiedSourceSnapshot

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the source and snapshot are compatible."""
        if not isinstance(self.source, ClassifiedSourceCandidate):
            raise TypeError("source must be a ClassifiedSourceCandidate.")
        if not isinstance(self.snapshot, VerifiedSourceSnapshot):
            raise TypeError("snapshot must be a VerifiedSourceSnapshot.")
        if self.source.candidate.locator != self.snapshot.locator:
            raise ValueError("snapshot locator must match the classified candidate.")

    @property
    def ordinal(self) -> int:
        """Return the source's deterministic discovery position."""
        return self.source.candidate.ordinal


def _persistence_failure(
    source: ClassifiedSourceCandidate,
    category: DiscoveryFailureCategory,
) -> DiscoveryFailure:
    return DiscoveryFailure.for_category(
        ordinal=source.candidate.ordinal,
        category=category,
        locator=source.candidate.locator,
        file_kind=DiscoveryFileKind.REGULAR_FILE,
        media_type=source.classification.media_type,
    )


def _attempt_identity_persistence(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    *,
    request: DiscoveryRequest,
    inventory: TraversalInventory,
    repository: IdentityPersistenceRepository,
    allocator: DiscoveryIdentityIdAllocator,
) -> IdentityApplicationResult | DiscoveryFailure:
    with repository.transaction() as transaction:
        resolved = resolve_discovered_source_identity(
            source,
            snapshot,
            request=request,
            inventory=inventory,
            transaction=transaction,
            allocator=allocator,
        )
        if isinstance(resolved, DiscoveryFailure):
            return resolved
        return apply_discovery_identity_outcome(
            resolved,
            corpus_id=request.corpus_id,
            transaction=transaction,
        )
    raise AssertionError("metadata transaction suppressed identity persistence")


def _log_retry(
    error: IdentityIdCollisionError | StorageConcurrencyError | StorageConflictError,
) -> None:
    error_category = (
        DiscoveryFailureCategory.IDENTITY_WRITE_FAILED.value
        if isinstance(error, IdentityIdCollisionError)
        else error.category.value
    )
    emit_discovery_event(
        __name__,
        "warning",
        "identity_persistence_retry",
        error_category=error_category,
        retry_number=1,
    )


def _log_failure(failure: DiscoveryFailure) -> None:
    emit_discovery_event(
        __name__,
        "error",
        "identity_persistence_failed",
        **failure_log_fields(failure),
    )


def _log_completion(
    result: IdentityApplicationResult,
    *,
    attempt_count: int,
) -> None:
    source = result.source
    outcome = source.identity_outcome
    document_id = None
    if outcome.document is not None:
        document_id = outcome.document.document_id.as_str()
    with bound_contextvars(document_id=document_id):
        emit_discovery_event(
            __name__,
            "info",
            "identity_persistence_completed",
            source_ordinal=source.ordinal,
            source_observation_id=source.source_observation_id.as_str(),
            outcome_kind=outcome.kind.value,
            outcome_reason=outcome.reason.value,
            created_document_version=result.created_document_version,
            attempt_count=attempt_count,
        )


def persist_discovered_source_identity(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    *,
    request: DiscoveryRequest,
    inventory: TraversalInventory,
    repository: IdentityPersistenceRepository,
    allocator: DiscoveryIdentityIdAllocator,
) -> IdentityApplicationResult | DiscoveryFailure:
    """Resolve and persist one source, rebuilding once after a write conflict."""
    if not isinstance(request, DiscoveryRequest):
        raise TypeError("request must be a DiscoveryRequest.")
    if not isinstance(inventory, TraversalInventory):
        raise TypeError("inventory must be a TraversalInventory.")
    if inventory.corpus_id != request.corpus_id:
        raise ValueError("inventory corpus must match the discovery request.")
    if not callable(getattr(repository, "transaction", None)):
        raise TypeError("repository must provide transaction().")

    with bound_contextvars(
        corpus_id=request.corpus_id.as_str(),
        corpus_version=None,
        document_id=None,
    ):
        for attempt in range(_MAX_IDENTITY_ATTEMPTS):
            try:
                result = _attempt_identity_persistence(
                    source,
                    snapshot,
                    request=request,
                    inventory=inventory,
                    repository=repository,
                    allocator=allocator,
                )
            except (
                IdentityIdCollisionError,
                StorageConcurrencyError,
                StorageConflictError,
            ) as error:
                if attempt + 1 < _MAX_IDENTITY_ATTEMPTS:
                    _log_retry(error)
                    continue
                category = (
                    DiscoveryFailureCategory.CONCURRENT_WRITE_CONFLICT
                    if isinstance(error, StorageConcurrencyError)
                    else DiscoveryFailureCategory.IDENTITY_WRITE_FAILED
                )
                failure = _persistence_failure(source, category)
                _log_failure(failure)
                return failure
            if isinstance(result, IdentityApplicationResult):
                _log_completion(result, attempt_count=attempt + 1)
            return result

    raise AssertionError("identity attempt loop must return a result")


def _validate_persistence_units(units: object) -> None:
    if not isinstance(units, tuple):
        raise TypeError("units must be a tuple.")
    if any(not isinstance(unit, IdentityPersistenceUnit) for unit in units):
        raise TypeError("units must contain IdentityPersistenceUnit values.")
    ordinals = tuple(unit.ordinal for unit in units)
    if ordinals != tuple(sorted(set(ordinals))):
        raise ValueError("persistence units must have unique ordered ordinals.")


def persist_discovered_source_identities(
    units: tuple[IdentityPersistenceUnit, ...],
    *,
    request: DiscoveryRequest,
    inventory: TraversalInventory,
    repository: IdentityPersistenceRepository,
    allocator: DiscoveryIdentityIdAllocator,
) -> tuple[IdentityApplicationResult | DiscoveryFailure, ...]:
    """Persist ordered sources as independently atomic identity units."""
    _validate_persistence_units(units)
    if not isinstance(request, DiscoveryRequest):
        raise TypeError("request must be a DiscoveryRequest.")
    if not isinstance(inventory, TraversalInventory):
        raise TypeError("inventory must be a TraversalInventory.")
    if inventory.corpus_id != request.corpus_id:
        raise ValueError("inventory corpus must match the discovery request.")
    if not callable(getattr(repository, "transaction", None)):
        raise TypeError("repository must provide transaction().")

    return tuple(
        persist_discovered_source_identity(
            unit.source,
            unit.snapshot,
            request=request,
            inventory=inventory,
            repository=repository,
            allocator=allocator,
        )
        for unit in units
    )
