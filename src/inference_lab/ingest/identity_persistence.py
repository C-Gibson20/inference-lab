"""Per-source identity resolution and persistence coordination."""

from typing import Final

import structlog
from structlog.contextvars import bound_contextvars

from inference_lab.ingest.classification import ClassifiedSourceCandidate
from inference_lab.ingest.discovery import DiscoveryRequest, VerifiedSourceSnapshot
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
)
from inference_lab.ingest.identity_application import (
    IdentityApplicationResult,
    apply_discovery_identity_outcome,
)
from inference_lab.ingest.identity_resolution import (
    DiscoveryIdentityIdAllocator,
    resolve_discovered_source_identity,
)
from inference_lab.ingest.traversal import TraversalInventory
from inference_lab.storage import (
    MetadataRepository,
    StorageConcurrencyError,
    StorageConflictError,
)

logger = structlog.get_logger(__name__)

_MAX_IDENTITY_ATTEMPTS: Final = 2


def _safe_log(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded identity metadata without changing persistence outcomes."""
    try:
        getattr(logger, method_name)(event, pipeline_stage="ingest", **fields)
    except Exception:
        pass


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
    repository: MetadataRepository,
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


def _log_retry(error: StorageConcurrencyError | StorageConflictError) -> None:
    _safe_log(
        "warning",
        "identity_persistence_retry",
        error_category=error.category.value,
        retry_number=1,
    )


def _log_failure(failure: DiscoveryFailure) -> None:
    _safe_log(
        "error",
        "identity_persistence_failed",
        error_category=failure.category.value,
        retryable=failure.retryable,
        source_ordinal=failure.ordinal,
    )


def persist_discovered_source_identity(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    *,
    request: DiscoveryRequest,
    inventory: TraversalInventory,
    repository: MetadataRepository,
    allocator: DiscoveryIdentityIdAllocator,
) -> IdentityApplicationResult | DiscoveryFailure:
    """Resolve and persist one source, rebuilding once after a write conflict."""
    if not isinstance(request, DiscoveryRequest):
        raise TypeError("request must be a DiscoveryRequest.")
    if not callable(getattr(repository, "transaction", None)):
        raise TypeError("repository must provide transaction().")

    with bound_contextvars(corpus_id=request.corpus_id.as_str()):
        for attempt in range(_MAX_IDENTITY_ATTEMPTS):
            try:
                return _attempt_identity_persistence(
                    source,
                    snapshot,
                    request=request,
                    inventory=inventory,
                    repository=repository,
                    allocator=allocator,
                )
            except (StorageConcurrencyError, StorageConflictError) as error:
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

    raise AssertionError("identity attempt loop must return a result")
