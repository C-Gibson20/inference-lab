"""Transactional identity resolution for one verified discovered source."""

from typing import Protocol

from structlog.contextvars import bound_contextvars

from inference_lab.ingest.classification import ClassifiedSourceCandidate
from inference_lab.ingest.continuity import (
    SourceObservationIdAllocator,
    resolve_source_observation,
)
from inference_lab.ingest.discovery import (
    DiscoveredSource,
    DiscoveryRequest,
    SourceSupport,
    VerifiedSourceSnapshot,
)
from inference_lab.ingest.discovery_failures import DiscoveryFailure
from inference_lab.ingest.traversal import TraversalInventory
from inference_lab.provenance import (
    DocumentId,
    DocumentVersion,
    DocumentVersionId,
    ExternalIdentifierAssertion,
    IdentityIdAllocator,
    IdentityResolutionRequest,
    SourceObservationId,
    resolve_identity,
)
from inference_lab.storage import CorruptStorageError, MetadataTransaction


class DiscoveryIdentityIdAllocator(
    SourceObservationIdAllocator,
    IdentityIdAllocator,
    Protocol,
):
    """Allocate every opaque identity used during source discovery."""


class UuidDiscoveryIdentityIdAllocator:
    """Allocate production UUID-backed discovery identities."""

    def new_source_observation_id(self) -> SourceObservationId:
        """Allocate one new source-observation ID."""
        return SourceObservationId.generate()

    def new_document_id(self) -> DocumentId:
        """Allocate one new logical-document ID."""
        return DocumentId.generate()

    def new_document_version_id(self) -> DocumentVersionId:
        """Allocate one new immutable source-version ID."""
        return DocumentVersionId.generate()


def _continuity_candidates(
    source_observation_id: SourceObservationId,
    *,
    request: DiscoveryRequest,
    transaction: MetadataTransaction,
) -> tuple[DocumentVersion, ...]:
    state = transaction.get_source_observation(
        request.corpus_id,
        source_observation_id,
    )
    if state is None:
        return ()
    candidate = transaction.get_document_version(state.document_version_id)
    if candidate is None:
        raise CorruptStorageError()
    if candidate.document_id != state.document_id:
        raise CorruptStorageError()
    return (candidate,)


def _identifier_assertions(
    candidate_roles: tuple[tuple[DocumentVersion, ...], ...],
    *,
    request: DiscoveryRequest,
    transaction: MetadataTransaction,
) -> tuple[ExternalIdentifierAssertion, ...]:
    candidates = {
        candidate.document_version_id: candidate
        for role in candidate_roles
        for candidate in role
    }
    assertions = {
        assertion.key(): assertion
        for document_version_id in sorted(
            candidates,
            key=DocumentVersionId.as_str,
        )
        for assertion in transaction.list_external_identifier_assertions(
            request.corpus_id,
            document_version_id,
        )
    }
    return tuple(assertions[key] for key in sorted(assertions))


def _resolution_request(
    source_observation_id: SourceObservationId,
    snapshot: VerifiedSourceSnapshot,
    source: ClassifiedSourceCandidate,
    *,
    request: DiscoveryRequest,
    transaction: MetadataTransaction,
) -> IdentityResolutionRequest:
    continuity_candidates = _continuity_candidates(
        source_observation_id,
        request=request,
        transaction=transaction,
    )
    content_hash_candidates = transaction.find_document_versions_by_content_hash(
        request.corpus_id,
        snapshot.content_hash,
    )
    external_identifier_candidates = (
        transaction.find_document_versions_by_external_identifiers(
            request.corpus_id,
            source.candidate.external_identifiers,
        )
    )
    identifier_assertions = _identifier_assertions(
        (
            continuity_candidates,
            content_hash_candidates,
            external_identifier_candidates,
        ),
        request=request,
        transaction=transaction,
    )
    return IdentityResolutionRequest(
        content_hash=snapshot.content_hash,
        source_observation_id=source_observation_id,
        external_identifiers=source.candidate.external_identifiers,
        continuity_candidates=continuity_candidates,
        content_hash_candidates=content_hash_candidates,
        external_identifier_candidates=external_identifier_candidates,
        identifier_assertions=identifier_assertions,
        external_identifier_policy=request.external_identifier_policy,
    )


def _validate_source_snapshot(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    inventory: TraversalInventory,
) -> None:
    if source.classification.support is not SourceSupport.SUPPORTED:
        raise ValueError("source must have a supported classification.")
    if snapshot.locator != source.candidate.locator:
        raise ValueError("snapshot locator must match the classified candidate.")
    if source.candidate not in inventory.candidates:
        raise ValueError("source must belong to the supplied traversal inventory.")
    if snapshot.file_evidence.pre_open != source.candidate.stat_evidence:
        raise ValueError("snapshot evidence must match the traversal candidate.")
    if snapshot.media_type != source.classification.media_type:
        raise ValueError("snapshot media type must match the classification.")
    if snapshot.encoding != source.classification.encoding:
        raise ValueError("snapshot encoding must match the classification.")


def _resolve_with_bound_corpus(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    *,
    request: DiscoveryRequest,
    inventory: TraversalInventory,
    transaction: MetadataTransaction,
    allocator: DiscoveryIdentityIdAllocator,
) -> DiscoveredSource | DiscoveryFailure:
    assignment = resolve_source_observation(
        snapshot,
        ordinal=source.candidate.ordinal,
        corpus_id=request.corpus_id,
        external_identifiers=source.candidate.external_identifiers,
        inventory=inventory,
        reader=transaction,
        allocator=allocator,
    )
    if isinstance(assignment, DiscoveryFailure):
        return assignment

    resolution_request = _resolution_request(
        assignment.source_observation_id,
        snapshot,
        source,
        request=request,
        transaction=transaction,
    )
    outcome = resolve_identity(
        resolution_request,
        allocator=allocator,
        pipeline_stage="ingest",
    )
    return DiscoveredSource(
        ordinal=source.candidate.ordinal,
        source_observation_id=assignment.source_observation_id,
        snapshot=snapshot,
        external_identifiers=source.candidate.external_identifiers,
        family_candidates=source.candidate.family_candidates,
        classification=source.classification,
        identity_outcome=outcome,
    )


def resolve_discovered_source_identity(
    source: ClassifiedSourceCandidate,
    snapshot: VerifiedSourceSnapshot,
    *,
    request: DiscoveryRequest,
    inventory: TraversalInventory,
    transaction: MetadataTransaction,
    allocator: DiscoveryIdentityIdAllocator,
) -> DiscoveredSource | DiscoveryFailure:
    """Resolve one source inside a caller-owned metadata transaction."""
    if not isinstance(source, ClassifiedSourceCandidate):
        raise TypeError("source must be a ClassifiedSourceCandidate.")
    if not isinstance(snapshot, VerifiedSourceSnapshot):
        raise TypeError("snapshot must be a VerifiedSourceSnapshot.")
    if not isinstance(request, DiscoveryRequest):
        raise TypeError("request must be a DiscoveryRequest.")
    if not isinstance(inventory, TraversalInventory):
        raise TypeError("inventory must be a TraversalInventory.")
    for method_name in (
        "new_source_observation_id",
        "new_document_id",
        "new_document_version_id",
    ):
        if not callable(getattr(allocator, method_name, None)):
            raise TypeError(f"allocator must provide {method_name}().")
    _validate_source_snapshot(source, snapshot, inventory)

    with bound_contextvars(corpus_id=request.corpus_id.as_str()):
        return _resolve_with_bound_corpus(
            source,
            snapshot,
            request=request,
            inventory=inventory,
            transaction=transaction,
            allocator=allocator,
        )
