"""Deterministic document identity resolution."""

from typing import Protocol

import structlog

from inference_lab.provenance._resolver_context import _ResolutionContext
from inference_lab.provenance._validation import is_lowercase_slug
from inference_lab.provenance.identifiers import DocumentId, DocumentVersionId
from inference_lab.provenance.records import DocumentIdentity, DocumentVersion
from inference_lab.provenance.relationships import DuplicateRelationship
from inference_lab.provenance.resolution import (
    ExternalIdentifierAssertion,
    IdentityIdCollisionError,
    IdentityResolutionOutcome,
    IdentityResolutionRequest,
    ResolutionReason,
)
from inference_lab.provenance.values import (
    DuplicateClass,
    ResolutionKind,
)

logger = structlog.get_logger(__name__)


class IdentityIdAllocator(Protocol):
    """Allocate opaque IDs for newly observed documents and versions."""

    def new_document_id(self) -> DocumentId:
        """Allocate a new logical document ID."""
        ...

    def new_document_version_id(self) -> DocumentVersionId:
        """Allocate a new immutable source-version ID."""
        ...


class UuidIdentityIdAllocator:
    """Allocate production UUID-backed provenance identifiers."""

    def new_document_id(self) -> DocumentId:
        """Allocate a new logical document ID."""
        return DocumentId.generate()

    def new_document_version_id(self) -> DocumentVersionId:
        """Allocate a new immutable source-version ID."""
        return DocumentVersionId.generate()


def _identifier_assertions_to_add(
    context: _ResolutionContext,
    candidate: DocumentVersion,
) -> tuple[ExternalIdentifierAssertion, ...]:
    """Return append-only assertions that add a new namespace."""
    request = context.request
    existing_namespaces = context.candidate_namespaces(candidate)
    return tuple(
        ExternalIdentifierAssertion(
            source_observation_id=request.source_observation_id,
            document_version_id=candidate.document_version_id,
            external_identifier=identifier,
        )
        for identifier in request.external_identifiers
        if identifier.namespace not in existing_namespaces
    )


def _allocate_document_id(
    allocator: IdentityIdAllocator,
    existing_document_ids: frozenset[DocumentId],
) -> DocumentId:
    """Allocate and validate a non-colliding document ID."""
    document_id = allocator.new_document_id()
    if not isinstance(document_id, DocumentId):
        raise TypeError("allocator.new_document_id() must return a DocumentId.")
    if document_id in existing_document_ids:
        raise IdentityIdCollisionError("allocator returned an existing DocumentId.")
    return document_id


def _new_version(
    context: _ResolutionContext,
    *,
    document_id: DocumentId,
    allocator: IdentityIdAllocator,
) -> DocumentVersion:
    """Create and validate an immutable version for the incoming source."""
    request = context.request
    version_id = allocator.new_document_version_id()
    if not isinstance(version_id, DocumentVersionId):
        raise TypeError(
            "allocator.new_document_version_id() must return a DocumentVersionId."
        )
    if version_id in context.existing_version_ids:
        raise IdentityIdCollisionError(
            "allocator returned an existing DocumentVersionId."
        )
    return DocumentVersion(
        document_version_id=version_id,
        document_id=document_id,
        content_hash=request.content_hash,
        external_identifiers=request.external_identifiers,
    )


def _duplicate_relationship(
    resolved_version_id: DocumentVersionId,
    candidate_version_id: DocumentVersionId,
) -> DuplicateRelationship:
    """Create one canonical byte-identical relationship."""
    endpoint_ids = tuple(
        sorted(
            (resolved_version_id, candidate_version_id),
            key=lambda version_id: version_id.as_str(),
        )
    )
    return DuplicateRelationship(
        first_version_id=endpoint_ids[0],
        second_version_id=endpoint_ids[1],
        duplicate_class=DuplicateClass.BYTE_IDENTICAL,
    )


def _resolve_continuity(
    context: _ResolutionContext,
    allocator: IdentityIdAllocator,
) -> IdentityResolutionOutcome | None:
    """Resolve source-continuity evidence before weaker evidence stages."""
    candidates = context.request.continuity_candidates
    if not candidates:
        return None

    matched_version_ids = context.version_ids(candidates)
    if context.has_identifier_conflict(candidates):
        return IdentityResolutionOutcome(
            kind=ResolutionKind.CONFLICT,
            reason=ResolutionReason.CONFLICTING_EVIDENCE,
            matched_version_ids=matched_version_ids,
        )

    matching_document_ids = context.document_ids(candidates)
    if len(matching_document_ids) > 1 or len(context.continuity_hash_matches) > 1:
        return IdentityResolutionOutcome(
            kind=ResolutionKind.AMBIGUOUS,
            reason=ResolutionReason.MULTIPLE_CANDIDATES,
            matched_version_ids=matched_version_ids,
        )

    if context.continuity_hash_matches:
        candidate = context.continuity_hash_matches[0]
        return IdentityResolutionOutcome(
            kind=ResolutionKind.UNCHANGED,
            reason=ResolutionReason.EXACT_HASH_MATCH,
            document=DocumentIdentity(candidate.document_id),
            version=candidate,
            matched_version_ids=(candidate.document_version_id,),
            identifier_assertions_to_add=_identifier_assertions_to_add(
                context, candidate
            ),
        )

    document_id = candidates[0].document_id
    version = _new_version(context, document_id=document_id, allocator=allocator)
    return IdentityResolutionOutcome(
        kind=ResolutionKind.NEW_VERSION,
        reason=ResolutionReason.SOURCE_CONTINUITY_MATCH,
        document=DocumentIdentity(document_id),
        version=version,
        matched_version_ids=matched_version_ids,
    )


def _resolve_duplicate_hash(
    context: _ResolutionContext,
    allocator: IdentityIdAllocator,
) -> IdentityResolutionOutcome | None:
    """Resolve byte-identical evidence after source continuity."""
    candidates = context.request.content_hash_candidates
    if not candidates:
        return None

    document_id = _allocate_document_id(allocator, context.existing_document_ids)
    version = _new_version(context, document_id=document_id, allocator=allocator)
    relationships = tuple(
        _duplicate_relationship(
            version.document_version_id,
            candidate.document_version_id,
        )
        for candidate in candidates
    )
    return IdentityResolutionOutcome(
        kind=ResolutionKind.DUPLICATE,
        reason=ResolutionReason.DUPLICATE_HASH_MATCH,
        document=DocumentIdentity(document_id),
        version=version,
        matched_version_ids=context.version_ids(candidates),
        duplicate_relationships=relationships,
    )


def _resolve_external_identifier(
    context: _ResolutionContext,
    allocator: IdentityIdAllocator,
) -> IdentityResolutionOutcome | None:
    """Resolve external-identifier evidence after stronger evidence stages."""
    candidates = context.request.external_identifier_candidates
    if context.has_identifier_conflict(candidates):
        return IdentityResolutionOutcome(
            kind=ResolutionKind.CONFLICT,
            reason=ResolutionReason.CONFLICTING_EVIDENCE,
            matched_version_ids=context.version_ids(candidates),
        )

    identifier_matches = context.identifier_matches
    matching_document_ids = context.document_ids(identifier_matches)
    matched_version_ids = context.version_ids(identifier_matches)
    if len(matching_document_ids) > 1:
        return IdentityResolutionOutcome(
            kind=ResolutionKind.AMBIGUOUS,
            reason=ResolutionReason.MULTIPLE_CANDIDATES,
            matched_version_ids=matched_version_ids,
        )

    if not matching_document_ids:
        return None

    matched_namespaces = {
        namespace
        for candidate in identifier_matches
        for namespace in context.matching_namespaces_for(candidate)
    }
    authoritative_namespaces = set(
        context.request.external_identifier_policy.authoritative_namespaces
    )
    if not matched_namespaces & authoritative_namespaces:
        return IdentityResolutionOutcome(
            kind=ResolutionKind.AMBIGUOUS,
            reason=ResolutionReason.UNESTABLISHED_IDENTIFIER_AUTHORITY,
            matched_version_ids=matched_version_ids,
        )

    document_id = matching_document_ids[0]
    version = _new_version(context, document_id=document_id, allocator=allocator)
    return IdentityResolutionOutcome(
        kind=ResolutionKind.NEW_VERSION,
        reason=ResolutionReason.EXTERNAL_IDENTIFIER_MATCH,
        document=DocumentIdentity(document_id),
        version=version,
        matched_version_ids=matched_version_ids,
    )


def _resolve_new_document(
    context: _ResolutionContext,
    allocator: IdentityIdAllocator,
) -> IdentityResolutionOutcome:
    """Create a new identity after every evidence stage produces no match."""
    document_id = _allocate_document_id(allocator, context.existing_document_ids)
    version = _new_version(context, document_id=document_id, allocator=allocator)
    return IdentityResolutionOutcome(
        kind=ResolutionKind.NEW_DOCUMENT,
        reason=ResolutionReason.NO_MATCH,
        document=DocumentIdentity(document_id),
        version=version,
    )


def _resolve_by_precedence(
    context: _ResolutionContext,
    allocator: IdentityIdAllocator,
) -> IdentityResolutionOutcome:
    """Return the first outcome selected by the evidence precedence rules."""
    outcome = _resolve_continuity(context, allocator)
    if outcome is not None:
        return outcome

    outcome = _resolve_duplicate_hash(context, allocator)
    if outcome is not None:
        return outcome

    outcome = _resolve_external_identifier(context, allocator)
    if outcome is not None:
        return outcome

    return _resolve_new_document(context, allocator)


def _validate_allocator(allocator: object) -> None:
    """Validate the callable allocator boundary before resolution."""
    if not callable(getattr(allocator, "new_document_id", None)):
        raise TypeError("allocator must provide new_document_id().")
    if not callable(getattr(allocator, "new_document_version_id", None)):
        raise TypeError("allocator must provide new_document_version_id().")


def _validate_pipeline_stage(pipeline_stage: str) -> None:
    """Validate the pipeline stage string before resolution."""
    if not isinstance(pipeline_stage, str):
        raise TypeError("pipeline_stage must be a string.")
    if not is_lowercase_slug(pipeline_stage):
        raise ValueError("pipeline_stage must be a lowercase slug.")


def _log_resolution_completed(
    *,
    outcome: IdentityResolutionOutcome,
    candidate_count: int,
    pipeline_stage: str,
) -> None:
    """Log one privacy-safe event for a completed resolution."""
    logger.info(
        "identity_resolution_completed",
        pipeline_stage=pipeline_stage,
        document_id=(
            outcome.document.document_id.as_str()
            if outcome.document is not None
            else None
        ),
        resolution_kind=outcome.kind.value,
        resolution_reason=outcome.reason.value,
        matched_candidate_count=len(outcome.matched_version_ids),
        duplicate_relationship_count=len(outcome.duplicate_relationships),
        candidate_count=candidate_count,
    )


def _log_resolution_failed(
    *,
    error: Exception,
    candidate_count: int,
    pipeline_stage: str,
) -> None:
    """Log one privacy-safe event for a failed resolution operation."""
    logger.error(
        "identity_resolution_failed",
        pipeline_stage=pipeline_stage,
        failure_type=type(error).__name__,
        candidate_count=candidate_count,
    )


def resolve_identity(
    request: IdentityResolutionRequest,
    *,
    allocator: IdentityIdAllocator,
    pipeline_stage: str,
) -> IdentityResolutionOutcome:
    """Resolve one source candidate against existing document versions."""
    if not isinstance(request, IdentityResolutionRequest):
        raise TypeError("request must be an IdentityResolutionRequest.")
    _validate_pipeline_stage(pipeline_stage)
    _validate_allocator(allocator)
    context = _ResolutionContext.from_request(request)
    try:
        outcome = _resolve_by_precedence(context, allocator)
    except Exception as error:
        _log_resolution_failed(
            error=error,
            candidate_count=len(context.all_candidates),
            pipeline_stage=pipeline_stage,
        )
        raise

    _log_resolution_completed(
        outcome=outcome,
        candidate_count=len(context.all_candidates),
        pipeline_stage=pipeline_stage,
    )
    return outcome
