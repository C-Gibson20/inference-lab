"""Deterministic document identity resolution."""

from typing import Protocol

import structlog

from inference_lab.provenance.identifiers import DocumentId, DocumentVersionId
from inference_lab.provenance.records import DocumentIdentity, DocumentVersion
from inference_lab.provenance.relationships import DuplicateRelationship
from inference_lab.provenance.resolution import (
    IdentityResolutionOutcome,
    IdentityResolutionRequest,
    InvalidResolutionError,
    ResolutionReason,
)
from inference_lab.provenance.utils import is_lowercase_slug
from inference_lab.provenance.values import DuplicateClass, ResolutionKind

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


def _hash_matches(
    request: IdentityResolutionRequest,
) -> tuple[DocumentVersion, ...]:
    """Return global candidates with exactly matching source hashes."""
    return request.content_hash_candidates


def _identifier_matches(
    request: IdentityResolutionRequest,
) -> tuple[DocumentVersion, ...]:
    """Return candidates sharing at least one external identifier."""
    request_identifiers = set(
        identifier.as_tuple() for identifier in request.external_identifiers
    )
    if not request_identifiers:
        return ()

    return tuple(
        candidate
        for candidate in request.external_identifier_candidates
        if request_identifiers.intersection(
            identifier.as_tuple() for identifier in candidate.external_identifiers
        )
    )


def _matching_document_ids(
    candidates: tuple[DocumentVersion, ...],
) -> tuple[DocumentId, ...]:
    """Return unique matching document IDs in canonical order."""
    return tuple(
        sorted(
            {candidate.document_id for candidate in candidates},
            key=lambda document_id: document_id.as_str(),
        )
    )


def _matched_version_ids(
    candidates: tuple[DocumentVersion, ...],
) -> tuple[DocumentVersionId, ...]:
    """Return unique matched version IDs in canonical order."""
    return tuple(
        sorted(
            {candidate.document_version_id for candidate in candidates},
            key=lambda version_id: version_id.as_str(),
        )
    )


def _conflicting_identifiers(
    request: IdentityResolutionRequest,
    candidate: DocumentVersion,
) -> bool:
    """Return whether shared namespaces contain conflicting values."""
    request_by_namespace = {
        identifier.namespace: identifier.value
        for identifier in request.external_identifiers
    }
    return any(
        request_by_namespace.get(identifier.namespace) is not None
        and request_by_namespace[identifier.namespace] != identifier.value
        for identifier in candidate.external_identifiers
    )


def _allocate_document_id(
    allocator: IdentityIdAllocator,
    existing: tuple[DocumentVersion, ...],
) -> DocumentId:
    """Allocate and validate a non-colliding document ID."""
    document_id = allocator.new_document_id()
    if not isinstance(document_id, DocumentId):
        raise TypeError("allocator.new_document_id() must return a DocumentId.")
    if document_id in {candidate.document_id for candidate in existing}:
        raise InvalidResolutionError("allocator returned an existing DocumentId.")
    return document_id


def _new_version(
    request: IdentityResolutionRequest,
    *,
    document_id: DocumentId,
    allocator: IdentityIdAllocator,
) -> DocumentVersion:
    """Create and validate an immutable version for the incoming source."""
    version_id = allocator.new_document_version_id()
    if not isinstance(version_id, DocumentVersionId):
        raise TypeError(
            "allocator.new_document_version_id() must return a DocumentVersionId."
        )
    if any(
        candidate.document_version_id == version_id
        for candidate in request.all_candidates
    ):
        raise InvalidResolutionError(
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

    def complete(outcome: IdentityResolutionOutcome) -> IdentityResolutionOutcome:
        _log_resolution_completed(
            outcome=outcome,
            candidate_count=len(request.all_candidates),
            pipeline_stage=pipeline_stage,
        )
        return outcome

    try:
        continuity_candidates = request.continuity_candidates
        continuity_hash_matches = tuple(
            candidate
            for candidate in continuity_candidates
            if candidate.content_hash == request.content_hash
        )
        if continuity_candidates:
            if any(
                _conflicting_identifiers(request, candidate)
                for candidate in continuity_candidates
            ):
                return complete(
                    IdentityResolutionOutcome(
                        kind=ResolutionKind.CONFLICT,
                        reason=ResolutionReason.CONFLICTING_EVIDENCE,
                        matched_version_ids=_matched_version_ids(continuity_candidates),
                    )
                )

            matching_documents = _matching_document_ids(continuity_candidates)
            if len(matching_documents) > 1 or len(continuity_hash_matches) > 1:
                return complete(
                    IdentityResolutionOutcome(
                        kind=ResolutionKind.AMBIGUOUS,
                        reason=ResolutionReason.MULTIPLE_CANDIDATES,
                        matched_version_ids=_matched_version_ids(continuity_candidates),
                    )
                )

            if continuity_hash_matches:
                candidate = continuity_hash_matches[0]
                return complete(
                    IdentityResolutionOutcome(
                        kind=ResolutionKind.UNCHANGED,
                        reason=ResolutionReason.EXACT_HASH_MATCH,
                        document=DocumentIdentity(candidate.document_id),
                        version=candidate,
                        matched_version_ids=(candidate.document_version_id,),
                    )
                )

            document_id = continuity_candidates[0].document_id
            version = _new_version(
                request, document_id=document_id, allocator=allocator
            )
            return complete(
                IdentityResolutionOutcome(
                    kind=ResolutionKind.NEW_VERSION,
                    reason=ResolutionReason.SOURCE_CONTINUITY_MATCH,
                    document=DocumentIdentity(document_id),
                    version=version,
                    matched_version_ids=_matched_version_ids(continuity_candidates),
                )
            )

        hash_matches = _hash_matches(request)
        if hash_matches:
            document_id = _allocate_document_id(allocator, request.all_candidates)
            version = _new_version(
                request, document_id=document_id, allocator=allocator
            )
            relationships = tuple(
                _duplicate_relationship(
                    version.document_version_id,
                    candidate.document_version_id,
                )
                for candidate in hash_matches
            )
            return complete(
                IdentityResolutionOutcome(
                    kind=ResolutionKind.DUPLICATE,
                    reason=ResolutionReason.DUPLICATE_HASH_MATCH,
                    document=DocumentIdentity(document_id),
                    version=version,
                    matched_version_ids=_matched_version_ids(hash_matches),
                    duplicate_relationships=relationships,
                )
            )

        identifier_matches = _identifier_matches(request)
        matching_documents = _matching_document_ids(identifier_matches)
        if len(matching_documents) > 1:
            return complete(
                IdentityResolutionOutcome(
                    kind=ResolutionKind.AMBIGUOUS,
                    reason=ResolutionReason.MULTIPLE_CANDIDATES,
                    matched_version_ids=_matched_version_ids(identifier_matches),
                )
            )

        if matching_documents:
            document_id = matching_documents[0]
            version = _new_version(
                request, document_id=document_id, allocator=allocator
            )
            return complete(
                IdentityResolutionOutcome(
                    kind=ResolutionKind.NEW_VERSION,
                    reason=ResolutionReason.EXTERNAL_IDENTIFIER_MATCH,
                    document=DocumentIdentity(document_id),
                    version=version,
                    matched_version_ids=_matched_version_ids(identifier_matches),
                )
            )

        document_id = _allocate_document_id(allocator, request.all_candidates)
        version = _new_version(request, document_id=document_id, allocator=allocator)
        return complete(
            IdentityResolutionOutcome(
                kind=ResolutionKind.NEW_DOCUMENT,
                reason=ResolutionReason.NO_MATCH,
                document=DocumentIdentity(document_id),
                version=version,
            )
        )
    except Exception as error:
        _log_resolution_failed(
            error=error,
            candidate_count=len(request.all_candidates),
            pipeline_stage=pipeline_stage,
        )
        raise
