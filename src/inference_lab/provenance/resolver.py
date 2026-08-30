"""Deterministic document identity resolution."""

from typing import Protocol

from inference_lab.provenance.identifiers import DocumentId, DocumentVersionId
from inference_lab.provenance.records import DocumentIdentity, DocumentVersion
from inference_lab.provenance.relationships import DuplicateRelationship
from inference_lab.provenance.resolution import (
    IdentityResolutionOutcome,
    IdentityResolutionRequest,
    InvalidResolutionError,
    ResolutionReason,
)
from inference_lab.provenance.values import DuplicateClass, ResolutionKind


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
    """Return candidates with exactly matching source hashes."""
    return tuple(
        candidate
        for candidate in request.candidates
        if candidate.content_hash == request.content_hash
    )


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
        for candidate in request.candidates
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
        candidate.document_version_id == version_id for candidate in request.candidates
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


def resolve_identity(
    request: IdentityResolutionRequest,
    *,
    allocator: IdentityIdAllocator,
) -> IdentityResolutionOutcome:
    """Resolve one source candidate against existing document versions."""
    if not isinstance(request, IdentityResolutionRequest):
        raise TypeError("request must be an IdentityResolutionRequest.")
    _validate_allocator(allocator)

    hash_matches = _hash_matches(request)
    if hash_matches:
        if any(
            _conflicting_identifiers(request, candidate) for candidate in hash_matches
        ):
            return IdentityResolutionOutcome(
                kind=ResolutionKind.CONFLICT,
                reason=ResolutionReason.CONFLICTING_EVIDENCE,
                matched_version_ids=tuple(
                    candidate.document_version_id for candidate in hash_matches
                ),
            )

        if len(hash_matches) == 1:
            candidate = hash_matches[0]
            return IdentityResolutionOutcome(
                kind=ResolutionKind.UNCHANGED,
                reason=ResolutionReason.EXACT_HASH_MATCH,
                document=DocumentIdentity(candidate.document_id),
                version=candidate,
                matched_version_ids=(candidate.document_version_id,),
            )

        document_id = _allocate_document_id(allocator, request.candidates)
        version = _new_version(request, document_id=document_id, allocator=allocator)
        relationships = tuple(
            _duplicate_relationship(
                version.document_version_id,
                candidate.document_version_id,
            )
            for candidate in hash_matches
        )
        return IdentityResolutionOutcome(
            kind=ResolutionKind.DUPLICATE,
            reason=ResolutionReason.DUPLICATE_HASH_MATCH,
            document=DocumentIdentity(document_id),
            version=version,
            matched_version_ids=tuple(
                candidate.document_version_id for candidate in hash_matches
            ),
            duplicate_relationships=relationships,
        )

    identifier_matches = _identifier_matches(request)
    matching_documents = _matching_document_ids(identifier_matches)
    if len(matching_documents) > 1:
        return IdentityResolutionOutcome(
            kind=ResolutionKind.AMBIGUOUS,
            reason=ResolutionReason.MULTIPLE_CANDIDATES,
            matched_version_ids=tuple(
                candidate.document_version_id for candidate in identifier_matches
            ),
        )

    if matching_documents:
        document_id = matching_documents[0]
        version = _new_version(request, document_id=document_id, allocator=allocator)
        return IdentityResolutionOutcome(
            kind=ResolutionKind.NEW_VERSION,
            reason=ResolutionReason.EXTERNAL_IDENTIFIER_MATCH,
            document=DocumentIdentity(document_id),
            version=version,
            matched_version_ids=tuple(
                candidate.document_version_id for candidate in identifier_matches
            ),
        )

    document_id = _allocate_document_id(allocator, request.candidates)
    version = _new_version(request, document_id=document_id, allocator=allocator)
    return IdentityResolutionOutcome(
        kind=ResolutionKind.NEW_DOCUMENT,
        reason=ResolutionReason.NO_MATCH,
        document=DocumentIdentity(document_id),
        version=version,
    )
