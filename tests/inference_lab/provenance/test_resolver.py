"""Tests for deterministic document identity resolution."""

from dataclasses import dataclass

import pytest

from inference_lab.provenance import (
    ContentHash,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateClass,
    ExternalIdentifier,
    HashAlgorithm,
    IdentityResolutionRequest,
    InvalidResolutionError,
    ResolutionKind,
    ResolutionReason,
    resolve_identity,
)

HASH_A = ContentHash(HashAlgorithm.SHA256, "a" * 64)
HASH_B = ContentHash(HashAlgorithm.SHA256, "b" * 64)
HASH_C = ContentHash(HashAlgorithm.SHA256, "c" * 64)
DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000001")
OTHER_DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000002")
FIRST_VERSION_ID = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000003")
SECOND_VERSION_ID = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000004")
THIRD_VERSION_ID = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000005")
ALLOCATED_DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000006")
ALLOCATED_VERSION_ID = DocumentVersionId.from_str(
    "00000000-0000-4000-8000-000000000007"
)


@dataclass
class FixedAllocator:
    """Deterministic allocator for resolver tests."""

    document_id: DocumentId = ALLOCATED_DOCUMENT_ID
    version_id: DocumentVersionId = ALLOCATED_VERSION_ID
    document_calls: int = 0
    version_calls: int = 0

    def new_document_id(self) -> DocumentId:
        self.document_calls += 1
        return self.document_id

    def new_document_version_id(self) -> DocumentVersionId:
        self.version_calls += 1
        return self.version_id


def _version(
    version_id: DocumentVersionId,
    document_id: DocumentId = DOCUMENT_ID,
    content_hash: ContentHash = HASH_A,
    identifiers: tuple[ExternalIdentifier, ...] = (),
) -> DocumentVersion:
    return DocumentVersion(version_id, document_id, content_hash, identifiers)


def test_no_candidates_creates_new_document_and_version() -> None:
    allocator = FixedAllocator()

    outcome = resolve_identity(IdentityResolutionRequest(HASH_A), allocator=allocator)

    assert outcome.kind is ResolutionKind.NEW_DOCUMENT
    assert outcome.reason is ResolutionReason.NO_MATCH
    assert outcome.document == DocumentIdentity(ALLOCATED_DOCUMENT_ID)
    assert outcome.version is not None
    assert outcome.version.document_version_id == ALLOCATED_VERSION_ID
    assert allocator.document_calls == 1
    assert allocator.version_calls == 1


def test_exactly_one_hash_match_is_unchanged_without_allocation() -> None:
    candidate = _version(FIRST_VERSION_ID)
    allocator = FixedAllocator()

    outcome = resolve_identity(
        IdentityResolutionRequest(HASH_A, candidates=(candidate,)),
        allocator=allocator,
    )

    assert outcome.kind is ResolutionKind.UNCHANGED
    assert outcome.version == candidate
    assert outcome.matched_version_ids == (FIRST_VERSION_ID,)
    assert allocator.document_calls == 0
    assert allocator.version_calls == 0


def test_changed_hash_with_one_identifier_match_creates_new_version() -> None:
    identifier = ExternalIdentifier("doi", "10.1000/example")
    candidate = _version(
        FIRST_VERSION_ID, content_hash=HASH_A, identifiers=(identifier,)
    )
    allocator = FixedAllocator()

    outcome = resolve_identity(
        IdentityResolutionRequest(HASH_B, (identifier,), (candidate,)),
        allocator=allocator,
    )

    assert outcome.kind is ResolutionKind.NEW_VERSION
    assert outcome.document == DocumentIdentity(DOCUMENT_ID)
    assert outcome.version is not None
    assert outcome.version.content_hash == HASH_B
    assert outcome.matched_version_ids == (FIRST_VERSION_ID,)


def test_multiple_hash_matches_create_independently_citable_duplicates() -> None:
    first = _version(FIRST_VERSION_ID, DOCUMENT_ID)
    second = _version(SECOND_VERSION_ID, OTHER_DOCUMENT_ID)
    allocator = FixedAllocator()

    outcome = resolve_identity(
        IdentityResolutionRequest(HASH_A, candidates=(first, second)),
        allocator=allocator,
    )

    assert outcome.kind is ResolutionKind.DUPLICATE
    assert outcome.matched_version_ids == (FIRST_VERSION_ID, SECOND_VERSION_ID)
    assert len(outcome.duplicate_relationships) == 2
    assert all(
        relationship.duplicate_class is DuplicateClass.BYTE_IDENTICAL
        for relationship in outcome.duplicate_relationships
    )


def test_multiple_identifier_documents_are_ambiguous() -> None:
    identifier = ExternalIdentifier("doi", "10.1000/example")
    candidates = (
        _version(FIRST_VERSION_ID, DOCUMENT_ID, HASH_A, (identifier,)),
        _version(SECOND_VERSION_ID, OTHER_DOCUMENT_ID, HASH_B, (identifier,)),
    )

    outcome = resolve_identity(
        IdentityResolutionRequest(HASH_C, (identifier,), candidates),
        allocator=FixedAllocator(),
    )

    assert outcome.kind is ResolutionKind.AMBIGUOUS


def test_conflicting_identifier_for_hash_match_fails_closed() -> None:
    candidate_identifier = ExternalIdentifier("doi", "10.1000/old")
    request_identifier = ExternalIdentifier("doi", "10.1000/new")
    outcome = resolve_identity(
        IdentityResolutionRequest(
            HASH_A,
            (request_identifier,),
            (_version(FIRST_VERSION_ID, identifiers=(candidate_identifier,)),),
        ),
        allocator=FixedAllocator(),
    )

    assert outcome.kind is ResolutionKind.CONFLICT
    assert outcome.document is None
    assert outcome.version is None


def test_allocator_type_errors_are_reported() -> None:
    with pytest.raises(TypeError, match="new_document_id"):
        resolve_identity(IdentityResolutionRequest(HASH_A), allocator=object())  # type: ignore[arg-type]


def test_allocator_collisions_fail_closed() -> None:
    allocator = FixedAllocator(document_id=DOCUMENT_ID)
    candidate = _version(FIRST_VERSION_ID, content_hash=HASH_A)

    with pytest.raises(InvalidResolutionError, match="existing DocumentId"):
        resolve_identity(
            IdentityResolutionRequest(HASH_B, candidates=(candidate,)),
            allocator=allocator,
        )
