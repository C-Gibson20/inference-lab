"""Tests for deterministic document identity resolution."""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

import inference_lab.provenance.resolver as resolver_module
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

    outcome = resolve_identity(
        IdentityResolutionRequest(HASH_A),
        allocator=allocator,
        pipeline_stage="ingest",
    )

    assert outcome.kind is ResolutionKind.NEW_DOCUMENT
    assert outcome.reason is ResolutionReason.NO_MATCH
    assert outcome.document == DocumentIdentity(ALLOCATED_DOCUMENT_ID)
    assert outcome.version is not None
    assert outcome.version.document_version_id == ALLOCATED_VERSION_ID
    assert allocator.document_calls == 1
    assert allocator.version_calls == 1


@pytest.mark.parametrize("source_change", ["move", "rename"])
def test_source_location_changes_do_not_change_identity_resolution(
    source_change: str,
) -> None:
    """Path-level changes are outside the resolver's provenance evidence."""
    candidate = _version(FIRST_VERSION_ID)

    first_outcome = resolve_identity(
        IdentityResolutionRequest(HASH_A, candidates=(candidate,)),
        allocator=FixedAllocator(),
        pipeline_stage="ingest",
    )
    second_outcome = resolve_identity(
        IdentityResolutionRequest(HASH_A, candidates=(candidate,)),
        allocator=FixedAllocator(),
        pipeline_stage="ingest",
    )

    assert source_change in {"move", "rename"}
    assert second_outcome == first_outcome
    assert second_outcome.kind is ResolutionKind.UNCHANGED


def test_repeated_new_document_resolution_is_deterministic_with_fixed_allocator() -> (
    None
):
    first_outcome = resolve_identity(
        IdentityResolutionRequest(HASH_C),
        allocator=FixedAllocator(),
        pipeline_stage="ingest",
    )
    second_outcome = resolve_identity(
        IdentityResolutionRequest(HASH_C),
        allocator=FixedAllocator(),
        pipeline_stage="ingest",
    )

    assert second_outcome == first_outcome


def test_exactly_one_hash_match_is_unchanged_without_allocation() -> None:
    candidate = _version(FIRST_VERSION_ID)
    allocator = FixedAllocator()

    outcome = resolve_identity(
        IdentityResolutionRequest(HASH_A, candidates=(candidate,)),
        allocator=allocator,
        pipeline_stage="ingest",
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
        pipeline_stage="ingest",
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
        pipeline_stage="ingest",
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
        pipeline_stage="ingest",
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
        pipeline_stage="ingest",
    )

    assert outcome.kind is ResolutionKind.CONFLICT
    assert outcome.document is None
    assert outcome.version is None


def test_allocator_type_errors_are_reported() -> None:
    with pytest.raises(TypeError, match="new_document_id"):
        resolve_identity(
            IdentityResolutionRequest(HASH_A),
            allocator=object(),  # type: ignore[arg-type]
            pipeline_stage="ingest",
        )


@pytest.mark.parametrize(
    ("allocator", "request_obj", "message"),
    [
        (
            FixedAllocator(document_id=object()),  # type: ignore[arg-type]
            IdentityResolutionRequest(HASH_B),
            "new_document_id",
        ),
        (
            FixedAllocator(version_id=object()),  # type: ignore[arg-type]
            IdentityResolutionRequest(
                HASH_B,
                candidates=(_version(FIRST_VERSION_ID, content_hash=HASH_A),),
            ),
            "new_document_version_id",
        ),
    ],
)
def test_allocator_return_types_are_validated(
    allocator: FixedAllocator,
    request_obj: IdentityResolutionRequest,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        resolve_identity(
            request_obj,
            allocator=allocator,
            pipeline_stage="ingest",
        )


def test_allocator_collisions_fail_closed() -> None:
    allocator = FixedAllocator(document_id=DOCUMENT_ID)
    candidate = _version(FIRST_VERSION_ID, content_hash=HASH_A)

    with pytest.raises(InvalidResolutionError, match="existing DocumentId"):
        resolve_identity(
            IdentityResolutionRequest(HASH_B, candidates=(candidate,)),
            allocator=allocator,
            pipeline_stage="ingest",
        )


def test_allocator_version_collision_fails_closed() -> None:
    allocator = FixedAllocator(version_id=FIRST_VERSION_ID)
    candidate = _version(FIRST_VERSION_ID, content_hash=HASH_A)

    with pytest.raises(InvalidResolutionError, match="existing DocumentVersionId"):
        resolve_identity(
            IdentityResolutionRequest(HASH_B, candidates=(candidate,)),
            allocator=allocator,
            pipeline_stage="ingest",
        )


def test_completed_resolution_emits_privacy_safe_structured_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(resolver_module, "logger", logger)
    candidate = _version(FIRST_VERSION_ID)

    resolve_identity(
        IdentityResolutionRequest(HASH_A, candidates=(candidate,)),
        allocator=FixedAllocator(),
        pipeline_stage="ingest",
    )

    logger.info.assert_called_once_with(
        "identity_resolution_completed",
        pipeline_stage="ingest",
        document_id=DOCUMENT_ID.as_str(),
        resolution_kind="unchanged",
        resolution_reason="exact_hash_match",
        matched_candidate_count=1,
        duplicate_relationship_count=0,
        candidate_count=1,
    )
    logger.error.assert_not_called()


@pytest.mark.parametrize(
    ("request_obj", "expected_kind", "expected_reason", "expected_document_id"),
    [
        (
            IdentityResolutionRequest(HASH_A, candidates=(_version(FIRST_VERSION_ID),)),
            "unchanged",
            "exact_hash_match",
            DOCUMENT_ID.as_str(),
        ),
        (
            IdentityResolutionRequest(HASH_B),
            "new_document",
            "no_match",
            ALLOCATED_DOCUMENT_ID.as_str(),
        ),
        (
            IdentityResolutionRequest(
                HASH_A,
                candidates=(
                    _version(FIRST_VERSION_ID, DOCUMENT_ID),
                    _version(SECOND_VERSION_ID, OTHER_DOCUMENT_ID),
                ),
            ),
            "duplicate",
            "duplicate_hash_match",
            ALLOCATED_DOCUMENT_ID.as_str(),
        ),
        (
            IdentityResolutionRequest(
                HASH_C,
                (ExternalIdentifier("doi", "10.1000/example"),),
                (
                    _version(
                        FIRST_VERSION_ID,
                        identifiers=(ExternalIdentifier("doi", "10.1000/example"),),
                    ),
                ),
            ),
            "new_version",
            "external_identifier_match",
            DOCUMENT_ID.as_str(),
        ),
        (
            IdentityResolutionRequest(
                HASH_C,
                (ExternalIdentifier("doi", "10.1000/example"),),
                (
                    _version(
                        FIRST_VERSION_ID,
                        DOCUMENT_ID,
                        HASH_A,
                        (ExternalIdentifier("doi", "10.1000/example"),),
                    ),
                    _version(
                        SECOND_VERSION_ID,
                        OTHER_DOCUMENT_ID,
                        HASH_B,
                        (ExternalIdentifier("doi", "10.1000/example"),),
                    ),
                ),
            ),
            "ambiguous",
            "multiple_candidates",
            None,
        ),
        (
            IdentityResolutionRequest(
                HASH_A,
                (ExternalIdentifier("doi", "10.1000/new"),),
                (
                    _version(
                        FIRST_VERSION_ID,
                        identifiers=(ExternalIdentifier("doi", "10.1000/old"),),
                    ),
                ),
            ),
            "conflict",
            "conflicting_evidence",
            None,
        ),
    ],
)
def test_each_resolution_outcome_emits_completion_event(
    request_obj: IdentityResolutionRequest,
    expected_kind: str,
    expected_reason: str,
    expected_document_id: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(resolver_module, "logger", logger)

    resolve_identity(
        request_obj,
        allocator=FixedAllocator(),
        pipeline_stage="ingest",
    )

    assert logger.info.call_count == 1
    fields = logger.info.call_args.kwargs
    assert logger.info.call_args.args == ("identity_resolution_completed",)
    assert fields["pipeline_stage"] == "ingest"
    assert fields["resolution_kind"] == expected_kind
    assert fields["resolution_reason"] == expected_reason
    assert fields["document_id"] == expected_document_id
    assert not any(
        sensitive in fields
        for sensitive in ("content_hash", "external_identifiers", "path", "source")
    )


@pytest.mark.parametrize("pipeline_stage", ["", "   ", 42])
def test_invalid_pipeline_stage_is_rejected_without_logging(
    pipeline_stage: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(resolver_module, "logger", logger)

    expected_error = TypeError if not isinstance(pipeline_stage, str) else ValueError
    with pytest.raises(expected_error, match="pipeline_stage"):
        resolve_identity(
            IdentityResolutionRequest(HASH_A),
            allocator=FixedAllocator(),
            pipeline_stage=pipeline_stage,  # type: ignore[arg-type]
        )

    logger.info.assert_not_called()
    logger.error.assert_not_called()


def test_resolution_failure_emits_error_without_sensitive_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(resolver_module, "logger", logger)
    allocator = FixedAllocator(document_id=DOCUMENT_ID)
    candidate = _version(FIRST_VERSION_ID, content_hash=HASH_A)

    with pytest.raises(InvalidResolutionError, match="existing DocumentId"):
        resolve_identity(
            IdentityResolutionRequest(HASH_B, candidates=(candidate,)),
            allocator=allocator,
            pipeline_stage="ingest",
        )

    logger.error.assert_called_once_with(
        "identity_resolution_failed",
        pipeline_stage="ingest",
        failure_type="InvalidResolutionError",
        candidate_count=1,
    )
    logger.info.assert_not_called()
