"""Negative contract and boundary coverage for identity resolution."""

import pytest

from inference_lab.provenance import (
    ContentHash,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    ExternalIdentifierResolutionPolicy,
    HashAlgorithm,
    IdentityResolutionOutcome,
    IdentityResolutionRequest,
    InvalidResolutionError,
    ResolutionKind,
    ResolutionReason,
    SourceObservationId,
    UuidIdentityIdAllocator,
    resolve_identity,
)

_HASH = ContentHash(HashAlgorithm.SHA256, "a" * 64)
_OTHER_HASH = ContentHash(HashAlgorithm.SHA256, "b" * 64)
_DOCUMENT_A = DocumentId.from_str("00000000-0000-4000-8000-000000000001")
_DOCUMENT_B = DocumentId.from_str("00000000-0000-4000-8000-000000000002")
_DOCUMENT_NEW = DocumentId.from_str("00000000-0000-4000-8000-000000000003")
_VERSION_A = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000004")
_VERSION_B = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000005")
_VERSION_NEW = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000006")
_SOURCE_A = SourceObservationId.from_str("00000000-0000-4000-8000-000000000007")
_SOURCE_B = SourceObservationId.from_str("00000000-0000-4000-8000-000000000008")
_DOI = ExternalIdentifier("doi", "10.1000/example")


def _version(
    version_id: DocumentVersionId,
    document_id: DocumentId = _DOCUMENT_A,
    content_hash: ContentHash = _HASH,
) -> DocumentVersion:
    """Build one deterministic document version."""
    return DocumentVersion(version_id, document_id, content_hash)


def _assertion(
    version_id: DocumentVersionId = _VERSION_A,
    source_id: SourceObservationId = _SOURCE_A,
) -> ExternalIdentifierAssertion:
    """Build one deterministic external-identifier assertion."""
    return ExternalIdentifierAssertion(source_id, version_id, _DOI)


def _new_version_outcome(
    *,
    matched_version_ids: tuple[DocumentVersionId, ...] = (_VERSION_A,),
) -> IdentityResolutionOutcome:
    """Build a valid new-version outcome."""
    return IdentityResolutionOutcome(
        ResolutionKind.NEW_VERSION,
        ResolutionReason.SOURCE_CONTINUITY_MATCH,
        DocumentIdentity(_DOCUMENT_A),
        _version(_VERSION_NEW, content_hash=_OTHER_HASH),
        matched_version_ids,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (1, "1.0.0", ()),
        ("fixture-policy", 1, ()),
        ("fixture-policy", "1.0.0", (1,)),
    ],
)
def test_identifier_policy_rejects_wrong_field_types(
    arguments: tuple[object, object, object],
) -> None:
    with pytest.raises(TypeError):
        ExternalIdentifierResolutionPolicy(*arguments)  # type: ignore[arg-type]


def test_identifier_policy_rejects_noncanonical_namespace() -> None:
    with pytest.raises(ValueError, match="lowercase namespaces"):
        ExternalIdentifierResolutionPolicy(
            "fixture-policy",
            "1.0.0",
            ("DOI",),
        )


def test_identifier_assertion_rejects_wrong_field_types() -> None:
    with pytest.raises(TypeError, match="source_observation_id"):
        ExternalIdentifierAssertion(_DOCUMENT_A, _VERSION_A, _DOI)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="document_version_id"):
        ExternalIdentifierAssertion(_SOURCE_A, _DOCUMENT_A, _DOI)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="external_identifier"):
        ExternalIdentifierAssertion(_SOURCE_A, _VERSION_A, "doi")  # type: ignore[arg-type]


def test_request_rejects_wrong_primary_and_policy_types() -> None:
    with pytest.raises(TypeError, match="content_hash"):
        IdentityResolutionRequest(_DOI, _SOURCE_A)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="external_identifier_policy"):
        IdentityResolutionRequest(
            _HASH,
            _SOURCE_A,
            external_identifier_policy="conservative",  # type: ignore[arg-type]
        )


def test_request_rejects_wrong_candidate_collection_shapes() -> None:
    candidate = _version(_VERSION_A)

    with pytest.raises(TypeError, match="candidates must be a tuple"):
        IdentityResolutionRequest(
            _HASH,
            _SOURCE_A,
            continuity_candidates=[candidate],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="DocumentVersion"):
        IdentityResolutionRequest(
            _HASH,
            _SOURCE_A,
            continuity_candidates=(_DOCUMENT_A,),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="duplicate versions"):
        IdentityResolutionRequest(
            _HASH,
            _SOURCE_A,
            continuity_candidates=(candidate, candidate),
        )


def test_request_rejects_invalid_assertion_collections() -> None:
    first = _assertion(_VERSION_A, _SOURCE_A)
    second = _assertion(_VERSION_B, _SOURCE_B)
    candidates = (
        _version(_VERSION_A, _DOCUMENT_A),
        _version(_VERSION_B, _DOCUMENT_B),
    )

    with pytest.raises(TypeError, match="identifier_assertions must be a tuple"):
        IdentityResolutionRequest(
            _HASH,
            _SOURCE_A,
            continuity_candidates=candidates,
            identifier_assertions=[first],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="ExternalIdentifierAssertion"):
        IdentityResolutionRequest(
            _HASH,
            _SOURCE_A,
            continuity_candidates=candidates,
            identifier_assertions=(_DOI,),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="duplicates"):
        IdentityResolutionRequest(
            _HASH,
            _SOURCE_A,
            continuity_candidates=candidates,
            identifier_assertions=(first, first),
        )
    with pytest.raises(ValueError, match="canonical order"):
        IdentityResolutionRequest(
            _HASH,
            _SOURCE_A,
            continuity_candidates=candidates,
            identifier_assertions=(second, first),
        )


def test_outcome_rejects_wrong_field_types() -> None:
    valid_version = _version(_VERSION_NEW, content_hash=_OTHER_HASH)

    with pytest.raises(TypeError, match="kind"):
        IdentityResolutionOutcome("new_version", ResolutionReason.NO_MATCH)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reason"):
        IdentityResolutionOutcome(ResolutionKind.NEW_VERSION, "no_match")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="document"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_VERSION,
            ResolutionReason.SOURCE_CONTINUITY_MATCH,
            _DOCUMENT_A,  # type: ignore[arg-type]
            valid_version,
            (_VERSION_A,),
        )
    with pytest.raises(TypeError, match="version"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_VERSION,
            ResolutionReason.SOURCE_CONTINUITY_MATCH,
            DocumentIdentity(_DOCUMENT_A),
            _VERSION_NEW,  # type: ignore[arg-type]
            (_VERSION_A,),
        )
    with pytest.raises(TypeError, match="duplicate_relationships must be a tuple"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_VERSION,
            ResolutionReason.SOURCE_CONTINUITY_MATCH,
            DocumentIdentity(_DOCUMENT_A),
            valid_version,
            (_VERSION_A,),
            [],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="DuplicateRelationship"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_VERSION,
            ResolutionReason.SOURCE_CONTINUITY_MATCH,
            DocumentIdentity(_DOCUMENT_A),
            valid_version,
            (_VERSION_A,),
            (_DOI,),  # type: ignore[arg-type]
        )


def test_outcome_rejects_invalid_matched_version_collections() -> None:
    with pytest.raises(TypeError, match="matched_version_ids must be a tuple"):
        _new_version_outcome(matched_version_ids=[_VERSION_A])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DocumentVersionId"):
        _new_version_outcome(matched_version_ids=(_DOCUMENT_A,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicates"):
        _new_version_outcome(matched_version_ids=(_VERSION_A, _VERSION_A))
    with pytest.raises(InvalidResolutionError, match="canonical order"):
        _new_version_outcome(matched_version_ids=(_VERSION_B, _VERSION_A))


def test_unchanged_outcome_rejects_wrong_match_and_assertion_targets() -> None:
    version = _version(_VERSION_A)
    document = DocumentIdentity(_DOCUMENT_A)

    with pytest.raises(InvalidResolutionError, match="exactly one retained version"):
        IdentityResolutionOutcome(
            ResolutionKind.UNCHANGED,
            ResolutionReason.EXACT_HASH_MATCH,
            document,
            version,
            (_VERSION_B,),
        )
    with pytest.raises(InvalidResolutionError, match="target the retained version"):
        IdentityResolutionOutcome(
            ResolutionKind.UNCHANGED,
            ResolutionReason.EXACT_HASH_MATCH,
            document,
            version,
            (_VERSION_A,),
            identifier_assertions_to_add=(_assertion(_VERSION_B),),
        )


def test_assigned_outcomes_reject_kind_specific_semantic_conflicts() -> None:
    document = DocumentIdentity(_DOCUMENT_NEW)
    version = _version(_VERSION_NEW, _DOCUMENT_NEW, _OTHER_HASH)
    relationship = DuplicateRelationship(
        _VERSION_A,
        _VERSION_NEW,
        DuplicateClass.BYTE_IDENTICAL,
    )

    with pytest.raises(InvalidResolutionError, match="requires document and version"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_DOCUMENT,
            ResolutionReason.NO_MATCH,
        )
    with pytest.raises(InvalidResolutionError, match="source continuity"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_VERSION,
            ResolutionReason.NO_MATCH,
            document,
            version,
            (_VERSION_A,),
        )
    with pytest.raises(InvalidResolutionError, match="matching candidate"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_VERSION,
            ResolutionReason.SOURCE_CONTINUITY_MATCH,
            document,
            version,
        )
    with pytest.raises(InvalidResolutionError, match="matching candidates"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_DOCUMENT,
            ResolutionReason.NO_MATCH,
            document,
            version,
            (_VERSION_A,),
        )
    with pytest.raises(InvalidResolutionError, match="requires reason"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_DOCUMENT,
            ResolutionReason.EXACT_HASH_MATCH,
            document,
            version,
        )
    with pytest.raises(InvalidResolutionError, match="duplicate relationships"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_DOCUMENT,
            ResolutionReason.NO_MATCH,
            document,
            version,
            duplicate_relationships=(relationship,),
        )
    with pytest.raises(InvalidResolutionError, match="identifier assertions"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_DOCUMENT,
            ResolutionReason.NO_MATCH,
            document,
            version,
            identifier_assertions_to_add=(_assertion(),),
        )


def test_duplicate_outcome_rejects_incomplete_or_inconsistent_relationships() -> None:
    document = DocumentIdentity(_DOCUMENT_NEW)
    version = _version(_VERSION_NEW, _DOCUMENT_NEW, _OTHER_HASH)
    matching = DuplicateRelationship(
        _VERSION_A,
        _VERSION_NEW,
        DuplicateClass.BYTE_IDENTICAL,
    )

    with pytest.raises(InvalidResolutionError, match="candidates and relationships"):
        IdentityResolutionOutcome(
            ResolutionKind.DUPLICATE,
            ResolutionReason.DUPLICATE_HASH_MATCH,
            document,
            version,
            (_VERSION_A,),
        )
    with pytest.raises(InvalidResolutionError, match="must be unique"):
        IdentityResolutionOutcome(
            ResolutionKind.DUPLICATE,
            ResolutionReason.DUPLICATE_HASH_MATCH,
            document,
            version,
            (_VERSION_A,),
            (matching, matching),
        )
    with pytest.raises(InvalidResolutionError, match="resolved version"):
        IdentityResolutionOutcome(
            ResolutionKind.DUPLICATE,
            ResolutionReason.DUPLICATE_HASH_MATCH,
            document,
            version,
            (_VERSION_A,),
            (
                DuplicateRelationship(
                    _VERSION_A,
                    _VERSION_B,
                    DuplicateClass.BYTE_IDENTICAL,
                ),
            ),
        )
    with pytest.raises(InvalidResolutionError, match="matched versions"):
        IdentityResolutionOutcome(
            ResolutionKind.DUPLICATE,
            ResolutionReason.DUPLICATE_HASH_MATCH,
            document,
            version,
            (_VERSION_A,),
            (
                DuplicateRelationship(
                    _VERSION_B,
                    _VERSION_NEW,
                    DuplicateClass.BYTE_IDENTICAL,
                ),
            ),
        )


def test_unassigned_outcomes_reject_assignments_reasons_and_assertions() -> None:
    with pytest.raises(InvalidResolutionError, match="cannot assign"):
        IdentityResolutionOutcome(
            ResolutionKind.AMBIGUOUS,
            ResolutionReason.MULTIPLE_CANDIDATES,
            DocumentIdentity(_DOCUMENT_A),
            _version(_VERSION_A),
            (_VERSION_A,),
        )
    with pytest.raises(InvalidResolutionError, match="multiple candidates"):
        IdentityResolutionOutcome(
            ResolutionKind.AMBIGUOUS,
            ResolutionReason.EXACT_HASH_MATCH,
            matched_version_ids=(_VERSION_A,),
        )
    with pytest.raises(InvalidResolutionError, match="identifier assertions"):
        IdentityResolutionOutcome(
            ResolutionKind.AMBIGUOUS,
            ResolutionReason.MULTIPLE_CANDIDATES,
            matched_version_ids=(_VERSION_A,),
            identifier_assertions_to_add=(_assertion(),),
        )


class _FixedAllocator:
    """Allocate deterministic identities for boundary tests."""

    def new_document_id(self) -> DocumentId:
        return _DOCUMENT_NEW

    def new_document_version_id(self) -> DocumentVersionId:
        return _VERSION_NEW


class _IncompleteAllocator:
    """Expose only half of the allocator protocol."""

    def new_document_id(self) -> DocumentId:
        return _DOCUMENT_NEW


def test_uuid_allocator_returns_typed_fresh_identifiers() -> None:
    allocator = UuidIdentityIdAllocator()

    assert isinstance(allocator.new_document_id(), DocumentId)
    assert isinstance(allocator.new_document_version_id(), DocumentVersionId)


def test_resolver_rejects_wrong_request_and_incomplete_allocator() -> None:
    with pytest.raises(TypeError, match="request"):
        resolve_identity(
            _HASH,  # type: ignore[arg-type]
            allocator=_FixedAllocator(),
            pipeline_stage="ingest",
        )
    with pytest.raises(TypeError, match="new_document_version_id"):
        resolve_identity(
            IdentityResolutionRequest(_HASH, _SOURCE_A),
            allocator=_IncompleteAllocator(),  # type: ignore[arg-type]
            pipeline_stage="ingest",
        )


def test_multiple_continuity_hash_matches_are_ambiguous_without_allocation() -> None:
    candidates = (
        _version(_VERSION_A, _DOCUMENT_A),
        _version(_VERSION_B, _DOCUMENT_B),
    )

    outcome = resolve_identity(
        IdentityResolutionRequest(
            _HASH,
            _SOURCE_A,
            continuity_candidates=candidates,
        ),
        allocator=_FixedAllocator(),
        pipeline_stage="ingest",
    )

    assert outcome.kind is ResolutionKind.AMBIGUOUS
    assert outcome.reason is ResolutionReason.MULTIPLE_CANDIDATES
    assert outcome.matched_version_ids == (_VERSION_A, _VERSION_B)
