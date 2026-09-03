"""Tests for document identity-resolution contracts."""

import pytest

from inference_lab.provenance import (
    CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY,
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
)

HASH = ContentHash(HashAlgorithm.SHA256, "a" * 64)
OTHER_HASH = ContentHash(HashAlgorithm.SHA256, "b" * 64)
DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000001")
OTHER_DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000002")
FIRST_VERSION_ID = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000003")
SECOND_VERSION_ID = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000004")
SOURCE_OBSERVATION_ID = SourceObservationId.from_str(
    "00000000-0000-4000-8000-000000000005"
)


def _version(
    version_id: DocumentVersionId,
    document_id: DocumentId = DOCUMENT_ID,
) -> DocumentVersion:
    return DocumentVersion(version_id, document_id, HASH)


def _identity(document_id: DocumentId = DOCUMENT_ID) -> DocumentIdentity:
    return DocumentIdentity(document_id)


def test_request_validates_and_serialises_candidates() -> None:
    request = IdentityResolutionRequest(
        HASH,
        SOURCE_OBSERVATION_ID,
        continuity_candidates=(_version(FIRST_VERSION_ID),),
    )

    assert request.as_dict()["content_hash"] == HASH.as_str()
    assert request.as_dict()["source_observation_id"] == SOURCE_OBSERVATION_ID.as_str()
    assert request.as_dict()["external_identifier_policy"] == {
        "policy_name": "conservative",
        "policy_version": "1.0.0",
        "authoritative_namespaces": [],
    }
    assert request.external_identifier_policy is CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY


def test_request_rejects_reversed_candidates() -> None:
    with pytest.raises(InvalidResolutionError, match="canonical order"):
        IdentityResolutionRequest(
            HASH,
            SOURCE_OBSERVATION_ID,
            continuity_candidates=(
                _version(SECOND_VERSION_ID),
                _version(FIRST_VERSION_ID),
            ),
        )


def test_request_requires_typed_source_observation_identity() -> None:
    with pytest.raises(TypeError, match="source_observation_id"):
        IdentityResolutionRequest(HASH, object())  # type: ignore[arg-type]


def test_request_rejects_invalid_content_hash_candidate_role() -> None:
    with pytest.raises(InvalidResolutionError, match="requested content hash"):
        IdentityResolutionRequest(
            HASH,
            SOURCE_OBSERVATION_ID,
            content_hash_candidates=(
                DocumentVersion(FIRST_VERSION_ID, DOCUMENT_ID, OTHER_HASH),
            ),
        )


def test_request_rejects_conflicting_cross_role_representation() -> None:
    with pytest.raises(InvalidResolutionError, match="one representation"):
        IdentityResolutionRequest(
            HASH,
            SOURCE_OBSERVATION_ID,
            continuity_candidates=(
                DocumentVersion(FIRST_VERSION_ID, DOCUMENT_ID, OTHER_HASH),
            ),
            content_hash_candidates=(_version(FIRST_VERSION_ID),),
        )


def test_request_rejects_assertion_for_unsupplied_candidate() -> None:
    assertion = ExternalIdentifierAssertion(
        SOURCE_OBSERVATION_ID,
        FIRST_VERSION_ID,
        ExternalIdentifier("doi", "10.1000/example"),
    )

    with pytest.raises(InvalidResolutionError, match="supplied candidate versions"):
        IdentityResolutionRequest(
            HASH,
            SOURCE_OBSERVATION_ID,
            identifier_assertions=(assertion,),
        )


def test_request_serialises_identifier_assertion_evidence() -> None:
    assertion = ExternalIdentifierAssertion(
        SOURCE_OBSERVATION_ID,
        FIRST_VERSION_ID,
        ExternalIdentifier("doi", "10.1000/example"),
    )
    request = IdentityResolutionRequest(
        HASH,
        SOURCE_OBSERVATION_ID,
        continuity_candidates=(_version(FIRST_VERSION_ID),),
        identifier_assertions=(assertion,),
    )

    assert request.as_dict()["identifier_assertions"] == [
        {
            "source_observation_id": SOURCE_OBSERVATION_ID.as_str(),
            "document_version_id": FIRST_VERSION_ID.as_str(),
            "external_identifier": {
                "namespace": "doi",
                "value": "10.1000/example",
            },
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"policy_name": "Not-Canonical"}, ValueError),
        ({"policy_version": "1"}, ValueError),
        ({"authoritative_namespaces": ["doi"]}, TypeError),
        ({"authoritative_namespaces": ("pmid", "doi")}, ValueError),
        ({"authoritative_namespaces": ("doi", "doi")}, ValueError),
    ],
)
def test_external_identifier_policy_rejects_noncanonical_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "policy_name": "fixture-policy",
        "policy_version": "1.0.0",
        "authoritative_namespaces": (),
    }
    arguments.update(kwargs)

    with pytest.raises(error_type):
        ExternalIdentifierResolutionPolicy(**arguments)  # type: ignore[arg-type]


def test_unchanged_outcome_requires_retained_version() -> None:
    outcome = IdentityResolutionOutcome(
        ResolutionKind.UNCHANGED,
        ResolutionReason.EXACT_HASH_MATCH,
        _identity(),
        _version(FIRST_VERSION_ID),
        (FIRST_VERSION_ID,),
    )

    assert outcome.as_dict()["kind"] == "unchanged"


def test_unchanged_outcome_allows_new_namespace_enrichment() -> None:
    enrichment = ExternalIdentifier("pmid", "12345678")
    assertion = ExternalIdentifierAssertion(
        SOURCE_OBSERVATION_ID,
        FIRST_VERSION_ID,
        enrichment,
    )
    outcome = IdentityResolutionOutcome(
        ResolutionKind.UNCHANGED,
        ResolutionReason.EXACT_HASH_MATCH,
        _identity(),
        _version(FIRST_VERSION_ID),
        (FIRST_VERSION_ID,),
        identifier_assertions_to_add=(assertion,),
    )

    assert outcome.identifier_assertions_to_add == (assertion,)


def test_unchanged_outcome_rejects_enrichment_of_existing_namespace() -> None:
    existing = ExternalIdentifier("doi", "10.1000/existing")
    version = DocumentVersion(
        FIRST_VERSION_ID,
        DOCUMENT_ID,
        HASH,
        (existing,),
    )

    with pytest.raises(InvalidResolutionError, match="unasserted namespaces"):
        IdentityResolutionOutcome(
            ResolutionKind.UNCHANGED,
            ResolutionReason.EXACT_HASH_MATCH,
            _identity(),
            version,
            (FIRST_VERSION_ID,),
            identifier_assertions_to_add=(
                ExternalIdentifierAssertion(
                    SOURCE_OBSERVATION_ID,
                    FIRST_VERSION_ID,
                    ExternalIdentifier("doi", "10.1000/different"),
                ),
            ),
        )


def test_new_document_outcome_requires_no_match() -> None:
    outcome = IdentityResolutionOutcome(
        ResolutionKind.NEW_DOCUMENT,
        ResolutionReason.NO_MATCH,
        _identity(OTHER_DOCUMENT_ID),
        _version(SECOND_VERSION_ID, OTHER_DOCUMENT_ID),
    )

    assert outcome.matched_version_ids == ()


def test_duplicate_outcome_requires_matching_relationship() -> None:
    relationship = DuplicateRelationship(
        FIRST_VERSION_ID,
        SECOND_VERSION_ID,
        DuplicateClass.BYTE_IDENTICAL,
    )

    outcome = IdentityResolutionOutcome(
        ResolutionKind.DUPLICATE,
        ResolutionReason.DUPLICATE_HASH_MATCH,
        _identity(OTHER_DOCUMENT_ID),
        _version(SECOND_VERSION_ID, OTHER_DOCUMENT_ID),
        (FIRST_VERSION_ID,),
        (relationship,),
    )

    assert outcome.duplicate_relationships == (relationship,)


def test_ambiguous_outcome_cannot_assign_identity() -> None:
    with pytest.raises(InvalidResolutionError, match="candidate versions"):
        IdentityResolutionOutcome(
            ResolutionKind.AMBIGUOUS,
            ResolutionReason.MULTIPLE_CANDIDATES,
        )


def test_outcome_rejects_document_version_mismatch() -> None:
    with pytest.raises(InvalidResolutionError, match="same document"):
        IdentityResolutionOutcome(
            ResolutionKind.NEW_VERSION,
            ResolutionReason.EXTERNAL_IDENTIFIER_MATCH,
            _identity(),
            _version(SECOND_VERSION_ID, OTHER_DOCUMENT_ID),
            (FIRST_VERSION_ID,),
        )
