"""Contracts for deterministic document identity resolution."""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from inference_lab._validation import is_canonical_semantic_version
from inference_lab.provenance.identifiers import DocumentVersionId, SourceObservationId
from inference_lab.provenance.records import DocumentIdentity, DocumentVersion
from inference_lab.provenance.relationships import DuplicateRelationship
from inference_lab.provenance.utils import is_lowercase_slug
from inference_lab.provenance.values import (
    ContentHash,
    ExternalIdentifier,
    ResolutionKind,
    validate_external_identifiers,
)

_IDENTIFIER_NAMESPACE: Final = re.compile(r"[a-z][a-z0-9_-]*")


class InvalidResolutionError(ValueError):
    """Raised when a resolution request or outcome violates its contract."""


class ResolutionReason(StrEnum):
    """Reasons explaining a deterministic resolution outcome."""

    EXACT_HASH_MATCH = "exact_hash_match"
    SOURCE_CONTINUITY_MATCH = "source_continuity_match"
    EXTERNAL_IDENTIFIER_MATCH = "external_identifier_match"
    NO_MATCH = "no_match"
    DUPLICATE_HASH_MATCH = "duplicate_hash_match"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    MULTIPLE_CANDIDATES = "multiple_candidates"
    UNESTABLISHED_IDENTIFIER_AUTHORITY = "unestablished_identifier_authority"


@dataclass(frozen=True, slots=True)
class ExternalIdentifierResolutionPolicy:
    """Versioned authority policy for external-identifier evidence."""

    policy_name: str
    policy_version: str
    authoritative_namespaces: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the policy fields and authoritative namespaces."""
        if not isinstance(self.policy_name, str):
            raise TypeError("policy_name must be a string.")
        if not is_lowercase_slug(self.policy_name):
            raise ValueError("policy_name must be a lowercase policy name.")
        if not isinstance(self.policy_version, str):
            raise TypeError("policy_version must be a string.")
        if not is_canonical_semantic_version(self.policy_version):
            raise ValueError("policy_version must be a canonical semantic version.")
        if not isinstance(self.authoritative_namespaces, tuple):
            raise TypeError("authoritative_namespaces must be a tuple.")
        if any(
            not isinstance(namespace, str)
            for namespace in self.authoritative_namespaces
        ):
            raise TypeError("authoritative_namespaces must contain strings.")
        if any(
            _IDENTIFIER_NAMESPACE.fullmatch(namespace) is None
            for namespace in self.authoritative_namespaces
        ):
            raise ValueError(
                "authoritative_namespaces must contain lowercase namespaces."
            )
        if len(self.authoritative_namespaces) != len(
            set(self.authoritative_namespaces)
        ):
            raise ValueError("authoritative_namespaces must not contain duplicates.")
        if self.authoritative_namespaces != tuple(
            sorted(self.authoritative_namespaces)
        ):
            raise ValueError("authoritative_namespaces must be in canonical order.")


CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY: Final = ExternalIdentifierResolutionPolicy(
    policy_name="conservative",
    policy_version="1.0.0",
)


@dataclass(frozen=True, slots=True)
class ExternalIdentifierAssertion:
    """Append-only identifier evidence for an immutable source version."""

    source_observation_id: SourceObservationId
    document_version_id: DocumentVersionId
    external_identifier: ExternalIdentifier

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the assertion fields."""
        if not isinstance(self.source_observation_id, SourceObservationId):
            raise TypeError("source_observation_id must be a SourceObservationId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.external_identifier, ExternalIdentifier):
            raise TypeError("external_identifier must be an ExternalIdentifier.")

    def key(self) -> tuple[str, str, str, str]:
        """Return the canonical assertion key."""
        return (
            self.document_version_id.as_str(),
            self.source_observation_id.as_str(),
            self.external_identifier.namespace,
            self.external_identifier.value,
        )


def _validate_identifier_assertions(assertions: object) -> None:
    """Validate unique, canonically ordered identifier assertions."""
    if not isinstance(assertions, tuple):
        raise TypeError("identifier_assertions must be a tuple.")
    if any(
        not isinstance(assertion, ExternalIdentifierAssertion)
        for assertion in assertions
    ):
        raise TypeError(
            "identifier_assertions must contain ExternalIdentifierAssertion values."
        )
    keys = tuple(assertion.key() for assertion in assertions)
    if len(keys) != len(set(keys)):
        raise ValueError("identifier_assertions must not contain duplicates.")
    if keys != tuple(sorted(keys)):
        raise ValueError("identifier_assertions must be in canonical order.")


def _validate_candidates(candidates: object) -> None:
    """Validate unique, canonically ordered candidate versions."""
    if not isinstance(candidates, tuple):
        raise TypeError("candidates must be a tuple.")
    if any(not isinstance(candidate, DocumentVersion) for candidate in candidates):
        raise TypeError("candidates must contain only DocumentVersion values.")

    keys = tuple(
        (candidate.document_id.as_str(), candidate.document_version_id.as_str())
        for candidate in candidates
    )
    if len(keys) != len(set(keys)):
        raise ValueError("candidates must not contain duplicate versions.")
    if keys != tuple(sorted(keys)):
        raise InvalidResolutionError("candidates must be in canonical order.")


def _validate_matched_versions(version_ids: object) -> None:
    """Validate unique, canonically ordered matched-version identifiers."""
    if not isinstance(version_ids, tuple):
        raise TypeError("matched_version_ids must be a tuple.")
    if any(not isinstance(version_id, DocumentVersionId) for version_id in version_ids):
        raise TypeError(
            "matched_version_ids must contain only DocumentVersionId values."
        )

    values = tuple(version_id.as_str() for version_id in version_ids)
    if len(values) != len(set(values)):
        raise ValueError("matched_version_ids must not contain duplicates.")
    if values != tuple(sorted(values)):
        raise InvalidResolutionError("matched_version_ids must be in canonical order.")


@dataclass(frozen=True, slots=True)
class IdentityResolutionRequest:
    """Evidence and candidates supplied to identity resolution."""

    content_hash: ContentHash
    source_observation_id: SourceObservationId
    external_identifiers: tuple[ExternalIdentifier, ...] = ()
    continuity_candidates: tuple[DocumentVersion, ...] = ()
    content_hash_candidates: tuple[DocumentVersion, ...] = ()
    external_identifier_candidates: tuple[DocumentVersion, ...] = ()
    identifier_assertions: tuple[ExternalIdentifierAssertion, ...] = ()
    external_identifier_policy: ExternalIdentifierResolutionPolicy = (
        CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the request fields and candidate evidence roles."""
        if not isinstance(self.content_hash, ContentHash):
            raise TypeError("content_hash must be a ContentHash.")
        if not isinstance(self.source_observation_id, SourceObservationId):
            raise TypeError("source_observation_id must be a SourceObservationId.")
        validate_external_identifiers(self.external_identifiers)
        _validate_candidates(self.continuity_candidates)
        _validate_candidates(self.content_hash_candidates)
        _validate_candidates(self.external_identifier_candidates)
        _validate_identifier_assertions(self.identifier_assertions)
        if not isinstance(
            self.external_identifier_policy, ExternalIdentifierResolutionPolicy
        ):
            raise TypeError(
                "external_identifier_policy must be an "
                "ExternalIdentifierResolutionPolicy."
            )
        self._validate_candidate_roles()
        candidate_version_ids = {
            candidate.document_version_id for candidate in self.all_candidates
        }
        if any(
            assertion.document_version_id not in candidate_version_ids
            for assertion in self.identifier_assertions
        ):
            raise InvalidResolutionError(
                "identifier assertions must refer to supplied candidate versions."
            )

    def _validate_candidate_roles(self) -> None:
        """Validate candidate evidence and cross-role representations."""
        if any(
            candidate.content_hash != self.content_hash
            for candidate in self.content_hash_candidates
        ):
            raise InvalidResolutionError(
                "content_hash_candidates must match the requested content hash."
            )

        representations: dict[DocumentVersionId, DocumentVersion] = {}
        for role in (
            self.continuity_candidates,
            self.content_hash_candidates,
            self.external_identifier_candidates,
        ):
            for candidate in role:
                existing = representations.setdefault(
                    candidate.document_version_id, candidate
                )
                if existing != candidate:
                    raise InvalidResolutionError(
                        "candidate roles must use one representation per version."
                    )

    @property
    def all_candidates(self) -> tuple[DocumentVersion, ...]:
        """Return unique candidates across evidence roles in canonical order."""
        candidates = {
            candidate.document_version_id: candidate
            for role in (
                self.continuity_candidates,
                self.content_hash_candidates,
                self.external_identifier_candidates,
            )
            for candidate in role
        }
        return tuple(
            sorted(
                candidates.values(),
                key=lambda candidate: candidate.document_version_id.as_str(),
            )
        )

    def as_dict(self) -> dict[str, object]:
        """Return deterministic, JSON-compatible request data."""
        return {
            "content_hash": self.content_hash.as_str(),
            "source_observation_id": self.source_observation_id.as_str(),
            "external_identifiers": [
                {"namespace": item.namespace, "value": item.value}
                for item in self.external_identifiers
            ],
            "continuity_candidates": self._candidate_payloads(
                self.continuity_candidates
            ),
            "content_hash_candidates": self._candidate_payloads(
                self.content_hash_candidates
            ),
            "external_identifier_candidates": self._candidate_payloads(
                self.external_identifier_candidates
            ),
            "identifier_assertions": [
                {
                    "source_observation_id": assertion.source_observation_id.as_str(),
                    "document_version_id": assertion.document_version_id.as_str(),
                    "external_identifier": {
                        "namespace": assertion.external_identifier.namespace,
                        "value": assertion.external_identifier.value,
                    },
                }
                for assertion in self.identifier_assertions
            ],
            "external_identifier_policy": {
                "policy_name": self.external_identifier_policy.policy_name,
                "policy_version": self.external_identifier_policy.policy_version,
                "authoritative_namespaces": list(
                    self.external_identifier_policy.authoritative_namespaces
                ),
            },
        }

    @staticmethod
    def _candidate_payloads(
        candidates: tuple[DocumentVersion, ...],
    ) -> list[dict[str, object]]:
        """Return deterministic candidate payloads for one evidence role."""
        return [
            {
                "document_id": candidate.document_id.as_str(),
                "document_version_id": candidate.document_version_id.as_str(),
                "content_hash": candidate.content_hash.as_str(),
                "external_identifiers": [
                    {"namespace": item.namespace, "value": item.value}
                    for item in candidate.external_identifiers
                ],
            }
            for candidate in candidates
        ]


@dataclass(frozen=True, slots=True)
class IdentityResolutionOutcome:
    """Validated result of one deterministic identity-resolution operation."""

    kind: ResolutionKind
    reason: ResolutionReason
    document: DocumentIdentity | None = None
    version: DocumentVersion | None = None
    matched_version_ids: tuple[DocumentVersionId, ...] = ()
    duplicate_relationships: tuple[DuplicateRelationship, ...] = ()
    identifier_assertions_to_add: tuple[ExternalIdentifierAssertion, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the outcome fields and relationships."""
        if not isinstance(self.kind, ResolutionKind):
            raise TypeError("kind must be a ResolutionKind.")
        if not isinstance(self.reason, ResolutionReason):
            raise TypeError("reason must be a ResolutionReason.")
        if self.document is not None and not isinstance(
            self.document, DocumentIdentity
        ):
            raise TypeError("document must be a DocumentIdentity or None.")
        if self.version is not None and not isinstance(self.version, DocumentVersion):
            raise TypeError("version must be a DocumentVersion or None.")
        _validate_matched_versions(self.matched_version_ids)
        if not isinstance(self.duplicate_relationships, tuple):
            raise TypeError("duplicate_relationships must be a tuple.")
        if any(
            not isinstance(relationship, DuplicateRelationship)
            for relationship in self.duplicate_relationships
        ):
            raise TypeError(
                "duplicate_relationships must contain only DuplicateRelationship values."
            )
        _validate_identifier_assertions(self.identifier_assertions_to_add)
        self._validate_semantics()

    def _validate_semantics(self) -> None:
        """Validate the fields required by the selected outcome kind."""
        if self.document is not None and self.version is not None:
            if self.document.document_id != self.version.document_id:
                raise InvalidResolutionError(
                    "document and version must identify the same document."
                )

        if self.kind is ResolutionKind.UNCHANGED:
            _, version = self._require_assigned()
            self._require_reason(ResolutionReason.EXACT_HASH_MATCH)
            if self.matched_version_ids != (version.document_version_id,):
                raise InvalidResolutionError(
                    "unchanged resolution must match exactly one retained version."
                )
            self._require_no_relationship()
            if any(
                assertion.document_version_id != version.document_version_id
                for assertion in self.identifier_assertions_to_add
            ):
                raise InvalidResolutionError(
                    "identifier assertions must target the retained version."
                )
            existing_namespaces = {
                identifier.namespace for identifier in version.external_identifiers
            }
            if any(
                assertion.external_identifier.namespace in existing_namespaces
                for assertion in self.identifier_assertions_to_add
            ):
                raise InvalidResolutionError(
                    "identifier assertions must add previously unasserted namespaces."
                )
        elif self.kind is ResolutionKind.NEW_VERSION:
            self._require_assigned()
            if self.reason not in {
                ResolutionReason.SOURCE_CONTINUITY_MATCH,
                ResolutionReason.EXTERNAL_IDENTIFIER_MATCH,
            }:
                raise InvalidResolutionError(
                    "new_version resolution requires source continuity or "
                    "external-identifier evidence."
                )
            if not self.matched_version_ids:
                raise InvalidResolutionError(
                    "new-version resolution requires a matching candidate."
                )
            self._require_no_relationship()
            self._require_no_assertions()
        elif self.kind is ResolutionKind.NEW_DOCUMENT:
            self._require_assigned()
            self._require_reason(ResolutionReason.NO_MATCH)
            if self.matched_version_ids:
                raise InvalidResolutionError(
                    "new-document resolution cannot have matching candidates."
                )
            self._require_no_relationship()
            self._require_no_assertions()
        elif self.kind is ResolutionKind.DUPLICATE:
            _, version = self._require_assigned()
            self._require_reason(ResolutionReason.DUPLICATE_HASH_MATCH)
            relationships = self.duplicate_relationships
            if not self.matched_version_ids or not relationships:
                raise InvalidResolutionError(
                    "duplicate resolution requires candidates and relationships."
                )
            relationship_keys = tuple(
                relationship.key() for relationship in relationships
            )
            if len(relationship_keys) != len(set(relationship_keys)):
                raise InvalidResolutionError("duplicate relationships must be unique.")
            for relationship in relationships:
                endpoints = (
                    relationship.first_version_id,
                    relationship.second_version_id,
                )
                if version.document_version_id not in endpoints:
                    raise InvalidResolutionError(
                        "duplicate relationships must include the resolved version."
                    )
                if not any(
                    version_id in endpoints for version_id in self.matched_version_ids
                ):
                    raise InvalidResolutionError(
                        "duplicate relationships must include matched versions."
                    )
            self._require_no_assertions()
        elif self.kind is ResolutionKind.AMBIGUOUS:
            self._require_unassigned()
            if self.reason not in {
                ResolutionReason.MULTIPLE_CANDIDATES,
                ResolutionReason.UNESTABLISHED_IDENTIFIER_AUTHORITY,
            }:
                raise InvalidResolutionError(
                    "ambiguous resolution requires multiple candidates or "
                    "unestablished identifier authority."
                )
            if not self.matched_version_ids:
                raise InvalidResolutionError(
                    "ambiguous resolution requires candidate versions."
                )
            self._require_no_assertions()
        elif self.kind is ResolutionKind.CONFLICT:
            self._require_unassigned()
            self._require_reason(ResolutionReason.CONFLICTING_EVIDENCE)
            self._require_no_assertions()
        else:
            raise InvalidResolutionError("unsupported resolution kind.")

    def _require_assigned(self) -> tuple[DocumentIdentity, DocumentVersion]:
        if self.document is None or self.version is None:
            raise InvalidResolutionError(
                f"{self.kind.value} resolution requires document and version."
            )
        return self.document, self.version

    def _require_unassigned(self) -> None:
        if self.document is not None or self.version is not None:
            raise InvalidResolutionError(
                f"{self.kind.value} resolution cannot assign document or version."
            )
        self._require_no_relationship()

    def _require_reason(self, expected: ResolutionReason) -> None:
        if self.reason is not expected:
            raise InvalidResolutionError(
                f"{self.kind.value} resolution requires reason {expected.value}."
            )

    def _require_no_relationship(self) -> None:
        if self.duplicate_relationships:
            raise InvalidResolutionError(
                f"{self.kind.value} resolution cannot include duplicate relationships."
            )

    def _require_no_assertions(self) -> None:
        if self.identifier_assertions_to_add:
            raise InvalidResolutionError(
                f"{self.kind.value} resolution cannot add identifier assertions."
            )

    def as_dict(self) -> dict[str, object]:
        """Return deterministic, JSON-compatible outcome data."""
        return {
            "kind": self.kind.value,
            "reason": self.reason.value,
            "document_id": (
                self.document.document_id.as_str()
                if self.document is not None
                else None
            ),
            "document_version_id": (
                self.version.document_version_id.as_str()
                if self.version is not None
                else None
            ),
            "matched_version_ids": [
                version_id.as_str() for version_id in self.matched_version_ids
            ],
            "duplicate_relationships": [
                {
                    "first_version_id": relationship.first_version_id.as_str(),
                    "second_version_id": relationship.second_version_id.as_str(),
                    "duplicate_class": relationship.duplicate_class.value,
                }
                for relationship in self.duplicate_relationships
            ],
            "identifier_assertions_to_add": [
                {
                    "source_observation_id": assertion.source_observation_id.as_str(),
                    "document_version_id": assertion.document_version_id.as_str(),
                    "external_identifier": {
                        "namespace": assertion.external_identifier.namespace,
                        "value": assertion.external_identifier.value,
                    },
                }
                for assertion in self.identifier_assertions_to_add
            ],
        }
