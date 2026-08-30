"""Contracts for deterministic document identity resolution."""

from dataclasses import dataclass
from enum import StrEnum

from inference_lab.provenance.identifiers import DocumentVersionId
from inference_lab.provenance.records import DocumentIdentity, DocumentVersion
from inference_lab.provenance.relationships import DuplicateRelationship
from inference_lab.provenance.utils import validate_external_identifiers
from inference_lab.provenance.values import (
    ContentHash,
    ExternalIdentifier,
    ResolutionKind,
)


class InvalidResolutionError(ValueError):
    """Raised when a resolution request or outcome violates its contract."""


class ResolutionReason(StrEnum):
    """Reasons explaining a deterministic resolution outcome."""

    EXACT_HASH_MATCH = "exact_hash_match"
    EXTERNAL_IDENTIFIER_MATCH = "external_identifier_match"
    NO_MATCH = "no_match"
    DUPLICATE_HASH_MATCH = "duplicate_hash_match"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    MULTIPLE_CANDIDATES = "multiple_candidates"


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
    external_identifiers: tuple[ExternalIdentifier, ...] = ()
    candidates: tuple[DocumentVersion, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.content_hash, ContentHash):
            raise TypeError("content_hash must be a ContentHash.")
        validate_external_identifiers(self.external_identifiers)
        _validate_candidates(self.candidates)

    def as_dict(self) -> dict[str, object]:
        """Return deterministic, JSON-compatible request data."""
        return {
            "content_hash": self.content_hash.as_str(),
            "external_identifiers": [
                {"namespace": item.namespace, "value": item.value}
                for item in self.external_identifiers
            ],
            "candidates": [
                {
                    "document_id": candidate.document_id.as_str(),
                    "document_version_id": candidate.document_version_id.as_str(),
                    "content_hash": candidate.content_hash.as_str(),
                    "external_identifiers": [
                        {"namespace": item.namespace, "value": item.value}
                        for item in candidate.external_identifiers
                    ],
                }
                for candidate in self.candidates
            ],
        }


@dataclass(frozen=True, slots=True)
class IdentityResolutionOutcome:
    """Validated result of one deterministic identity-resolution operation."""

    kind: ResolutionKind
    reason: ResolutionReason
    document: DocumentIdentity | None = None
    version: DocumentVersion | None = None
    matched_version_ids: tuple[DocumentVersionId, ...] = ()
    duplicate_relationship: DuplicateRelationship | None = None

    def __post_init__(self) -> None:
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
        if self.duplicate_relationship is not None and not isinstance(
            self.duplicate_relationship, DuplicateRelationship
        ):
            raise TypeError(
                "duplicate_relationship must be a DuplicateRelationship or None."
            )
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
        elif self.kind is ResolutionKind.NEW_VERSION:
            self._require_assigned()
            self._require_reason(ResolutionReason.EXTERNAL_IDENTIFIER_MATCH)
            if not self.matched_version_ids:
                raise InvalidResolutionError(
                    "new-version resolution requires a matching candidate."
                )
            self._require_no_relationship()
        elif self.kind is ResolutionKind.NEW_DOCUMENT:
            self._require_assigned()
            self._require_reason(ResolutionReason.NO_MATCH)
            if self.matched_version_ids:
                raise InvalidResolutionError(
                    "new-document resolution cannot have matching candidates."
                )
            self._require_no_relationship()
        elif self.kind is ResolutionKind.DUPLICATE:
            _, version = self._require_assigned()
            self._require_reason(ResolutionReason.DUPLICATE_HASH_MATCH)
            relationship = self.duplicate_relationship
            if not self.matched_version_ids or relationship is None:
                raise InvalidResolutionError(
                    "duplicate resolution requires a candidate and relationship."
                )
            endpoints = (
                relationship.first_version_id,
                relationship.second_version_id,
            )
            if version.document_version_id not in endpoints:
                raise InvalidResolutionError(
                    "duplicate relationship must include the resolved version."
                )
            if not any(
                version_id in endpoints for version_id in self.matched_version_ids
            ):
                raise InvalidResolutionError(
                    "duplicate relationship must include a matched version."
                )
        elif self.kind is ResolutionKind.AMBIGUOUS:
            self._require_unassigned()
            self._require_reason(ResolutionReason.MULTIPLE_CANDIDATES)
            if not self.matched_version_ids:
                raise InvalidResolutionError(
                    "ambiguous resolution requires candidate versions."
                )
        elif self.kind is ResolutionKind.CONFLICT:
            self._require_unassigned()
            self._require_reason(ResolutionReason.CONFLICTING_EVIDENCE)
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
        if self.duplicate_relationship is not None:
            raise InvalidResolutionError(
                f"{self.kind.value} resolution cannot include a duplicate relationship."
            )

    def as_dict(self) -> dict[str, object]:
        """Return deterministic, JSON-compatible outcome data."""
        relationship = self.duplicate_relationship
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
            "duplicate_relationship": (
                {
                    "first_version_id": relationship.first_version_id.as_str(),
                    "second_version_id": relationship.second_version_id.as_str(),
                    "duplicate_class": relationship.duplicate_class.value,
                }
                if relationship is not None
                else None
            ),
        }
