"""Precomputed evidence for deterministic identity resolution."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from inference_lab.provenance.identifiers import DocumentId, DocumentVersionId
from inference_lab.provenance.records import DocumentVersion
from inference_lab.provenance.resolution import IdentityResolutionRequest
from inference_lab.provenance.values import ExternalIdentifier


def _identifier_values_by_namespace(
    identifiers: tuple[ExternalIdentifier, ...],
) -> dict[str, frozenset[str]]:
    """Group validated external-identifier values by namespace."""
    grouped: dict[str, set[str]] = {}
    for identifier in identifiers:
        grouped.setdefault(identifier.namespace, set()).add(identifier.value)
    return {namespace: frozenset(values) for namespace, values in grouped.items()}


def _effective_identifier_values(
    candidate: DocumentVersion,
    asserted_identifiers: tuple[ExternalIdentifier, ...],
) -> dict[str, frozenset[str]]:
    """Index base and append-only identifier evidence for one candidate."""
    identifiers = {
        identifier.as_tuple(): identifier
        for identifier in candidate.external_identifiers
    }
    for identifier in asserted_identifiers:
        identifiers[identifier.as_tuple()] = identifier
    return _identifier_values_by_namespace(tuple(identifiers.values()))


@dataclass(frozen=True, slots=True)
class _CandidateIdentifierEvidence:
    """Indexed identifier comparison for one candidate version."""

    values: Mapping[str, frozenset[str]]
    matching_namespaces: tuple[str, ...]
    conflicts: bool


def _candidate_identifier_evidence(
    request: IdentityResolutionRequest,
    candidates: tuple[DocumentVersion, ...],
    request_values: Mapping[str, frozenset[str]],
) -> dict[DocumentVersionId, _CandidateIdentifierEvidence]:
    """Build effective identifier comparisons for every candidate."""
    asserted_identifiers: dict[DocumentVersionId, list[ExternalIdentifier]] = {}
    for assertion in request.identifier_assertions:
        asserted_identifiers.setdefault(assertion.document_version_id, []).append(
            assertion.external_identifier
        )

    evidence: dict[DocumentVersionId, _CandidateIdentifierEvidence] = {}
    for candidate in candidates:
        version_id = candidate.document_version_id
        candidate_values = _effective_identifier_values(
            candidate,
            tuple(asserted_identifiers.get(version_id, ())),
        )
        shared_namespaces = request_values.keys() & candidate_values.keys()
        evidence[version_id] = _CandidateIdentifierEvidence(
            values=MappingProxyType(candidate_values),
            matching_namespaces=tuple(
                sorted(
                    namespace
                    for namespace in shared_namespaces
                    if request_values[namespace] == candidate_values[namespace]
                )
            ),
            conflicts=any(
                request_values[namespace] != candidate_values[namespace]
                for namespace in shared_namespaces
            ),
        )
    return evidence


@dataclass(frozen=True, slots=True)
class _ResolutionContext:
    """Canonical candidate and identifier views for one resolution."""

    request: IdentityResolutionRequest
    all_candidates: tuple[DocumentVersion, ...]
    existing_document_ids: frozenset[DocumentId]
    existing_version_ids: frozenset[DocumentVersionId]
    request_identifier_values: Mapping[str, frozenset[str]]
    candidate_identifier_evidence: Mapping[
        DocumentVersionId, _CandidateIdentifierEvidence
    ]
    continuity_hash_matches: tuple[DocumentVersion, ...]
    identifier_matches: tuple[DocumentVersion, ...]

    @classmethod
    def from_request(cls, request: IdentityResolutionRequest) -> "_ResolutionContext":
        """Build every reusable evidence view once for a validated request."""
        all_candidates = request.all_candidates
        request_identifier_values = _identifier_values_by_namespace(
            request.external_identifiers
        )
        candidate_identifier_evidence = _candidate_identifier_evidence(
            request,
            all_candidates,
            request_identifier_values,
        )

        return cls(
            request=request,
            all_candidates=all_candidates,
            existing_document_ids=frozenset(
                candidate.document_id for candidate in all_candidates
            ),
            existing_version_ids=frozenset(
                candidate.document_version_id for candidate in all_candidates
            ),
            request_identifier_values=MappingProxyType(request_identifier_values),
            candidate_identifier_evidence=MappingProxyType(
                candidate_identifier_evidence
            ),
            continuity_hash_matches=tuple(
                candidate
                for candidate in request.continuity_candidates
                if candidate.content_hash == request.content_hash
            ),
            identifier_matches=tuple(
                candidate
                for candidate in request.external_identifier_candidates
                if candidate_identifier_evidence[
                    candidate.document_version_id
                ].matching_namespaces
            ),
        )

    def document_ids(
        self, candidates: tuple[DocumentVersion, ...]
    ) -> tuple[DocumentId, ...]:
        """Return unique candidate document IDs in canonical order."""
        return tuple(
            sorted(
                {candidate.document_id for candidate in candidates},
                key=lambda document_id: document_id.as_str(),
            )
        )

    def version_ids(
        self, candidates: tuple[DocumentVersion, ...]
    ) -> tuple[DocumentVersionId, ...]:
        """Return unique candidate version IDs in canonical order."""
        return tuple(
            sorted(
                {candidate.document_version_id for candidate in candidates},
                key=lambda version_id: version_id.as_str(),
            )
        )

    def has_identifier_conflict(self, candidates: tuple[DocumentVersion, ...]) -> bool:
        """Return whether any candidate conflicts with asserted identifiers."""
        return any(
            self.candidate_identifier_evidence[candidate.document_version_id].conflicts
            for candidate in candidates
        )

    def candidate_namespaces(self, candidate: DocumentVersion) -> frozenset[str]:
        """Return namespaces asserted for one candidate version."""
        return frozenset(
            self.candidate_identifier_evidence[candidate.document_version_id].values
        )

    def matching_namespaces_for(self, candidate: DocumentVersion) -> tuple[str, ...]:
        """Return identifier namespaces that match one candidate."""
        return self.candidate_identifier_evidence[
            candidate.document_version_id
        ].matching_namespaces
