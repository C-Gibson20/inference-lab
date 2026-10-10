"""Privacy-safe adapters from ingest values into corpus-audit observations."""

from __future__ import annotations

from inference_lab.evaluation.audit_contracts import (
    AcquisitionStatus,
    AuditAcquisitionObservation,
    AuditMissingReason,
    AuditParseEvidence,
    AuditParseObservation,
    AuditSourceInventory,
    CorpusAuditScope,
)
from inference_lab.evaluation.audit_errors import (
    AuditContractError,
    AuditErrorCategory,
)
from inference_lab.ingest.classification import ClassificationInventory
from inference_lab.ingest.discovery_results import DiscoveryResult
from inference_lab.ingest.outcomes import ParseOutcome
from inference_lab.ingest.parse_persistence import ParsePersistenceRequest
from inference_lab.ingest.parse_reconstruction import ReconstructedParseOutcome
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    DocumentVersionId,
)


def adapt_source_inventory(
    classification: ClassificationInventory,
    discovery: DiscoveryResult,
) -> AuditSourceInventory:
    """Adapt complete discovery evidence while dropping source-level metadata."""
    return AuditSourceInventory.from_ingest(classification, discovery)


def adapt_parse_persistence_request(
    request: ParsePersistenceRequest,
    *,
    membership: CorpusDocumentVersionMembership,
) -> AuditParseObservation:
    """Adapt one corpus-authorised parser attempt.

    The source bytes are inspected only to derive their byte count.  The
    returned observation retains neither the parser input nor any source
    payload, and the caller must supply the asserted corpus membership.
    """
    if not isinstance(request, ParsePersistenceRequest):
        raise TypeError("request must be a ParsePersistenceRequest.")
    if not isinstance(membership, CorpusDocumentVersionMembership):
        raise TypeError("membership must be a CorpusDocumentVersionMembership.")
    if request.corpus_id != membership.corpus_id:
        raise AuditContractError(AuditErrorCategory.CORPUS_MISMATCH)
    parser_input = request.parser_input
    if (
        parser_input.document_id != membership.document_id
        or parser_input.document_version_id != membership.document_version_id
    ):
        raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
    return _build_parse_observation(
        membership=membership,
        parser_version=request.parser_version,
        media_type=parser_input.media_type,
        source_byte_count=len(parser_input.content),
        outcome=request.outcome,
    )


def adapt_reconstructed_parse_outcome(
    reconstructed: ReconstructedParseOutcome,
    *,
    membership: CorpusDocumentVersionMembership,
    media_type: str | None = None,
    source_byte_count: int | None = None,
) -> AuditParseObservation:
    """Adapt a storage-reconstructed parse outcome without storage leakage."""
    if not isinstance(reconstructed, ReconstructedParseOutcome):
        raise TypeError("reconstructed must be a ReconstructedParseOutcome.")
    if not isinstance(membership, CorpusDocumentVersionMembership):
        raise TypeError("membership must be a CorpusDocumentVersionMembership.")
    return _build_parse_observation(
        membership=membership,
        parser_version=reconstructed.parser_version,
        media_type=media_type,
        source_byte_count=source_byte_count,
        outcome=reconstructed.outcome,
    )


def complete_parse_observations(
    scope: CorpusAuditScope,
    observed: tuple[AuditParseObservation, ...],
    *,
    missing_reason: AuditMissingReason = AuditMissingReason.PARSE_NOT_ATTEMPTED,
) -> tuple[AuditParseObservation, ...]:
    """Fill absent member observations in the scope's canonical order."""
    if not isinstance(scope, CorpusAuditScope):
        raise TypeError("scope must be a CorpusAuditScope.")
    if not isinstance(observed, tuple):
        raise TypeError("observed must be a tuple.")
    if not all(isinstance(item, AuditParseObservation) for item in observed):
        raise TypeError("observed must contain AuditParseObservation values.")
    if not isinstance(missing_reason, AuditMissingReason):
        raise TypeError("missing_reason must be an AuditMissingReason.")

    by_version: dict[DocumentVersionId, AuditParseObservation] = {}
    for observation in observed:
        if observation.membership not in scope.memberships:
            raise AuditContractError(AuditErrorCategory.MEMBERSHIP_MISMATCH)
        version_id = observation.membership.document_version_id
        if version_id in by_version:
            raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
        by_version[version_id] = observation

    return tuple(
        by_version.get(
            membership.document_version_id,
            AuditParseObservation(
                membership=membership,
                missing_reason=missing_reason,
            ),
        )
        for membership in scope.memberships
    )


def adapt_text_native_acquisition(
    observation: AuditParseObservation,
) -> AuditAcquisitionObservation | None:
    """Assess a positively parsed text-native member when evidence permits it."""
    if not isinstance(observation, AuditParseObservation):
        raise TypeError("observation must be an AuditParseObservation.")
    if (
        observation.parser_version is None
        or observation.evidence is None
        or observation.evidence.document is None
        or observation.media_type != "text/plain"
    ):
        return None
    parser_version = observation.parser_version
    return AuditAcquisitionObservation(
        membership=observation.membership,
        status=AcquisitionStatus.TEXT_NATIVE,
        producer_name=parser_version.implementation,
        producer_version=parser_version.contract_version,
    )


def _build_parse_observation(
    *,
    membership: CorpusDocumentVersionMembership,
    parser_version: ParserVersion,
    media_type: str | None,
    source_byte_count: int | None,
    outcome: ParseOutcome,
) -> AuditParseObservation:
    if not isinstance(membership, CorpusDocumentVersionMembership):
        raise TypeError("membership must be a CorpusDocumentVersionMembership.")
    if not isinstance(parser_version, ParserVersion):
        raise TypeError("parser_version must be a ParserVersion.")
    if not isinstance(outcome, ParseOutcome):
        raise TypeError("outcome must be a ParseOutcome.")
    document = outcome.document
    if document is not None and (
        document.document_id != membership.document_id
        or document.document_version_id != membership.document_version_id
        or document.parser_version != parser_version
    ):
        raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
    return AuditParseObservation(
        membership=membership,
        parser_version=parser_version,
        media_type=media_type,
        source_byte_count=source_byte_count,
        evidence=AuditParseEvidence.from_ingest(outcome),
    )
