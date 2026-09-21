"""Pure helpers for serving ingest identities and bounded outcomes."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.ingest import (
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineFailure,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineStage,
    BaselineStageCounts,
    BaselineStageDisposition,
    DiscoveryFailure,
    DiscoveryRequest,
)
from inference_lab.ingest.discovery_failures import DiscoveryFailureStage
from inference_lab.ingest.discovery_results import DiscoveryResult
from inference_lab.provenance import (
    DigestAlgorithm,
    OutputDigest,
    RecordId,
    ResolutionKind,
)
from inference_lab.serving.ingest_contracts import ServingIngestRequest

_RECORD_NAMESPACE = uuid.UUID("6d9d6ff4-c92d-4b1b-8c71-bb2a1962f5f6")


def record_id(kind: str, *parts: str) -> RecordId:
    """Derive a stable record identity from immutable source-stage inputs."""
    return RecordId.from_str(
        str(uuid.uuid5(_RECORD_NAMESPACE, "\x00".join((kind, *parts))))
    )


def digest(value: JsonObject) -> OutputDigest:
    """Return the project-owned digest identity of one private summary."""
    return OutputDigest(
        algorithm=DigestAlgorithm.SHA256,
        digest=hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
    )


def discovery_request_payload(request: DiscoveryRequest) -> JsonObject:
    """Build a private, deterministic request identity without logging paths."""
    return {
        "corpus_id": request.corpus_id.as_str(),
        "external_identifier_policy": {
            "name": request.external_identifier_policy.policy_name,
            "version": request.external_identifier_policy.policy_version,
        },
        "inputs": [
            {
                "kind": item.kind.value,
                "root": str(item.root),
                "external_identifiers": [
                    {"namespace": identifier.namespace, "value": identifier.value}
                    for identifier in item.external_identifiers
                ],
                "family_candidates": [
                    {
                        "family_id": candidate.document_family_id.as_str(),
                        "evidence_kind": candidate.evidence_kind.value,
                        "document_id": (
                            candidate.document_id.as_str()
                            if candidate.document_id is not None
                            else None
                        ),
                        "document_version_id": (
                            candidate.document_version_id.as_str()
                            if candidate.document_version_id is not None
                            else None
                        ),
                    }
                    for candidate in item.family_candidates
                ],
            }
            for item in request.inputs
        ],
    }


def request_identity(request: ServingIngestRequest) -> OutputDigest:
    """Identify every configuration value that can change ingest output."""
    return digest(
        {
            "discovery": discovery_request_payload(request.discovery_request),
            "parser_version": request.parser_version.value,
            "chunker_version": {
                "implementation": request.chunker_version.implementation,
                "contract_version": request.chunker_version.contract_version,
            },
            "tokeniser_identity": {
                "implementation": request.tokeniser.identity.implementation,
                "contract_version": request.tokeniser.identity.contract_version,
                "artefact_identity": request.tokeniser.identity.artefact_identity,
                "artefact_revision": request.tokeniser.identity.artefact_revision,
            },
            "chunking_policy": {
                "min_tokens": request.chunking_policy.min_tokens,
                "max_tokens": request.chunking_policy.max_tokens,
                "overlap_numerator": request.chunking_policy.overlap_numerator,
                "overlap_denominator": request.chunking_policy.overlap_denominator,
                "structural_boundary_policy": request.chunking_policy.structural_boundary_policy.value,
                "oversize_unit_policy": request.chunking_policy.oversize_unit_policy.value,
            },
            "configuration_identity": request.configuration_identity.as_str(),
            "lexical_configuration": request.lexical_configuration.as_payload(),
        }
    )


def discovery_result_identity(result: DiscoveryResult) -> OutputDigest:
    """Identify discovered source outcomes without retaining source payload."""
    return digest(
        {
            "corpus_id": result.corpus_id.as_str(),
            "policy": {
                "name": result.policy.policy_name,
                "version": result.policy.policy_version,
                "classifier_name": result.policy.classifier_name,
                "classifier_version": result.policy.classifier_version,
            },
            "sources": [
                {
                    "ordinal": source.ordinal,
                    "observation_id": source.source_observation_id.as_str(),
                    "content_hash": source.snapshot.content_hash.as_str(),
                    "document_id": (
                        source.identity_outcome.document.document_id.as_str()
                        if source.identity_outcome.document is not None
                        else None
                    ),
                    "document_version_id": (
                        source.identity_outcome.version.document_version_id.as_str()
                        if source.identity_outcome.version is not None
                        else None
                    ),
                }
                for source in result.sources
            ],
            "failures": [
                {
                    "ordinal": failure.ordinal,
                    "stage": failure.stage.value,
                    "category": failure.category.value,
                }
                for failure in result.failures
            ],
        }
    )


def failure(
    *,
    stage: BaselineStage,
    category: BaselineFailureCategory,
    scope: BaselineFailureScope,
    severity: BaselineFailureSeverity,
    retryable: bool = False,
) -> BaselineFailure:
    """Build one bounded failure without retaining exception details."""
    return BaselineFailure(
        stage=stage,
        category=category,
        scope=scope,
        severity=severity,
        retryable=retryable,
    )


def discovery_failure_category(
    failure_value: DiscoveryFailure,
) -> BaselineFailureCategory:
    """Map discovery's detailed failures to the baseline closed vocabulary."""
    if failure_value.stage is DiscoveryFailureStage.HASHING:
        return BaselineFailureCategory.SOURCE_ACQUISITION_FAILED
    if failure_value.stage is DiscoveryFailureStage.PERSISTENCE:
        return BaselineFailureCategory.STORAGE_UNAVAILABLE
    return BaselineFailureCategory.DISCOVERY_FAILED


def resolution_kind(kind: ResolutionKind) -> BaselineDocumentOutcomeKind:
    """Map identity resolution to the baseline publication vocabulary."""
    return {
        ResolutionKind.UNCHANGED: BaselineDocumentOutcomeKind.UNCHANGED_REUSED,
        ResolutionKind.NEW_DOCUMENT: BaselineDocumentOutcomeKind.NEW_DOCUMENT,
        ResolutionKind.NEW_VERSION: BaselineDocumentOutcomeKind.CHANGED_VERSION,
        ResolutionKind.DUPLICATE: BaselineDocumentOutcomeKind.BYTE_IDENTICAL_DUPLICATE,
        ResolutionKind.AMBIGUOUS: BaselineDocumentOutcomeKind.AMBIGUOUS,
        ResolutionKind.CONFLICT: BaselineDocumentOutcomeKind.CONFLICTING,
    }[kind]


def parse_failure_category(category: object) -> BaselineFailureCategory:
    """Map a parse or chunk category without exposing the original exception."""
    value = getattr(category, "value", None)
    mapping = {
        "source_acquisition_failed": BaselineFailureCategory.SOURCE_ACQUISITION_FAILED,
        "parser_unavailable": BaselineFailureCategory.PARSER_UNAVAILABLE,
        "parse_failed": BaselineFailureCategory.PARSE_FAILED,
        "parse_persistence_failed": BaselineFailureCategory.PARSE_PERSISTENCE_FAILED,
        "chunk_failed": BaselineFailureCategory.CHUNK_FAILED,
        "chunk_persistence_failed": BaselineFailureCategory.CHUNK_PERSISTENCE_FAILED,
        "corrupt_state": BaselineFailureCategory.CORRUPT_STATE,
    }
    if not isinstance(value, str):
        return BaselineFailureCategory.CORRUPT_STATE
    return mapping.get(value, BaselineFailureCategory.CORRUPT_STATE)


def stage_count(
    stage: BaselineStage,
    outcomes: tuple[BaselineDocumentOutcome, ...],
    disposition: Callable[[BaselineDocumentOutcome], BaselineStageDisposition],
) -> BaselineStageCounts:
    """Count one stage's reuse, work, and failure outcomes."""
    return BaselineStageCounts(
        stage=stage,
        reused=sum(
            disposition(outcome) is BaselineStageDisposition.REUSED
            for outcome in outcomes
        ),
        worked=sum(
            disposition(outcome) is BaselineStageDisposition.COMPLETED
            for outcome in outcomes
        ),
        failed=sum(
            disposition(outcome) is BaselineStageDisposition.FAILED
            for outcome in outcomes
        ),
    )
