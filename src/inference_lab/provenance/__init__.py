"""Stable identities and source-span lineage for corpus and derived records."""

from inference_lab.provenance.identifiers import (
    CorpusId,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.records import (
    DocumentIdentity,
    DocumentVersion,
)
from inference_lab.provenance.relationships import (
    DocumentFamilyMembership,
    DuplicateRelationship,
    InvalidRelationshipError,
)
from inference_lab.provenance.resolution import (
    IdentityResolutionOutcome,
    IdentityResolutionRequest,
    InvalidResolutionError,
    ResolutionReason,
)
from inference_lab.provenance.values import (
    ContentHash,
    DuplicateClass,
    ExternalIdentifier,
    HashAlgorithm,
    ResolutionKind,
)

__all__ = [
    "ContentHash",
    "CorpusId",
    "CorpusVersionId",
    "DocumentFamilyId",
    "DocumentFamilyMembership",
    "DocumentId",
    "DocumentIdentity",
    "DocumentVersion",
    "DocumentVersionId",
    "DuplicateClass",
    "DuplicateRelationship",
    "ExternalIdentifier",
    "HashAlgorithm",
    "IdentityResolutionOutcome",
    "IdentityResolutionRequest",
    "InvalidRelationshipError",
    "InvalidResolutionError",
    "RecordId",
    "ResolutionKind",
    "ResolutionReason",
    "SourceSpanId",
]
