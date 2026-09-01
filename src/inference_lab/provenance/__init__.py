"""Stable identities and source-span lineage for corpus and derived records."""

from inference_lab.provenance.derived import (
    DerivationFingerprint,
    DerivedRecordKind,
    DerivedRecordMetadata,
    OutputDigest,
    build_derivation_fingerprint,
)
from inference_lab.provenance.errors import (
    InvalidDerivedRecordError,
    InvalidLineageError,
    InvalidLineageGraphError,
    InvalidLineageMetadataError,
    InvalidLineageReferenceError,
    InvalidLineageSerialisationError,
    UnsupportedLineageSchemaVersionError,
)
from inference_lab.provenance.identifiers import (
    CorpusId,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.lineage import (
    InputRole,
    LineageEdge,
    LineageInput,
    LineageInputs,
    ParentRecordReference,
    SourceAnchor,
)
from inference_lab.provenance.operations import (
    ComponentRole,
    ConfigurationIdentity,
    DigestAlgorithm,
    OperationKind,
    ProducingComponent,
    ProducingOperation,
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
from inference_lab.provenance.resolver import (
    IdentityIdAllocator,
    UuidIdentityIdAllocator,
    resolve_identity,
)
from inference_lab.provenance.serialisation import (
    deserialise_derived_record,
    deserialise_lineage_bundle,
    serialise_derived_record,
    serialise_lineage_bundle,
)
from inference_lab.provenance.validation import (
    KnownSourceSpan,
    LineageBundle,
    validate_lineage_bundle,
)
from inference_lab.provenance.values import (
    ContentHash,
    DuplicateClass,
    ExternalIdentifier,
    HashAlgorithm,
    ResolutionKind,
)
from inference_lab.provenance.versioning import (
    CURRENT_LINEAGE_SCHEMA_VERSION,
    SUPPORTED_LINEAGE_SCHEMA_VERSIONS,
    validate_lineage_schema_version,
)

__all__ = [
    "CURRENT_LINEAGE_SCHEMA_VERSION",
    "SUPPORTED_LINEAGE_SCHEMA_VERSIONS",
    "ComponentRole",
    "ConfigurationIdentity",
    "ContentHash",
    "CorpusId",
    "CorpusVersionId",
    "DerivationFingerprint",
    "DerivedRecordKind",
    "DerivedRecordMetadata",
    "DigestAlgorithm",
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
    "IdentityIdAllocator",
    "IdentityResolutionOutcome",
    "IdentityResolutionRequest",
    "InputRole",
    "InvalidDerivedRecordError",
    "InvalidLineageError",
    "InvalidLineageGraphError",
    "InvalidLineageMetadataError",
    "InvalidLineageReferenceError",
    "InvalidLineageSerialisationError",
    "InvalidRelationshipError",
    "InvalidResolutionError",
    "KnownSourceSpan",
    "LineageBundle",
    "LineageEdge",
    "LineageInput",
    "LineageInputs",
    "OperationKind",
    "OutputDigest",
    "ParentRecordReference",
    "ProducingComponent",
    "ProducingOperation",
    "RecordId",
    "ResolutionKind",
    "ResolutionReason",
    "SourceAnchor",
    "SourceSpanId",
    "UnsupportedLineageSchemaVersionError",
    "UuidIdentityIdAllocator",
    "build_derivation_fingerprint",
    "deserialise_derived_record",
    "deserialise_lineage_bundle",
    "resolve_identity",
    "serialise_derived_record",
    "serialise_lineage_bundle",
    "validate_lineage_bundle",
    "validate_lineage_schema_version",
]
