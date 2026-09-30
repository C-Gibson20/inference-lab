"""Stable identities and source-span lineage for corpus and derived records."""

from inference_lab.provenance._deserialisation import (
    deserialise_derived_record,
    deserialise_lineage_bundle,
)
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
    SourceObservationId,
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
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
    SourceObservationState,
)
from inference_lab.provenance.relationships import (
    CorpusDocumentVersionMembership,
    DocumentFamilyMembership,
    DuplicateRelationship,
    InvalidRelationshipError,
)
from inference_lab.provenance.resolution import (
    CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY,
    ExternalIdentifierAssertion,
    ExternalIdentifierResolutionPolicy,
    IdentityIdCollisionError,
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
    "CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY",
    "CURRENT_LINEAGE_SCHEMA_VERSION",
    "SUPPORTED_LINEAGE_SCHEMA_VERSIONS",
    "ComponentRole",
    "ConfigurationIdentity",
    "ContentHash",
    "CorpusDocumentVersionMembership",
    "CorpusId",
    "CorpusMetadata",
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
    "ExternalIdentifierAssertion",
    "ExternalIdentifierResolutionPolicy",
    "HashAlgorithm",
    "IdentityIdAllocator",
    "IdentityIdCollisionError",
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
    "SourceObservationId",
    "SourceObservationState",
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
