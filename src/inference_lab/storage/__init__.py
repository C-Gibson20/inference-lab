"""Persistent corpus metadata, indexes, and version-aware storage boundaries."""

from inference_lab.storage.errors import (
    CorruptStorageError,
    IncompatibleSchemaError,
    MigrationRequiredError,
    RepositoryClosedError,
    RepositoryUnavailableError,
    StorageConcurrencyError,
    StorageConflictError,
    StorageError,
    StorageErrorCategory,
    StorageMigrationError,
    StorageTransactionError,
)
from inference_lab.storage.factory import open_metadata_repository
from inference_lab.storage.outcomes import (
    AbortCorpusUpdateOutcome,
    BeginCorpusUpdateOutcome,
    CommitCorpusUpdateOutcome,
    CorpusVersionConflictKind,
    CorpusVersionConflictOutcome,
    NoCorpusChangeOutcome,
    RollbackCorpusVersionOutcome,
    SealCorpusUpdateOutcome,
)
from inference_lab.storage.parse_records import (
    ParseFailureCategory,
    ParseFailureRecord,
    ParseFailureSeverity,
    ParseFailureStage,
    ParseFailureUnitKind,
    ParseOutcome,
    ParseStatus,
)
from inference_lab.storage.parse_serialisation import serialise_parse_outcome
from inference_lab.storage.parsed_records import (
    ParsedRepresentation,
    build_parsed_representation_output_digest,
)
from inference_lab.storage.parsed_serialisation import (
    serialise_parsed_representation,
)
from inference_lab.storage.producers import ProducerIdentity
from inference_lab.storage.protocols import (
    ComponentGenerationValidator,
    MetadataReader,
    MetadataRepository,
    MetadataTransaction,
    MetadataWriter,
)
from inference_lab.storage.snapshots import CorpusSnapshot
from inference_lab.storage.source_locations import (
    BoundingBox,
    CollectionAvailability,
    ContentOrigin,
    CoordinateOrigin,
    CoordinateUnit,
    PageLayoutDescriptor,
    PageRotation,
    ResolvedSourceSpan,
    SourceLocation,
    StructuralBlock,
    StructuralBlockKind,
)
from inference_lab.storage.source_records import SourceArtefact
from inference_lab.storage.source_serialisation import serialise_source_artefact
from inference_lab.storage.source_versioning import (
    CURRENT_SOURCE_RECORD_SCHEMA_VERSION,
    SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS,
    validate_source_record_schema_version,
)
from inference_lab.storage.versioning import (
    CORPUS_MANIFEST_CONTRACT_VERSION,
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusUpdate,
    CorpusUpdateState,
    CorpusVersion,
    CorpusVersionManifest,
    InvalidCorpusVersionContractError,
)

__all__ = [
    "CORPUS_MANIFEST_CONTRACT_VERSION",
    "CURRENT_SOURCE_RECORD_SCHEMA_VERSION",
    "SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS",
    "AbortCorpusUpdateOutcome",
    "BeginCorpusUpdateOutcome",
    "BoundingBox",
    "CollectionAvailability",
    "CommitCorpusUpdateOutcome",
    "ComponentGenerationReference",
    "ComponentGenerationState",
    "ComponentGenerationValidator",
    "ComponentKind",
    "ContentOrigin",
    "CoordinateOrigin",
    "CoordinateUnit",
    "CorpusSnapshot",
    "CorpusUpdate",
    "CorpusUpdateState",
    "CorpusVersion",
    "CorpusVersionConflictKind",
    "CorpusVersionConflictOutcome",
    "CorpusVersionManifest",
    "CorruptStorageError",
    "IncompatibleSchemaError",
    "InvalidCorpusVersionContractError",
    "MetadataReader",
    "MetadataRepository",
    "MetadataTransaction",
    "MetadataWriter",
    "MigrationRequiredError",
    "NoCorpusChangeOutcome",
    "PageLayoutDescriptor",
    "PageRotation",
    "ParseFailureCategory",
    "ParseFailureRecord",
    "ParseFailureSeverity",
    "ParseFailureStage",
    "ParseFailureUnitKind",
    "ParseOutcome",
    "ParseStatus",
    "ParsedRepresentation",
    "ProducerIdentity",
    "RepositoryClosedError",
    "RepositoryUnavailableError",
    "ResolvedSourceSpan",
    "RollbackCorpusVersionOutcome",
    "SealCorpusUpdateOutcome",
    "SourceArtefact",
    "SourceLocation",
    "StorageConcurrencyError",
    "StorageConflictError",
    "StorageError",
    "StorageErrorCategory",
    "StorageMigrationError",
    "StorageTransactionError",
    "StructuralBlock",
    "StructuralBlockKind",
    "build_parsed_representation_output_digest",
    "open_metadata_repository",
    "serialise_parse_outcome",
    "serialise_parsed_representation",
    "serialise_source_artefact",
    "validate_source_record_schema_version",
]
