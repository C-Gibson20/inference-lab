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
from inference_lab.storage.protocols import (
    MetadataReader,
    MetadataRepository,
    MetadataTransaction,
    MetadataWriter,
)
from inference_lab.storage.source_records import SourceArtefact
from inference_lab.storage.source_serialisation import serialise_source_artefact
from inference_lab.storage.source_versioning import (
    CURRENT_SOURCE_RECORD_SCHEMA_VERSION,
    SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS,
    validate_source_record_schema_version,
)

__all__ = [
    "CURRENT_SOURCE_RECORD_SCHEMA_VERSION",
    "SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS",
    "CorruptStorageError",
    "IncompatibleSchemaError",
    "MetadataReader",
    "MetadataRepository",
    "MetadataTransaction",
    "MetadataWriter",
    "MigrationRequiredError",
    "RepositoryClosedError",
    "RepositoryUnavailableError",
    "SourceArtefact",
    "StorageConcurrencyError",
    "StorageConflictError",
    "StorageError",
    "StorageErrorCategory",
    "StorageMigrationError",
    "StorageTransactionError",
    "open_metadata_repository",
    "serialise_source_artefact",
    "validate_source_record_schema_version",
]
