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
from inference_lab.storage.protocols import (
    MetadataReader,
    MetadataRepository,
    MetadataTransaction,
    MetadataWriter,
)

__all__ = [
    "CorruptStorageError",
    "IncompatibleSchemaError",
    "MetadataReader",
    "MetadataRepository",
    "MetadataTransaction",
    "MetadataWriter",
    "MigrationRequiredError",
    "RepositoryClosedError",
    "RepositoryUnavailableError",
    "StorageConcurrencyError",
    "StorageConflictError",
    "StorageError",
    "StorageErrorCategory",
    "StorageMigrationError",
    "StorageTransactionError",
]
