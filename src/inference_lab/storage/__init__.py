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
    "ComponentGenerationReference",
    "ComponentGenerationState",
    "ComponentKind",
    "CorpusUpdate",
    "CorpusUpdateState",
    "CorpusVersion",
    "CorpusVersionManifest",
    "CorruptStorageError",
    "IncompatibleSchemaError",
    "InvalidCorpusVersionContractError",
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
    "open_metadata_repository",
]
