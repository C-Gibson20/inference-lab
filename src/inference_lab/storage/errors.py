"""Typed, privacy-safe failures for persistent storage operations."""

from enum import StrEnum
from typing import ClassVar


class StorageErrorCategory(StrEnum):
    """Stable categories suitable for bounded operational diagnostics."""

    REPOSITORY_UNAVAILABLE = "repository_unavailable"
    REPOSITORY_CLOSED = "repository_closed"
    CONFLICT = "conflict"
    CONCURRENCY = "concurrency"
    TRANSACTION = "transaction"
    CORRUPT_STATE = "corrupt_state"
    MIGRATION = "migration"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    MIGRATION_REQUIRED = "migration_required"


class StorageError(RuntimeError):
    """Base class for failures crossing the storage adapter boundary.

    Storage errors deliberately use fixed public messages. Concrete adapters
    must not propagate database messages, statements, values, or paths through
    this boundary.
    """

    category: ClassVar[StorageErrorCategory]
    default_message: ClassVar[str] = "A persistent storage operation failed."
    retryable: ClassVar[bool] = False

    def __init__(self) -> None:
        """Create an error without accepting potentially sensitive context."""
        super().__init__(self.default_message)


class RepositoryUnavailableError(StorageError):
    """Raised when the configured local repository cannot be opened or used."""

    category = StorageErrorCategory.REPOSITORY_UNAVAILABLE
    default_message = "The metadata repository is unavailable."


class RepositoryClosedError(StorageError):
    """Raised when an operation is attempted after repository closure."""

    category = StorageErrorCategory.REPOSITORY_CLOSED
    default_message = "The metadata repository has been closed."


class StorageConflictError(StorageError):
    """Raised when immutable data or a relationship key is reused differently."""

    category = StorageErrorCategory.CONFLICT
    default_message = "The storage operation conflicts with existing metadata."


class StorageConcurrencyError(StorageError):
    """Raised when bounded lock acquisition or concurrent access fails."""

    category = StorageErrorCategory.CONCURRENCY
    default_message = "The storage operation could not acquire bounded access."
    retryable = True


class StorageTransactionError(StorageError):
    """Raised when a transaction cannot start, commit, or roll back safely."""

    category = StorageErrorCategory.TRANSACTION
    default_message = "The storage transaction failed."


class CorruptStorageError(StorageError):
    """Raised when persisted state cannot be reconstructed safely."""

    category = StorageErrorCategory.CORRUPT_STATE
    default_message = "Persistent storage contains invalid state."


class StorageMigrationError(StorageError):
    """Raised when a supported metadata migration does not complete."""

    category = StorageErrorCategory.MIGRATION
    default_message = "The metadata schema migration failed."


class IncompatibleSchemaError(StorageMigrationError):
    """Raised when repository schema state is unsupported by this software."""

    category = StorageErrorCategory.INCOMPATIBLE_SCHEMA
    default_message = "The metadata schema is incompatible with this software."


class MigrationRequiredError(StorageMigrationError):
    """Raised when opening without migration finds a supported pending revision."""

    category = StorageErrorCategory.MIGRATION_REQUIRED
    default_message = "The metadata repository requires a supported migration."
