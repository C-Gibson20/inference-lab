"""Tests for storage error contracts."""

import pytest

from inference_lab.storage import (
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


@pytest.mark.parametrize(
    ("error_type", "category", "retryable"),
    [
        (
            RepositoryUnavailableError,
            StorageErrorCategory.REPOSITORY_UNAVAILABLE,
            False,
        ),
        (RepositoryClosedError, StorageErrorCategory.REPOSITORY_CLOSED, False),
        (StorageConflictError, StorageErrorCategory.CONFLICT, False),
        (StorageConcurrencyError, StorageErrorCategory.CONCURRENCY, True),
        (StorageTransactionError, StorageErrorCategory.TRANSACTION, False),
        (CorruptStorageError, StorageErrorCategory.CORRUPT_STATE, False),
        (StorageMigrationError, StorageErrorCategory.MIGRATION, False),
        (IncompatibleSchemaError, StorageErrorCategory.INCOMPATIBLE_SCHEMA, False),
        (MigrationRequiredError, StorageErrorCategory.MIGRATION_REQUIRED, False),
    ],
)
def test_storage_errors_have_stable_categories_and_retryability(
    error_type: type[StorageError],
    category: StorageErrorCategory,
    retryable: bool,
) -> None:
    error = error_type()

    assert isinstance(error, RuntimeError)
    assert error.category is category
    assert error.retryable is retryable
    assert str(error)


def test_schema_errors_share_the_migration_failure_boundary() -> None:
    assert issubclass(IncompatibleSchemaError, StorageMigrationError)
    assert issubclass(MigrationRequiredError, StorageMigrationError)


def test_storage_errors_do_not_accept_adapter_payloads() -> None:
    private_marker = "/private/corpus/secret.sqlite"

    with pytest.raises(TypeError) as captured:
        StorageTransactionError(private_marker)  # type: ignore[call-arg]

    assert private_marker not in str(captured.value)
