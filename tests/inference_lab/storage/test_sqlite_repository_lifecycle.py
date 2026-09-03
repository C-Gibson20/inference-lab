"""Tests for SQLite repository ownership and closure."""

from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import RepositoryClosedError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import fixed_applied_at as _fixed_applied_at  # noqa: TID252


def test_repository_owns_validated_schema_state(tmp_path: Path) -> None:
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert repository.is_closed is False
        assert repository.schema_inspection.current_revision == 1
        assert repository.schema_inspection.latest_supported_revision == 1
        assert repository.schema_inspection.applied_migration_count == 1
    finally:
        repository.close()


def test_repository_reopens_persisted_schema_with_a_fresh_owner(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    first = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    first.close()

    second = SQLiteMetadataRepository.open(database_path)
    try:
        assert second.is_closed is False
        assert second.schema_inspection.current_revision == 1
    finally:
        second.close()


def test_close_is_idempotent_and_emits_one_safe_lifecycle_event(
    tmp_path: Path,
) -> None:
    private_marker = "private-repository-path"
    repository = SQLiteMetadataRepository.open(
        tmp_path / private_marker / "metadata.sqlite3",
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )

    with capture_logs() as logs:
        repository.close()
        repository.close()

    assert repository.is_closed is True
    assert logs == [
        {
            "event": "repository_close_completed",
            "pipeline_stage": "storage",
            "log_level": "info",
        }
    ]
    assert private_marker not in repr(logs)


def test_schema_inspection_rejects_use_after_close(tmp_path: Path) -> None:
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    repository.close()

    with pytest.raises(RepositoryClosedError):
        _ = repository.schema_inspection


def test_repository_does_not_publish_a_connection_handle(tmp_path: Path) -> None:
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert not hasattr(repository, "connection")
        assert not hasattr(repository, "cursor")
    finally:
        repository.close()
