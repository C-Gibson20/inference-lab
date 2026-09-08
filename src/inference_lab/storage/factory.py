"""Public construction boundary for local metadata repositories."""

from pathlib import Path

from inference_lab._validation import paths_overlap
from inference_lab.storage._sqlite_repository import (
    DEFAULT_SOURCE_RECONCILIATION_MAX_ENTRIES,
    SQLiteMetadataRepository,
)
from inference_lab.storage.errors import RepositoryUnavailableError
from inference_lab.storage.protocols import MetadataRepository


def _resolved_path_for_separation(path: Path, *, field_name: str) -> Path:
    """Resolve one storage path without retaining it in a public failure."""
    if not isinstance(path, Path):
        raise TypeError(f"{field_name} must be a Path.")
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        raise RepositoryUnavailableError() from None


def _validate_resolved_path_separation(
    metadata_db_path: Path,
    source_dir: Path,
) -> None:
    """Reject filesystem aliases and nesting before opening storage."""
    resolved_database = _resolved_path_for_separation(
        metadata_db_path,
        field_name="path",
    )
    resolved_sources = _resolved_path_for_separation(
        source_dir,
        field_name="source_dir",
    )
    if paths_overlap(resolved_database, resolved_sources):
        raise ValueError("metadata_db_path and source_dir must not overlap.")


def open_metadata_repository(
    metadata_db_path: Path,
    *,
    source_dir: Path,
    apply_migrations: bool = False,
    source_reconciliation_max_entries: int = (
        DEFAULT_SOURCE_RECONCILIATION_MAX_ENTRIES
    ),
) -> MetadataRepository:
    """Open the configured local metadata store under an explicit schema policy."""
    _validate_resolved_path_separation(metadata_db_path, source_dir)
    return SQLiteMetadataRepository.open(
        metadata_db_path,
        apply_migrations=apply_migrations,
        source_dir=source_dir,
        source_reconciliation_max_entries=source_reconciliation_max_entries,
    )
