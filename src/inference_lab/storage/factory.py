"""Public construction boundary for local metadata repositories."""

from pathlib import Path

from inference_lab.storage._sqlite_repository import (
    DEFAULT_SOURCE_RECONCILIATION_MAX_ENTRIES,
    SQLiteMetadataRepository,
)
from inference_lab.storage.protocols import MetadataRepository


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
    return SQLiteMetadataRepository.open(
        metadata_db_path,
        apply_migrations=apply_migrations,
        source_dir=source_dir,
        source_reconciliation_max_entries=source_reconciliation_max_entries,
    )
