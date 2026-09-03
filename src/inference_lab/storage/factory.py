"""Public construction boundary for local metadata repositories."""

from pathlib import Path

from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from inference_lab.storage.protocols import MetadataRepository


def open_metadata_repository(
    metadata_db_path: Path,
    *,
    apply_migrations: bool = False,
) -> MetadataRepository:
    """Open the configured local metadata store under an explicit schema policy."""
    return SQLiteMetadataRepository.open(
        metadata_db_path,
        apply_migrations=apply_migrations,
    )
