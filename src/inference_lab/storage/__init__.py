"""Persistent corpus metadata, indexes, and version-aware storage boundaries."""

from inference_lab.storage.protocols import (
    MetadataReader,
    MetadataRepository,
    MetadataTransaction,
    MetadataWriter,
)

__all__ = [
    "MetadataReader",
    "MetadataRepository",
    "MetadataTransaction",
    "MetadataWriter",
]
