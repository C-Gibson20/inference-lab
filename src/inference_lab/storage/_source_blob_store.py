"""Internal managed storage for immutable original-source blobs."""

import hashlib
import os
import stat
from pathlib import Path
from typing import Final

from inference_lab._validation import is_local_path_shape
from inference_lab.provenance import ContentHash, HashAlgorithm
from inference_lab.storage.errors import CorruptStorageError
from inference_lab.storage.source_records import SourceArtefact

DEFAULT_BLOB_READ_CHUNK_SIZE: Final = 64 * 1024
_DIRECTORY_FLAGS: Final = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS: Final = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_HAS_SECURE_DIRECTORY_OPEN: Final = (
    os.open in os.supports_dir_fd
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
)


def build_source_blob_storage_key(content_hash: ContentHash) -> str:
    """Return the implementation-owned relative key for immutable source bytes."""
    if not isinstance(content_hash, ContentHash):
        raise TypeError("content_hash must be a ContentHash.")
    if content_hash.algorithm is not HashAlgorithm.SHA256:
        raise ValueError("managed source blobs require a SHA-256 content hash.")
    return f"sha256/{content_hash.digest[:2]}/{content_hash.digest}"


def _validate_chunk_size(chunk_size: int) -> None:
    """Validate one strict positive streaming chunk size."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")


def _is_link_or_junction(path: Path) -> bool:
    """Return whether a path redirects traversal through a filesystem link."""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


class ManagedSourceBlobStore:
    """Read immutable source blobs beneath one configured local root."""

    def __init__(self, source_dir: Path) -> None:
        if not isinstance(source_dir, Path):
            raise TypeError("source_dir must be a Path.")
        if not is_local_path_shape(source_dir):
            raise ValueError("source_dir must be a local filesystem path.")
        self._source_dir = source_dir

    def read_bytes(
        self,
        artefact: SourceArtefact,
        *,
        chunk_size: int = DEFAULT_BLOB_READ_CHUNK_SIZE,
    ) -> bytes:
        """Return one complete blob only after size and digest verification."""
        if not isinstance(artefact, SourceArtefact):
            raise TypeError("artefact must be a SourceArtefact.")
        _validate_chunk_size(chunk_size)
        expected_key = build_source_blob_storage_key(artefact.content_hash)
        if artefact.storage_key != expected_key:
            raise CorruptStorageError()

        descriptor: int | None = None
        try:
            descriptor = self._open_blob(expected_key)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise CorruptStorageError()
            if metadata.st_size != artefact.byte_count:
                raise CorruptStorageError()

            stream = os.fdopen(descriptor, "rb")
            descriptor = None
            with stream:
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                byte_count = 0
                while chunk := stream.read(chunk_size):
                    chunks.append(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)

            if (
                byte_count != artefact.byte_count
                or digest.hexdigest() != artefact.content_hash.digest
            ):
                raise CorruptStorageError()
        except CorruptStorageError:
            raise
        except (OSError, RuntimeError):
            raise CorruptStorageError() from None
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

        return b"".join(chunks)

    def _open_blob(self, storage_key: str) -> int:
        """Open a generated key without following managed-tree links."""
        parts = tuple(storage_key.split("/"))
        if _HAS_SECURE_DIRECTORY_OPEN:
            return self._open_blob_with_directory_descriptors(parts)
        return self._open_blob_with_path_checks(parts)

    def _open_blob_with_directory_descriptors(self, parts: tuple[str, ...]) -> int:
        """Open each managed path component relative to a pinned directory."""
        directory_descriptors: list[int] = []
        try:
            current = os.open(self._source_dir, _DIRECTORY_FLAGS)
            directory_descriptors.append(current)
            for part in parts[:-1]:
                current = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
                directory_descriptors.append(current)
            return os.open(parts[-1], _FILE_FLAGS, dir_fd=current)
        finally:
            for descriptor in reversed(directory_descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _open_blob_with_path_checks(self, parts: tuple[str, ...]) -> int:
        """Use containment and link checks where directory-relative open is absent."""
        source_dir = self._source_dir.expanduser()
        if _is_link_or_junction(source_dir):
            raise OSError
        resolved_root = source_dir.resolve(strict=True)
        if not resolved_root.is_dir():
            raise OSError

        candidate = resolved_root
        for part in parts:
            candidate /= part
            if _is_link_or_junction(candidate):
                raise OSError
        resolved_candidate = candidate.resolve(strict=True)
        if not resolved_candidate.is_relative_to(resolved_root):
            raise OSError
        return os.open(resolved_candidate, _FILE_FLAGS)
