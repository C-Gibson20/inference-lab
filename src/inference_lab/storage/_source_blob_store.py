"""Internal managed storage for immutable original-source blobs."""

import hashlib
import os
import secrets
import stat
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final

from inference_lab._validation import is_local_path_shape
from inference_lab.provenance import ContentHash, HashAlgorithm
from inference_lab.storage.errors import (
    CorruptStorageError,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage.source_records import SourceArtefact

DEFAULT_BLOB_READ_CHUNK_SIZE: Final = 64 * 1024
_DIRECTORY_MODE: Final = 0o700
_BLOB_MODE: Final = 0o400
_STAGING_MODE: Final = 0o600
_DIRECTORY_FLAGS: Final = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS: Final = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_STAGING_FLAGS: Final = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
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
    """Read and durably publish source blobs beneath one configured local root."""

    def __init__(self, source_dir: Path) -> None:
        if not isinstance(source_dir, Path):
            raise TypeError("source_dir must be a Path.")
        if not is_local_path_shape(source_dir):
            raise ValueError("source_dir must be a local filesystem path.")
        try:
            self._source_dir = source_dir.expanduser()
        except RuntimeError:
            raise ValueError("source_dir must be a usable local path.") from None

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
            content = self._verify_descriptor(
                descriptor,
                artefact,
                chunk_size=chunk_size,
                retain_content=True,
            )
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

        return content

    def stream_bytes(
        self,
        artefact: SourceArtefact,
        *,
        chunk_size: int = DEFAULT_BLOB_READ_CHUNK_SIZE,
    ) -> Iterator[bytes]:
        """Return a chunk iterator only after complete blob verification."""
        content = self.read_bytes(artefact, chunk_size=chunk_size)
        return (
            content[offset : offset + chunk_size]
            for offset in range(0, len(content), chunk_size)
        )

    def write_chunks(
        self,
        artefact: SourceArtefact,
        chunks: Iterable[bytes],
        *,
        max_source_bytes: int,
    ) -> None:
        """Durably publish verified source chunks under their immutable key."""
        if not isinstance(artefact, SourceArtefact):
            raise TypeError("artefact must be a SourceArtefact.")
        if not isinstance(max_source_bytes, int) or isinstance(max_source_bytes, bool):
            raise TypeError("max_source_bytes must be an integer.")
        if max_source_bytes <= 0:
            raise ValueError("max_source_bytes must be greater than zero.")
        try:
            chunk_iterator = iter(chunks)
        except TypeError:
            raise TypeError("chunks must be an iterable of bytes.") from None
        expected_key = build_source_blob_storage_key(artefact.content_hash)
        if artefact.storage_key != expected_key:
            raise StorageConflictError()
        if artefact.byte_count > max_source_bytes:
            raise StorageConflictError()

        if _HAS_SECURE_DIRECTORY_OPEN:
            self._write_with_directory_descriptors(
                artefact,
                chunk_iterator,
                max_source_bytes=max_source_bytes,
            )
            return
        self._write_with_path_checks(
            artefact,
            chunk_iterator,
            max_source_bytes=max_source_bytes,
        )

    @staticmethod
    def _verify_descriptor(
        descriptor: int,
        artefact: SourceArtefact,
        *,
        chunk_size: int,
        retain_content: bool,
    ) -> bytes:
        """Verify an open regular file and optionally retain its content."""
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CorruptStorageError()
        if metadata.st_size != artefact.byte_count:
            raise CorruptStorageError()

        chunks: list[bytes] = []
        digest = hashlib.sha256()
        byte_count = 0
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(chunk_size):
                if retain_content:
                    chunks.append(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
        if (
            byte_count != artefact.byte_count
            or digest.hexdigest() != artefact.content_hash.digest
        ):
            raise CorruptStorageError()
        return b"".join(chunks)

    @staticmethod
    def _write_staging_file(
        descriptor: int,
        artefact: SourceArtefact,
        chunks: Iterator[bytes],
        *,
        max_source_bytes: int,
    ) -> None:
        """Write and flush one staging file while verifying its declared identity."""
        digest = hashlib.sha256()
        byte_count = 0
        try:
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("chunks must contain only bytes.")
                byte_count += len(chunk)
                if byte_count > max_source_bytes:
                    raise StorageConflictError()
                digest.update(chunk)
                if stream.write(chunk) != len(chunk):
                    raise OSError
            stream.flush()
            if (
                byte_count != artefact.byte_count
                or digest.hexdigest() != artefact.content_hash.digest
            ):
                raise StorageConflictError()
            os.fchmod(stream.fileno(), _BLOB_MODE)
            os.fsync(stream.fileno())

    def _write_with_directory_descriptors(
        self,
        artefact: SourceArtefact,
        chunks: Iterator[bytes],
        *,
        max_source_bytes: int,
    ) -> None:
        """Publish through pinned directories without following managed links."""
        directory_descriptors: list[int] = []
        staging_name: str | None = None
        try:
            self._source_dir.mkdir(parents=True, exist_ok=True, mode=_DIRECTORY_MODE)
            root = os.open(self._source_dir, _DIRECTORY_FLAGS)
            directory_descriptors.append(root)
            staging = self._open_or_create_directory(root, ".staging")
            directory_descriptors.append(staging)
            algorithm = self._open_or_create_directory(root, "sha256")
            directory_descriptors.append(algorithm)
            shard = self._open_or_create_directory(
                algorithm,
                artefact.content_hash.digest[:2],
            )
            directory_descriptors.append(shard)

            staging_name, staging_descriptor = self._create_staging_file(staging)
            self._write_staging_file(
                staging_descriptor,
                artefact,
                chunks,
                max_source_bytes=max_source_bytes,
            )
            if self._existing_blob_is_exact(shard, artefact):
                os.unlink(staging_name, dir_fd=staging)
                staging_name = None
                os.fsync(staging)
                return

            os.replace(
                staging_name,
                artefact.content_hash.digest,
                src_dir_fd=staging,
                dst_dir_fd=shard,
            )
            staging_name = None
            os.fsync(shard)
            os.fsync(staging)
        except (StorageConflictError, TypeError, ValueError):
            raise
        except CorruptStorageError:
            raise StorageConflictError() from None
        except OSError:
            raise StorageTransactionError() from None
        finally:
            if staging_name is not None and len(directory_descriptors) >= 2:
                try:
                    os.unlink(staging_name, dir_fd=directory_descriptors[1])
                    os.fsync(directory_descriptors[1])
                except OSError:
                    pass
            for descriptor in reversed(directory_descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    @staticmethod
    def _open_or_create_directory(parent: int, name: str) -> int:
        """Open one private child directory, creating and flushing it if absent."""
        created = False
        try:
            os.mkdir(name, mode=_DIRECTORY_MODE, dir_fd=parent)
            created = True
        except FileExistsError:
            pass
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
        if created:
            os.fsync(parent)
        return descriptor

    @staticmethod
    def _create_staging_file(staging_directory: int) -> tuple[str, int]:
        """Create one unpredictable, writer-owned staging file."""
        for _ in range(10):
            name = f".{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    name,
                    _STAGING_FLAGS,
                    _STAGING_MODE,
                    dir_fd=staging_directory,
                )
            except FileExistsError:
                continue
            return name, descriptor
        raise StorageTransactionError()

    def _existing_blob_is_exact(
        self,
        shard_directory: int,
        artefact: SourceArtefact,
    ) -> bool:
        """Return whether the destination exists and matches immutable content."""
        try:
            descriptor = os.open(
                artefact.content_hash.digest,
                _FILE_FLAGS,
                dir_fd=shard_directory,
            )
        except FileNotFoundError:
            return False
        try:
            self._verify_descriptor(
                descriptor,
                artefact,
                chunk_size=DEFAULT_BLOB_READ_CHUNK_SIZE,
                retain_content=False,
            )
        finally:
            os.close(descriptor)
        return True

    def _write_with_path_checks(
        self,
        artefact: SourceArtefact,
        chunks: Iterator[bytes],
        *,
        max_source_bytes: int,
    ) -> None:
        """Publish with containment checks where directory-relative APIs are absent."""
        staging_path: Path | None = None
        try:
            root = self._prepare_fallback_directory(self._source_dir)
            staging = self._prepare_fallback_directory(root / ".staging")
            algorithm = self._prepare_fallback_directory(root / "sha256")
            shard = self._prepare_fallback_directory(
                algorithm / artefact.content_hash.digest[:2]
            )
            staging_path = staging / f".{secrets.token_hex(16)}.tmp"
            descriptor = os.open(staging_path, _STAGING_FLAGS, _STAGING_MODE)
            self._write_staging_file(
                descriptor,
                artefact,
                chunks,
                max_source_bytes=max_source_bytes,
            )
            destination = shard / artefact.content_hash.digest
            if destination.exists():
                try:
                    self.read_bytes(artefact)
                except CorruptStorageError:
                    raise StorageConflictError() from None
                staging_path.unlink()
                staging_path = None
                self._flush_directory(staging)
                return
            os.replace(staging_path, destination)
            staging_path = None
            self._flush_directory(shard)
            self._flush_directory(staging)
        except (StorageConflictError, TypeError, ValueError):
            raise
        except OSError:
            raise StorageTransactionError() from None
        finally:
            if staging_path is not None:
                try:
                    staging_path.unlink(missing_ok=True)
                    self._flush_directory(staging_path.parent)
                except OSError:
                    pass

    @staticmethod
    def _prepare_fallback_directory(path: Path) -> Path:
        """Create and validate one directory without accepting a redirect."""
        if _is_link_or_junction(path):
            raise OSError
        path.mkdir(parents=True, exist_ok=True, mode=_DIRECTORY_MODE)
        if _is_link_or_junction(path) or not path.is_dir():
            raise OSError
        return path.resolve(strict=True)

    @staticmethod
    def _flush_directory(path: Path) -> None:
        """Flush directory metadata where the host exposes directory handles."""
        descriptor = os.open(path, _DIRECTORY_FLAGS)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

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
