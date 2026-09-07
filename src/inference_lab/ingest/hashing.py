"""Stable, bounded hashing of local source candidates."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Final, Protocol, runtime_checkable

from inference_lab._validation import is_strict_integer
from inference_lab.ingest.classification import ClassifiedSourceCandidate
from inference_lab.ingest.discovery import (
    FileIdentityHint,
    FileStatEvidence,
    SourceSupport,
    StableFileEvidence,
    VerifiedSourceSnapshot,
)
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
)
from inference_lab.ingest.discovery_logging import (
    emit_discovery_event,
    failure_log_fields,
)
from inference_lab.ingest.traversal import (
    LocalTraversalFileSystem,
    TraversalCandidate,
    TraversalPathMetadata,
)
from inference_lab.provenance import ContentHash, HashAlgorithm

_HASH_CHUNK_SIZE_BYTES: Final = 1024 * 1024

Clock = Callable[[], int]


@runtime_checkable
class StableReadFileSystem(Protocol):
    """Provide bounded operations for hashing one open local file."""

    def lstat(self, path: Path) -> TraversalPathMetadata:
        """Return metadata without following the final path component."""
        ...

    def open_binary(self, path: Path) -> BinaryIO:
        """Open a source for binary reading with strongest no-follow semantics."""
        ...

    def fstat(self, source: BinaryIO) -> TraversalPathMetadata:
        """Return metadata for the open source handle."""
        ...

    def read(self, source: BinaryIO, byte_count: int) -> bytes:
        """Read at most one bounded chunk from the open source."""
        ...


def _metadata_from_stat_result(stat_result: os.stat_result) -> TraversalPathMetadata:
    """Convert host stat fields into the bounded traversal metadata contract."""
    return TraversalPathMetadata(
        mode=stat_result.st_mode,
        byte_count=stat_result.st_size,
        modified_time_ns=stat_result.st_mtime_ns,
        device_id=stat_result.st_dev,
        file_id=stat_result.st_ino,
    )


class LocalStableReadFileSystem(LocalTraversalFileSystem):
    """Adapt local no-follow file operations for stable source hashing."""

    def open_binary(self, path: Path) -> BinaryIO:
        """Open a binary descriptor with every available safety flag."""
        flags = os.O_RDONLY
        for flag_name in (
            "O_BINARY",
            "O_CLOEXEC",
            "O_NOCTTY",
            "O_NOFOLLOW",
            "O_NONBLOCK",
        ):
            flags |= getattr(os, flag_name, 0)
        descriptor = os.open(path, flags)
        try:
            return os.fdopen(descriptor, "rb", closefd=True)
        except Exception:
            os.close(descriptor)
            raise

    def fstat(self, source: BinaryIO) -> TraversalPathMetadata:
        """Return bounded metadata for the open source descriptor."""
        return _metadata_from_stat_result(os.fstat(source.fileno()))

    def read(self, source: BinaryIO, byte_count: int) -> bytes:
        """Read one bounded binary chunk from the open source."""
        return source.read(byte_count)


_LOCAL_STABLE_READ_FILESYSTEM = LocalStableReadFileSystem()


def _failure(
    source: ClassifiedSourceCandidate,
    category: DiscoveryFailureCategory,
) -> DiscoveryFailure:
    """Build a payload-free hashing failure for one traversal candidate."""
    return DiscoveryFailure.for_category(
        ordinal=source.candidate.ordinal,
        category=category,
        locator=source.candidate.locator,
        file_kind=DiscoveryFileKind.REGULAR_FILE,
        media_type=source.classification.media_type,
    )


def _file_evidence(metadata: TraversalPathMetadata) -> FileStatEvidence:
    """Convert bounded host metadata into stable-read evidence."""
    return FileStatEvidence(
        byte_count=metadata.byte_count,
        modified_time_ns=metadata.modified_time_ns,
        identity_hint=FileIdentityHint(
            device_id=metadata.device_id,
            file_id=metadata.file_id,
        ),
    )


def _is_regular(metadata: TraversalPathMetadata) -> bool:
    """Return whether metadata identifies a regular file."""
    return stat.S_ISREG(metadata.mode)


def _validated_metadata(metadata: object) -> TraversalPathMetadata:
    """Reject malformed adapter metadata without exposing its representation."""
    if not isinstance(metadata, TraversalPathMetadata):
        raise TypeError("filesystem must return TraversalPathMetadata.")
    return metadata


def _path_changed_after_open_failure(
    candidate: TraversalCandidate,
    pre_open: TraversalPathMetadata,
    filesystem: StableReadFileSystem,
) -> bool:
    """Check whether a failed open coincided with a safely observable path swap."""
    try:
        current = _validated_metadata(filesystem.lstat(candidate.locator.path))
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return not _is_regular(current) or _file_evidence(current) != _file_evidence(
        pre_open
    )


def _hash_open_source(
    binary_source: BinaryIO,
    *,
    source: ClassifiedSourceCandidate,
    max_source_bytes: int,
    pre_open: TraversalPathMetadata,
    filesystem: StableReadFileSystem,
) -> VerifiedSourceSnapshot | DiscoveryFailure:
    """Hash one open source and validate its complete observation boundary."""
    opened = _validated_metadata(filesystem.fstat(binary_source))
    if not _is_regular(opened):
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        )
    if opened.byte_count > max_source_bytes:
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_TOO_LARGE,
        )
    if _file_evidence(opened) != _file_evidence(pre_open):
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        )

    digest = hashlib.sha256()
    byte_count = 0
    while True:
        chunk = filesystem.read(binary_source, _HASH_CHUNK_SIZE_BYTES)
        if not isinstance(chunk, bytes):
            raise TypeError("filesystem read must return bytes.")
        if not chunk:
            break
        byte_count += len(chunk)
        if byte_count > max_source_bytes:
            return _failure(
                source,
                DiscoveryFailureCategory.SOURCE_TOO_LARGE,
            )
        digest.update(chunk)

    opened_after_read = _validated_metadata(filesystem.fstat(binary_source))
    try:
        path_after_read = _validated_metadata(
            filesystem.lstat(source.candidate.locator.path)
        )
    except FileNotFoundError:
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        )
    if not all(
        _is_regular(metadata)
        for metadata in (pre_open, opened, opened_after_read, path_after_read)
    ):
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        )

    observations = tuple(
        _file_evidence(metadata)
        for metadata in (pre_open, opened, opened_after_read, path_after_read)
    )
    if any(observation != observations[0] for observation in observations[1:]):
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        )
    if byte_count != opened_after_read.byte_count:
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        )

    stable_evidence = StableFileEvidence(
        pre_open=observations[0],
        opened=observations[1],
        opened_after_read=observations[2],
        path_after_read=observations[3],
    )
    return VerifiedSourceSnapshot(
        locator=source.candidate.locator,
        media_type=source.classification.media_type,
        encoding=source.classification.encoding,
        byte_count=byte_count,
        content_hash=ContentHash(HashAlgorithm.SHA256, digest.hexdigest()),
        file_evidence=stable_evidence,
    )


def _hash_source_candidate(
    source: ClassifiedSourceCandidate,
    *,
    max_source_bytes: int,
    filesystem: StableReadFileSystem,
) -> VerifiedSourceSnapshot | DiscoveryFailure:
    """Run the validated stable-read algorithm for one source."""
    candidate = source.candidate
    try:
        pre_open = _validated_metadata(filesystem.lstat(candidate.locator.path))
    except FileNotFoundError:
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        )
    except OSError:
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_READ_FAILED,
        )
    if not _is_regular(pre_open):
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        )
    if pre_open.byte_count > max_source_bytes:
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_TOO_LARGE,
        )

    try:
        binary_source = filesystem.open_binary(candidate.locator.path)
    except FileNotFoundError:
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        )
    except OSError:
        if _path_changed_after_open_failure(
            candidate,
            pre_open,
            filesystem,
        ):
            return _failure(
                source,
                DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
            )
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_READ_FAILED,
        )

    try:
        with binary_source:
            return _hash_open_source(
                binary_source,
                source=source,
                max_source_bytes=max_source_bytes,
                pre_open=pre_open,
                filesystem=filesystem,
            )
    except OSError:
        return _failure(
            source,
            DiscoveryFailureCategory.SOURCE_READ_FAILED,
        )


def _read_clock(clock: Clock) -> int:
    value = clock()
    if not is_strict_integer(value):
        raise TypeError("clock must return an integer nanosecond value.")
    return value


def hash_source_candidate(
    source: ClassifiedSourceCandidate,
    *,
    max_source_bytes: int,
    filesystem: StableReadFileSystem | None = None,
    clock: Clock = time.monotonic_ns,
) -> VerifiedSourceSnapshot | DiscoveryFailure:
    """Hash one supported regular-file candidate as a stable byte snapshot."""
    if not isinstance(source, ClassifiedSourceCandidate):
        raise TypeError("source must be a ClassifiedSourceCandidate.")
    if source.classification.support is not SourceSupport.SUPPORTED:
        raise ValueError("classification must describe a supported source.")
    if not is_strict_integer(max_source_bytes):
        raise TypeError("max_source_bytes must be an integer.")
    if max_source_bytes <= 0:
        raise ValueError("max_source_bytes must be positive.")
    selected_filesystem = (
        _LOCAL_STABLE_READ_FILESYSTEM if filesystem is None else filesystem
    )
    if not isinstance(selected_filesystem, StableReadFileSystem):
        raise TypeError("filesystem must implement StableReadFileSystem.")
    if not callable(clock):
        raise TypeError("clock must be callable.")

    started_ns = _read_clock(clock)
    result = _hash_source_candidate(
        source,
        max_source_bytes=max_source_bytes,
        filesystem=selected_filesystem,
    )
    finished_ns = _read_clock(clock)
    if finished_ns < started_ns:
        raise ValueError("clock must not move backwards during source hashing.")
    duration_ms = (finished_ns - started_ns) / 1_000_000

    if isinstance(result, VerifiedSourceSnapshot):
        emit_discovery_event(
            __name__,
            "debug",
            "source_hashed",
            source_ordinal=source.candidate.ordinal,
            file_kind=DiscoveryFileKind.REGULAR_FILE.value,
            media_type=result.media_type,
            byte_count=result.byte_count,
            duration_ms=duration_ms,
        )
    else:
        emit_discovery_event(
            __name__,
            "error",
            "source_failed",
            duration_ms=duration_ms,
            **failure_log_fields(result),
        )
    return result
