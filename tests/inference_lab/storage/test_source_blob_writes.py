"""Tests for bounded, durable managed source-blob writes."""

import hashlib
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from structlog.testing import capture_logs

import inference_lab.storage._source_blob_store as source_blob_store_module
from inference_lab.provenance import (
    ContentHash,
    DocumentId,
    DocumentVersionId,
    HashAlgorithm,
)
from inference_lab.storage import (
    SourceArtefact,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage._source_blob_store import (
    ManagedSourceBlobStore,
    build_source_blob_storage_key,
)
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    SOURCE_BYTES,
    SOURCE_HASH,
)


def _content_hash(content: bytes) -> ContentHash:
    return ContentHash(HashAlgorithm.SHA256, hashlib.sha256(content).hexdigest())


def _artefact(
    *,
    content_hash: ContentHash = SOURCE_HASH,
    byte_count: int = len(SOURCE_BYTES),
    storage_key: str | None = None,
    document_id: DocumentId = DOCUMENT_ID,
    document_version_id: DocumentVersionId = DOCUMENT_VERSION_ID,
) -> SourceArtefact:
    return SourceArtefact(
        document_id=document_id,
        document_version_id=document_version_id,
        content_hash=content_hash,
        byte_count=byte_count,
        media_type="text/plain",
        storage_key=(
            build_source_blob_storage_key(content_hash)
            if storage_key is None
            else storage_key
        ),
    )


def _blob_path(source_dir: Path, artefact: SourceArtefact) -> Path:
    return source_dir / artefact.storage_key


def _staging_files(source_dir: Path) -> tuple[Path, ...]:
    staging = source_dir / ".staging"
    return tuple(staging.iterdir()) if staging.exists() else ()


def test_write_streams_and_publishes_verified_immutable_blob(tmp_path: Path) -> None:
    artefact = _artefact()
    store = ManagedSourceBlobStore(tmp_path)

    store.write_chunks(
        artefact,
        (SOURCE_BYTES[index : index + 3] for index in range(0, len(SOURCE_BYTES), 3)),
        max_source_bytes=len(SOURCE_BYTES),
    )

    blob_path = _blob_path(tmp_path, artefact)
    assert store.read_bytes(artefact) == SOURCE_BYTES
    assert stat.S_IMODE(blob_path.stat().st_mode) & stat.S_IWUSR == 0
    assert _staging_files(tmp_path) == ()


def test_destination_is_not_visible_until_complete_payload_is_staged(
    tmp_path: Path,
) -> None:
    artefact = _artefact()
    destination = _blob_path(tmp_path, artefact)

    def chunks() -> Iterator[bytes]:
        assert not destination.exists()
        yield SOURCE_BYTES[:4]
        assert not destination.exists()
        yield SOURCE_BYTES[4:]

    ManagedSourceBlobStore(tmp_path).write_chunks(
        artefact,
        chunks(),
        max_source_bytes=len(SOURCE_BYTES),
    )

    assert destination.read_bytes() == SOURCE_BYTES


def test_write_flushes_blob_and_changed_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flushed_modes: list[int] = []
    real_fsync = os.fsync

    def record_flush(descriptor: int) -> None:
        flushed_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(source_blob_store_module.os, "fsync", record_flush)

    ManagedSourceBlobStore(tmp_path).write_chunks(
        _artefact(),
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )

    assert any(stat.S_ISREG(mode) for mode in flushed_modes)
    assert any(stat.S_ISDIR(mode) for mode in flushed_modes)


def test_empty_blob_can_be_written_and_read(tmp_path: Path) -> None:
    content_hash = _content_hash(b"")
    artefact = _artefact(content_hash=content_hash, byte_count=0)
    store = ManagedSourceBlobStore(tmp_path)

    store.write_chunks(artefact, (), max_source_bytes=1)

    assert store.read_bytes(artefact) == b""


@pytest.mark.parametrize("max_source_bytes", [0, -1, True, 1.5])
def test_write_rejects_invalid_size_limit(
    tmp_path: Path,
    max_source_bytes: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ManagedSourceBlobStore(tmp_path).write_chunks(
            _artefact(),
            (SOURCE_BYTES,),
            max_source_bytes=max_source_bytes,  # type: ignore[arg-type]
        )


def test_write_rejects_wrong_artefact_or_chunks_runtime_type(tmp_path: Path) -> None:
    store = ManagedSourceBlobStore(tmp_path)

    with pytest.raises(TypeError, match="SourceArtefact"):
        store.write_chunks(
            object(),  # type: ignore[arg-type]
            (),
            max_source_bytes=1,
        )
    with pytest.raises(TypeError, match="iterable"):
        store.write_chunks(
            _artefact(),
            None,  # type: ignore[arg-type]
            max_source_bytes=len(SOURCE_BYTES),
        )


def test_write_rejects_non_byte_chunks_and_removes_staging(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="only bytes"):
        ManagedSourceBlobStore(tmp_path).write_chunks(
            _artefact(),
            (SOURCE_BYTES, "private text"),  # type: ignore[arg-type]
            max_source_bytes=100,
        )

    assert not _blob_path(tmp_path, _artefact()).exists()
    assert _staging_files(tmp_path) == ()


def test_declared_size_over_limit_is_rejected_before_consuming_source(
    tmp_path: Path,
) -> None:
    consumed = False

    def chunks() -> Iterator[bytes]:
        nonlocal consumed
        consumed = True
        yield SOURCE_BYTES

    with pytest.raises(StorageConflictError):
        ManagedSourceBlobStore(tmp_path).write_chunks(
            _artefact(),
            chunks(),
            max_source_bytes=len(SOURCE_BYTES) - 1,
        )

    assert not consumed
    assert tuple(tmp_path.iterdir()) == ()


def test_stream_exceeding_limit_is_rejected_and_cleaned(tmp_path: Path) -> None:
    declared = b"safe"
    artefact = _artefact(content_hash=_content_hash(declared), byte_count=len(declared))

    with pytest.raises(StorageConflictError):
        ManagedSourceBlobStore(tmp_path).write_chunks(
            artefact,
            (b"unsafe",),
            max_source_bytes=5,
        )

    assert not _blob_path(tmp_path, artefact).exists()
    assert _staging_files(tmp_path) == ()


@pytest.mark.parametrize(
    "chunks",
    [
        (SOURCE_BYTES[:-1],),
        (b"x" * len(SOURCE_BYTES),),
    ],
)
def test_write_rejects_byte_count_or_digest_mismatch(
    tmp_path: Path,
    chunks: tuple[bytes, ...],
) -> None:
    artefact = _artefact()

    with pytest.raises(StorageConflictError):
        ManagedSourceBlobStore(tmp_path).write_chunks(
            artefact,
            chunks,
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert not _blob_path(tmp_path, artefact).exists()
    assert _staging_files(tmp_path) == ()


def test_write_rejects_non_policy_storage_key(tmp_path: Path) -> None:
    artefact = _artefact(storage_key=f"sha256/other/{SOURCE_HASH.digest}")

    with pytest.raises(StorageConflictError):
        ManagedSourceBlobStore(tmp_path).write_chunks(
            artefact,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert tuple(tmp_path.iterdir()) == ()


def test_exact_existing_blob_is_a_no_op(tmp_path: Path) -> None:
    artefact = _artefact()
    store = ManagedSourceBlobStore(tmp_path)
    store.write_chunks(
        artefact,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )
    blob_path = _blob_path(tmp_path, artefact)
    original_identity = (blob_path.stat().st_ino, blob_path.stat().st_mtime_ns)

    store.write_chunks(
        artefact,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )

    assert (blob_path.stat().st_ino, blob_path.stat().st_mtime_ns) == original_identity
    assert _staging_files(tmp_path) == ()


def test_existing_blob_does_not_hide_different_incoming_payload(tmp_path: Path) -> None:
    artefact = _artefact()
    store = ManagedSourceBlobStore(tmp_path)
    store.write_chunks(
        artefact,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )

    with pytest.raises(StorageConflictError):
        store.write_chunks(
            artefact,
            (b"x" * len(SOURCE_BYTES),),
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert store.read_bytes(artefact) == SOURCE_BYTES
    assert _staging_files(tmp_path) == ()


def test_corrupt_existing_blob_is_a_conflict_and_is_not_replaced(
    tmp_path: Path,
) -> None:
    artefact = _artefact()
    blob_path = _blob_path(tmp_path, artefact)
    blob_path.parent.mkdir(parents=True)
    corrupt_content = b"x" * len(SOURCE_BYTES)
    blob_path.write_bytes(corrupt_content)

    with pytest.raises(StorageConflictError):
        ManagedSourceBlobStore(tmp_path).write_chunks(
            artefact,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert blob_path.read_bytes() == corrupt_content
    assert _staging_files(tmp_path) == ()


def test_identical_content_is_reused_across_document_versions(tmp_path: Path) -> None:
    first = _artefact()
    second = _artefact(
        document_id=DocumentId.from_str("10000000-0000-4000-8000-000000000002"),
        document_version_id=DocumentVersionId.from_str(
            "20000000-0000-4000-8000-000000000002"
        ),
    )
    store = ManagedSourceBlobStore(tmp_path)

    store.write_chunks(first, (SOURCE_BYTES,), max_source_bytes=len(SOURCE_BYTES))
    first_inode = _blob_path(tmp_path, first).stat().st_ino
    store.write_chunks(second, (SOURCE_BYTES,), max_source_bytes=len(SOURCE_BYTES))

    assert first.storage_key == second.storage_key
    assert _blob_path(tmp_path, second).stat().st_ino == first_inode


def test_source_interruption_preserves_error_and_removes_staging(
    tmp_path: Path,
) -> None:
    class SourceInterrupted(RuntimeError):
        pass

    def interrupted_chunks() -> Iterator[bytes]:
        yield SOURCE_BYTES[:3]
        raise SourceInterrupted

    with pytest.raises(SourceInterrupted):
        ManagedSourceBlobStore(tmp_path).write_chunks(
            _artefact(),
            interrupted_chunks(),
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert not _blob_path(tmp_path, _artefact()).exists()
    assert _staging_files(tmp_path) == ()


def test_atomic_promotion_failure_is_private_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_marker = "private-source-root"
    source_dir = tmp_path / private_marker

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("private filesystem detail")

    monkeypatch.setattr(source_blob_store_module.os, "replace", fail_replace)

    with pytest.raises(StorageTransactionError) as captured:
        ManagedSourceBlobStore(source_dir).write_chunks(
            _artefact(),
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert private_marker not in str(captured.value)
    assert not _blob_path(source_dir, _artefact()).exists()
    assert _staging_files(source_dir) == ()


def test_flush_failure_does_not_publish_partial_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_flush(descriptor: int) -> None:
        raise OSError

    monkeypatch.setattr(source_blob_store_module.os, "fsync", fail_flush)

    with pytest.raises(StorageTransactionError):
        ManagedSourceBlobStore(tmp_path).write_chunks(
            _artefact(),
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert not _blob_path(tmp_path, _artefact()).exists()
    assert _staging_files(tmp_path) == ()


def test_write_rejects_managed_directory_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "sha256").symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageTransactionError):
        ManagedSourceBlobStore(source_dir).write_chunks(
            _artefact(),
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert tuple(outside.iterdir()) == ()
    assert _staging_files(source_dir) == ()


def test_write_rejects_source_root_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    source_dir = tmp_path / "source"
    source_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageTransactionError):
        ManagedSourceBlobStore(source_dir).write_chunks(
            _artefact(),
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert tuple(outside.iterdir()) == ()


def test_portable_fallback_writes_and_reuses_exact_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artefact = _artefact()
    monkeypatch.setattr(
        source_blob_store_module,
        "_HAS_SECURE_DIRECTORY_OPEN",
        False,
    )
    store = ManagedSourceBlobStore(tmp_path)

    store.write_chunks(
        artefact,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )
    store.write_chunks(
        artefact,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )

    assert store.read_bytes(artefact) == SOURCE_BYTES
    assert _staging_files(tmp_path) == ()


def test_write_emits_no_paths_or_logs(tmp_path: Path) -> None:
    artefact = _artefact()

    with capture_logs() as logs:
        ManagedSourceBlobStore(tmp_path).write_chunks(
            artefact,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
        )

    assert logs == []
    assert (
        os.fspath(tmp_path).encode() not in _blob_path(tmp_path, artefact).read_bytes()
    )
