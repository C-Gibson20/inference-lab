"""Tests for managed source-blob addressing and verified reads."""

import hashlib
from pathlib import Path

import pytest
from structlog.testing import capture_logs

import inference_lab.storage._source_blob_store as source_blob_store_module
from inference_lab.provenance import ContentHash, HashAlgorithm
from inference_lab.storage import CorruptStorageError, SourceArtefact
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


def _artefact(
    *,
    content_hash: ContentHash = SOURCE_HASH,
    byte_count: int = len(SOURCE_BYTES),
    storage_key: str | None = None,
) -> SourceArtefact:
    return SourceArtefact(
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        content_hash=content_hash,
        byte_count=byte_count,
        media_type="text/plain",
        storage_key=(
            build_source_blob_storage_key(content_hash)
            if storage_key is None
            else storage_key
        ),
    )


def _install_blob(
    source_dir: Path,
    artefact: SourceArtefact,
    content: bytes = SOURCE_BYTES,
) -> Path:
    blob_path = source_dir / artefact.storage_key
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(content)
    return blob_path


def test_storage_key_is_deterministic_and_content_addressed() -> None:
    key = build_source_blob_storage_key(SOURCE_HASH)

    assert key == f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}"
    assert not key.startswith("/")
    assert DOCUMENT_ID.as_str() not in key


def test_storage_key_rejects_wrong_runtime_type() -> None:
    with pytest.raises(TypeError, match="ContentHash"):
        build_source_blob_storage_key(object())  # type: ignore[arg-type]


def test_store_construction_is_side_effect_free(tmp_path: Path) -> None:
    source_dir = tmp_path / "missing"

    ManagedSourceBlobStore(source_dir)

    assert not source_dir.exists()


@pytest.mark.parametrize("source_dir", ["source", Path("https:/example.test/source")])
def test_store_rejects_invalid_root_type_or_shape(source_dir: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ManagedSourceBlobStore(source_dir)  # type: ignore[arg-type]


def test_verified_read_returns_exact_bytes_across_restart(tmp_path: Path) -> None:
    artefact = _artefact()
    _install_blob(tmp_path, artefact)

    first = ManagedSourceBlobStore(tmp_path).read_bytes(artefact, chunk_size=3)
    second = ManagedSourceBlobStore(tmp_path).read_bytes(artefact, chunk_size=5)

    assert first == SOURCE_BYTES
    assert second == SOURCE_BYTES


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_read_rejects_invalid_chunk_size(
    tmp_path: Path,
    chunk_size: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ManagedSourceBlobStore(tmp_path).read_bytes(
            _artefact(),
            chunk_size=chunk_size,  # type: ignore[arg-type]
        )


def test_read_rejects_wrong_artefact_runtime_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="SourceArtefact"):
        ManagedSourceBlobStore(tmp_path).read_bytes(object())  # type: ignore[arg-type]


def test_read_rejects_non_policy_storage_key(tmp_path: Path) -> None:
    artefact = _artefact(storage_key=f"sha256/other/{SOURCE_HASH.digest}")

    with pytest.raises(CorruptStorageError):
        ManagedSourceBlobStore(tmp_path).read_bytes(artefact)


def test_read_rejects_missing_blob_without_exposing_path(tmp_path: Path) -> None:
    private_marker = "private-source-root"
    source_dir = tmp_path / private_marker

    with pytest.raises(CorruptStorageError) as captured:
        ManagedSourceBlobStore(source_dir).read_bytes(_artefact())

    assert private_marker not in str(captured.value)


def test_read_rejects_truncated_blob(tmp_path: Path) -> None:
    artefact = _artefact()
    _install_blob(tmp_path, artefact, SOURCE_BYTES[:-1])

    with pytest.raises(CorruptStorageError):
        ManagedSourceBlobStore(tmp_path).read_bytes(artefact)


def test_read_rejects_digest_mismatch_with_matching_size(tmp_path: Path) -> None:
    artefact = _artefact()
    _install_blob(tmp_path, artefact, b"x" * len(SOURCE_BYTES))

    with pytest.raises(CorruptStorageError):
        ManagedSourceBlobStore(tmp_path).read_bytes(artefact)


def test_read_rejects_directory_at_blob_path(tmp_path: Path) -> None:
    artefact = _artefact()
    (tmp_path / artefact.storage_key).mkdir(parents=True)

    with pytest.raises(CorruptStorageError):
        ManagedSourceBlobStore(tmp_path).read_bytes(artefact)


def test_read_does_not_follow_managed_directory_symlink(tmp_path: Path) -> None:
    artefact = _artefact()
    outside = tmp_path / "outside"
    _install_blob(outside, artefact)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "sha256").symlink_to(outside / "sha256", target_is_directory=True)

    with pytest.raises(CorruptStorageError):
        ManagedSourceBlobStore(source_dir).read_bytes(artefact)


def test_read_does_not_follow_blob_symlink(tmp_path: Path) -> None:
    artefact = _artefact()
    outside_blob = tmp_path / "outside" / "blob"
    outside_blob.parent.mkdir()
    outside_blob.write_bytes(SOURCE_BYTES)
    blob_path = tmp_path / "source" / artefact.storage_key
    blob_path.parent.mkdir(parents=True)
    blob_path.symlink_to(outside_blob)

    with pytest.raises(CorruptStorageError):
        ManagedSourceBlobStore(tmp_path / "source").read_bytes(artefact)


def test_read_does_not_follow_source_root_symlink(tmp_path: Path) -> None:
    artefact = _artefact()
    outside = tmp_path / "outside"
    _install_blob(outside, artefact)
    source_dir = tmp_path / "source"
    source_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(CorruptStorageError):
        ManagedSourceBlobStore(source_dir).read_bytes(artefact)


def test_portable_path_fallback_reads_without_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artefact = _artefact()
    _install_blob(tmp_path, artefact)
    monkeypatch.setattr(
        source_blob_store_module,
        "_HAS_SECURE_DIRECTORY_OPEN",
        False,
    )

    assert ManagedSourceBlobStore(tmp_path).read_bytes(artefact) == SOURCE_BYTES


def test_portable_path_fallback_rejects_managed_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artefact = _artefact()
    outside = tmp_path / "outside"
    _install_blob(outside, artefact)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "sha256").symlink_to(outside / "sha256", target_is_directory=True)
    monkeypatch.setattr(
        source_blob_store_module,
        "_HAS_SECURE_DIRECTORY_OPEN",
        False,
    )

    with pytest.raises(CorruptStorageError):
        ManagedSourceBlobStore(source_dir).read_bytes(artefact)


def test_verified_read_emits_no_paths_or_logs(tmp_path: Path) -> None:
    artefact = _artefact()
    _install_blob(tmp_path, artefact)

    with capture_logs() as logs:
        content = ManagedSourceBlobStore(tmp_path).read_bytes(artefact)

    assert content == SOURCE_BYTES
    assert logs == []


def test_read_accepts_empty_verified_blob(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"").hexdigest()
    artefact = _artefact(
        content_hash=ContentHash(HashAlgorithm.SHA256, digest),
        byte_count=0,
    )
    _install_blob(tmp_path, artefact, b"")

    assert ManagedSourceBlobStore(tmp_path).read_bytes(artefact) == b""
