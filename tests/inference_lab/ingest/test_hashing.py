"""Tests for stable, bounded hashing of local source candidates."""

import hashlib
from pathlib import Path
from typing import BinaryIO

import pytest

from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    LocalStableReadFileSystem,
    SourceClassification,
    SourceSupport,
    StableReadFileSystem,
    TraversalCandidate,
    VerifiedSourceSnapshot,
    hash_source_candidate,
    inventory_local_sources,
)
from inference_lab.provenance import CorpusId, HashAlgorithm

from .fixtures import fixture_settings  # noqa: TID252


class _RecordingFileSystem(LocalStableReadFileSystem):
    """Record requested read bounds without retaining source content."""

    def __init__(self) -> None:
        self.read_sizes: list[int] = []

    def read(self, source: BinaryIO, byte_count: int) -> bytes:
        self.read_sizes.append(byte_count)
        return super().read(source, byte_count)


def _candidate(path: Path) -> TraversalCandidate:
    request = DiscoveryRequest(
        corpus_id=CorpusId.generate(),
        inputs=(DiscoveryInput(path, DiscoveryInputKind.FILE),),
    )
    inventory = inventory_local_sources(request, excluded_paths=())
    return inventory.candidates[0]


def _classification() -> SourceClassification:
    return SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name="test-classifier",
        classifier_version="1.0.0",
        parser_capability="text-native",
        encoding="utf-8",
    )


def test_regular_file_is_streamed_into_a_verified_sha256_snapshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "private-source.txt"
    content = (b"exact original bytes\x00" * 60_000) + b"end"
    source.write_bytes(content)
    candidate = _candidate(source)
    filesystem = _RecordingFileSystem()

    result = hash_source_candidate(
        candidate,
        _classification(),
        settings=fixture_settings(max_source_bytes=len(content)),
        filesystem=filesystem,
    )

    assert isinstance(result, VerifiedSourceSnapshot)
    assert result.byte_count == len(content)
    assert result.content_hash.algorithm is HashAlgorithm.SHA256
    assert result.content_hash.digest == hashlib.sha256(content).hexdigest()
    assert result.media_type == "text/plain"
    assert result.encoding == "utf-8"
    assert len(filesystem.read_sizes) >= 3
    assert set(filesystem.read_sizes) == {1024 * 1024}
    assert result.file_evidence.pre_open == result.file_evidence.opened
    assert result.file_evidence.opened == result.file_evidence.opened_after_read
    assert (
        result.file_evidence.opened_after_read == result.file_evidence.path_after_read
    )


def test_empty_file_has_the_standard_empty_sha256_digest(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.touch()

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(),
    )

    assert isinstance(result, VerifiedSourceSnapshot)
    assert result.byte_count == 0
    assert result.content_hash.digest == hashlib.sha256(b"").hexdigest()


def test_size_limit_accepts_boundary_and_rejects_oversized_source(
    tmp_path: Path,
) -> None:
    boundary = tmp_path / "boundary.txt"
    boundary.write_bytes(b"1234")
    oversized = tmp_path / "oversized.txt"
    oversized.write_bytes(b"12345")
    settings = fixture_settings(max_source_bytes=4)

    accepted = hash_source_candidate(
        _candidate(boundary),
        _classification(),
        settings=settings,
    )
    rejected = hash_source_candidate(
        _candidate(oversized),
        _classification(),
        settings=settings,
    )

    assert isinstance(accepted, VerifiedSourceSnapshot)
    assert accepted.byte_count == 4
    assert isinstance(rejected, DiscoveryFailure)
    assert rejected.category is DiscoveryFailureCategory.SOURCE_TOO_LARGE


def test_candidate_replaced_by_symlink_is_not_opened(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("original", encoding="utf-8")
    candidate = _candidate(source)
    target = tmp_path / "target.txt"
    target.write_text("replacement", encoding="utf-8")
    source.unlink()
    try:
        source.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {type(error).__name__}")

    result = hash_source_candidate(
        candidate,
        _classification(),
        settings=fixture_settings(),
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ


def test_hashing_boundary_requires_typed_supported_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    candidate = _candidate(source)
    unsupported = SourceClassification(
        media_type="application/pdf",
        support=SourceSupport.RECOGNISED_UNSUPPORTED,
        classifier_name="test-classifier",
        classifier_version="1.0.0",
    )

    assert isinstance(LocalStableReadFileSystem(), StableReadFileSystem)
    with pytest.raises(TypeError, match="candidate must be a TraversalCandidate"):
        hash_source_candidate(
            object(),  # pyright: ignore[reportArgumentType]
            _classification(),
            settings=fixture_settings(),
        )
    with pytest.raises(ValueError, match="must describe a supported source"):
        hash_source_candidate(
            candidate,
            unsupported,
            settings=fixture_settings(),
        )
    with pytest.raises(TypeError, match="filesystem must implement"):
        hash_source_candidate(
            candidate,
            _classification(),
            settings=fixture_settings(),
            filesystem=object(),  # pyright: ignore[reportArgumentType]
        )


def test_hash_failure_representation_hides_source_path(tmp_path: Path) -> None:
    marker = "confidential-hash-source-8192.txt"
    source = tmp_path / marker
    source.write_bytes(b"private payload")

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(max_source_bytes=1),
    )

    assert isinstance(result, DiscoveryFailure)
    assert marker not in repr(result)
    assert "private payload" not in repr(result)
