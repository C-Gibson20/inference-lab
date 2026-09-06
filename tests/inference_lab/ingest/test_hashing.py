"""Tests for stable, bounded hashing of local source candidates."""

import hashlib
import stat
from collections.abc import Callable
from dataclasses import replace
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
    TraversalPathMetadata,
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


class _FaultInjectingFileSystem(_RecordingFileSystem):
    """Inject deterministic open, read, and stat boundary failures."""

    def __init__(self) -> None:
        super().__init__()
        self.before_open: Callable[[], None] | None = None
        self.open_error = False
        self.open_missing = False
        self.read_errors: set[int] = set()
        self.injected_chunks: dict[int, bytes] = {}
        self.missing_lstats: set[int] = set()
        self.lstat_errors: set[int] = set()
        self.fstat_errors: set[int] = set()
        self.lstat_transforms: dict[
            int, Callable[[TraversalPathMetadata], TraversalPathMetadata]
        ] = {}
        self.fstat_transforms: dict[
            int, Callable[[TraversalPathMetadata], TraversalPathMetadata]
        ] = {}
        self.lstat_count = 0
        self.fstat_count = 0
        self.read_count = 0
        self.opened_source: BinaryIO | None = None

    def lstat(self, path: Path) -> TraversalPathMetadata:
        self.lstat_count += 1
        if self.lstat_count in self.missing_lstats:
            raise FileNotFoundError("injected private path")
        if self.lstat_count in self.lstat_errors:
            raise PermissionError("injected private path")
        metadata = super().lstat(path)
        transform = self.lstat_transforms.get(self.lstat_count)
        return metadata if transform is None else transform(metadata)

    def open_binary(self, path: Path) -> BinaryIO:
        if self.before_open is not None:
            self.before_open()
        if self.open_missing:
            raise FileNotFoundError("injected private path")
        if self.open_error:
            raise PermissionError("injected private path")
        self.opened_source = super().open_binary(path)
        return self.opened_source

    def fstat(self, source: BinaryIO) -> TraversalPathMetadata:
        self.fstat_count += 1
        if self.fstat_count in self.fstat_errors:
            raise OSError("injected private path")
        metadata = super().fstat(source)
        transform = self.fstat_transforms.get(self.fstat_count)
        return metadata if transform is None else transform(metadata)

    def read(self, source: BinaryIO, byte_count: int) -> bytes:
        self.read_count += 1
        if self.read_count in self.read_errors:
            raise OSError("injected private payload")
        chunk = super().read(source, byte_count)
        return self.injected_chunks.get(self.read_count, chunk)


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


@pytest.mark.parametrize("swap", ["identity", "type"])
def test_opened_handle_swap_is_rejected_before_read(
    tmp_path: Path,
    swap: str,
) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"original")
    filesystem = _FaultInjectingFileSystem()
    if swap == "identity":
        filesystem.fstat_transforms[1] = lambda metadata: replace(
            metadata, file_id=metadata.file_id + 1
        )
    else:
        filesystem.fstat_transforms[1] = lambda metadata: replace(
            metadata, mode=stat.S_IFDIR
        )

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(),
        filesystem=filesystem,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ
    assert filesystem.read_count == 0
    assert filesystem.opened_source is not None
    assert filesystem.opened_source.closed


def test_symlink_swap_at_open_is_rejected_without_reading_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"original")
    candidate = _candidate(source)
    target = tmp_path / "private-target.txt"
    target.write_bytes(b"replacement")
    filesystem = _FaultInjectingFileSystem()

    def replace_with_symlink() -> None:
        source.unlink()
        try:
            source.symlink_to(target)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {type(error).__name__}")

    filesystem.before_open = replace_with_symlink

    result = hash_source_candidate(
        candidate,
        _classification(),
        settings=fixture_settings(),
        filesystem=filesystem,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ
    assert filesystem.read_count == 0
    assert filesystem.opened_source is None or filesystem.opened_source.closed


@pytest.mark.parametrize(
    "change",
    ["mutation", "truncation", "replacement", "type_swap"],
)
def test_changed_post_read_evidence_discards_digest(
    tmp_path: Path,
    change: str,
) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"stable bytes")
    filesystem = _FaultInjectingFileSystem()
    if change == "mutation":
        filesystem.fstat_transforms[2] = lambda metadata: replace(
            metadata, modified_time_ns=metadata.modified_time_ns + 1
        )
    elif change == "truncation":
        filesystem.fstat_transforms[2] = lambda metadata: replace(
            metadata, byte_count=metadata.byte_count - 1
        )
    elif change == "replacement":
        filesystem.lstat_transforms[2] = lambda metadata: replace(
            metadata, file_id=metadata.file_id + 1
        )
    else:
        filesystem.lstat_transforms[2] = lambda metadata: replace(
            metadata, mode=stat.S_IFDIR
        )

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(),
        filesystem=filesystem,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ
    assert filesystem.opened_source is not None
    assert filesystem.opened_source.closed


def test_growth_past_limit_during_read_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"1234")
    filesystem = _FaultInjectingFileSystem()
    filesystem.injected_chunks[2] = b"5"

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(max_source_bytes=4),
        filesystem=filesystem,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_TOO_LARGE
    assert filesystem.opened_source is not None
    assert filesystem.opened_source.closed


def test_growth_past_limit_between_stat_and_open_is_rejected_before_read(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"1234")
    filesystem = _FaultInjectingFileSystem()
    filesystem.fstat_transforms[1] = lambda metadata: replace(metadata, byte_count=5)

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(max_source_bytes=4),
        filesystem=filesystem,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_TOO_LARGE
    assert filesystem.read_count == 0
    assert filesystem.opened_source is not None
    assert filesystem.opened_source.closed


def test_premature_end_of_read_is_reported_as_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"source bytes")
    filesystem = _FaultInjectingFileSystem()
    filesystem.injected_chunks[1] = b""

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(),
        filesystem=filesystem,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ
    assert filesystem.opened_source is not None
    assert filesystem.opened_source.closed


@pytest.mark.parametrize(
    "failure_point",
    ["pre_stat", "open", "opened_stat", "read", "post_stat", "path_stat"],
)
def test_io_failures_are_bounded_and_close_open_source(
    tmp_path: Path,
    failure_point: str,
) -> None:
    source = tmp_path / "private-source.txt"
    source.write_bytes(b"private payload")
    filesystem = _FaultInjectingFileSystem()
    if failure_point == "pre_stat":
        filesystem.lstat_errors.add(1)
    elif failure_point == "open":
        filesystem.open_error = True
    elif failure_point == "opened_stat":
        filesystem.fstat_errors.add(1)
    elif failure_point == "read":
        filesystem.read_errors.add(1)
    elif failure_point == "post_stat":
        filesystem.fstat_errors.add(2)
    else:
        filesystem.lstat_errors.add(2)

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(),
        filesystem=filesystem,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_READ_FAILED
    assert "private-source" not in repr(result)
    assert "private payload" not in repr(result)
    if filesystem.opened_source is not None:
        assert filesystem.opened_source.closed


@pytest.mark.parametrize("lstat_call", [1, 2])
def test_vanished_path_is_reported_as_changed_source(
    tmp_path: Path,
    lstat_call: int,
) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"source")
    filesystem = _FaultInjectingFileSystem()
    filesystem.missing_lstats.add(lstat_call)

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(),
        filesystem=filesystem,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ
    if filesystem.opened_source is not None:
        assert filesystem.opened_source.closed


def test_path_vanishing_at_open_is_reported_as_changed_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"source")
    filesystem = _FaultInjectingFileSystem()
    filesystem.open_missing = True

    result = hash_source_candidate(
        _candidate(source),
        _classification(),
        settings=fixture_settings(),
        filesystem=filesystem,
    )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ
    assert filesystem.opened_source is None


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
    with pytest.raises(TypeError, match="classification must be"):
        hash_source_candidate(
            candidate,
            object(),  # pyright: ignore[reportArgumentType]
            settings=fixture_settings(),
        )
    with pytest.raises(ValueError, match="must describe a supported source"):
        hash_source_candidate(
            candidate,
            unsupported,
            settings=fixture_settings(),
        )
    with pytest.raises(TypeError, match="settings must be Settings"):
        hash_source_candidate(
            candidate,
            _classification(),
            settings=object(),  # pyright: ignore[reportArgumentType]
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
