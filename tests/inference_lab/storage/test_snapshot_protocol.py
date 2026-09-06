"""Tests for the structural corpus snapshot lifecycle protocol."""

from datetime import UTC, datetime
from types import TracebackType
from typing import Self

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage import CorpusSnapshot, CorpusVersion, CorpusVersionManifest

CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002")
MANIFEST = CorpusVersionManifest(CORPUS_ID, VERSION_ID)
VERSION = CorpusVersion(
    CORPUS_ID,
    VERSION_ID,
    1,
    None,
    MANIFEST.manifest_digest,
    datetime(2026, 9, 5, 15, 0, tzinfo=UTC),
)


class StructuralSnapshot:
    """Complete structural snapshot independent of concrete storage adapters."""

    corpus_id = CORPUS_ID
    version = VERSION
    manifest = MANIFEST
    is_closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    def close(self) -> None:
        pass


def test_snapshot_protocol_is_structural_and_runtime_checkable() -> None:
    assert isinstance(StructuralSnapshot(), CorpusSnapshot)


def test_object_without_owned_close_lifecycle_is_not_snapshot() -> None:
    class SnapshotWithoutClose:
        corpus_id = CORPUS_ID
        version = VERSION
        manifest = MANIFEST
        is_closed = False

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            return False

    assert not isinstance(SnapshotWithoutClose(), CorpusSnapshot)
