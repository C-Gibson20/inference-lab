"""Storage-neutral lifecycle contract for version-pinned corpus readers."""

from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from inference_lab.provenance import CorpusId
from inference_lab.storage.versioning import CorpusVersion, CorpusVersionManifest


@runtime_checkable
class CorpusSnapshot(Protocol):
    """Context-managed reader pinned to one complete committed corpus version."""

    @property
    def corpus_id(self) -> CorpusId:
        """Return the selected corpus identity."""
        ...

    @property
    def version(self) -> CorpusVersion:
        """Return the immutable committed version selected on entry."""
        ...

    @property
    def manifest(self) -> CorpusVersionManifest:
        """Return the complete materialised manifest selected on entry."""
        ...

    @property
    def is_closed(self) -> bool:
        """Return whether the snapshot has permanently released its reader."""
        ...

    def __enter__(self) -> Self:
        """Open and return this snapshot, rejecting nested reuse."""
        ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Close the snapshot while preserving any primary exception."""
        ...

    def close(self) -> None:
        """Release the owned reader idempotently and prevent further use."""
        ...
