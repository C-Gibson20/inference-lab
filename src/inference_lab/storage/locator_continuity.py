"""Private deployment-local contracts for source-locator continuity."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

from inference_lab._validation import is_local_path_shape, is_strict_integer
from inference_lab.provenance import ContentHash, CorpusId, SourceObservationId

_SQLITE_MAX_INTEGER: Final = 2**63 - 1


def _validate_sqlite_integer(value: object, *, field_name: str) -> None:
    """Validate a non-negative integer representable by SQLite."""
    if not is_strict_integer(value):
        raise TypeError(f"{field_name} must be an integer.")
    integer_value = cast(int, value)
    if not 0 <= integer_value <= _SQLITE_MAX_INTEGER:
        raise ValueError(f"{field_name} must be a non-negative SQLite integer.")


@dataclass(frozen=True, slots=True)
class SourceFileIdentityHint:
    """Carry non-authoritative filesystem identity for persisted continuity."""

    device_id: int
    file_id: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate bounded platform identity components."""
        _validate_sqlite_integer(self.device_id, field_name="device_id")
        _validate_sqlite_integer(self.file_id, field_name="file_id")


@dataclass(frozen=True, slots=True, repr=False)
class SourceFingerprint:
    """Describe the last successfully verified bytes without exposing a digest."""

    content_hash: ContentHash = field(repr=False)
    byte_count: int
    modified_time_ns: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the verified hash and bounded stat metadata."""
        if not isinstance(self.content_hash, ContentHash):
            raise TypeError("content_hash must be a ContentHash.")
        _validate_sqlite_integer(self.byte_count, field_name="byte_count")
        _validate_sqlite_integer(
            self.modified_time_ns,
            field_name="modified_time_ns",
        )

    def __repr__(self) -> str:
        """Return a representation that does not disclose the source digest."""
        return (
            "SourceFingerprint("
            f"byte_count={self.byte_count}, "
            f"modified_time_ns={self.modified_time_ns}, "
            "content_hash=<private>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class SourceLocatorRevision:
    """Bind one corpus-owned observation revision to a private local locator."""

    corpus_id: CorpusId
    source_observation_id: SourceObservationId
    locator_revision: int
    path: Path = field(repr=False)
    identity_hint: SourceFileIdentityHint | None
    fingerprint: SourceFingerprint = field(repr=False)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate ownership, revision, private path, and continuity evidence."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.source_observation_id, SourceObservationId):
            raise TypeError("source_observation_id must be a SourceObservationId.")
        _validate_sqlite_integer(
            self.locator_revision,
            field_name="locator_revision",
        )
        if self.locator_revision == 0:
            raise ValueError("locator_revision must be greater than zero.")
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path.")
        if (
            not is_local_path_shape(self.path)
            or not self.path.is_absolute()
            or ".." in self.path.parts
        ):
            raise ValueError("path must be an absolute normalised local path.")
        if self.identity_hint is not None and not isinstance(
            self.identity_hint,
            SourceFileIdentityHint,
        ):
            raise TypeError("identity_hint must be a SourceFileIdentityHint or None.")
        if not isinstance(self.fingerprint, SourceFingerprint):
            raise TypeError("fingerprint must be a SourceFingerprint.")

    def __repr__(self) -> str:
        """Return a representation that suppresses locator and hash material."""
        return (
            "SourceLocatorRevision("
            f"corpus_id={self.corpus_id!r}, "
            f"source_observation_id={self.source_observation_id!r}, "
            f"locator_revision={self.locator_revision}, "
            f"identity_hint={self.identity_hint!r}, "
            "path=<private>, fingerprint=<private>)"
        )
