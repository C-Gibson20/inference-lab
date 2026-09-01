"""Immutable metadata and version semantics for derived records."""

import re
from dataclasses import dataclass
from typing import Final, Self

from inference_lab.provenance.errors import InvalidDerivedRecordError
from inference_lab.provenance.identifiers import RecordId
from inference_lab.provenance.lineage import LineageInputs
from inference_lab.provenance.operations import DigestAlgorithm, ProducingOperation
from inference_lab.provenance.versioning import validate_lineage_schema_version

_RECORD_KIND_PATTERN: Final = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SHA256_HEX_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


def _validate_digest(
    algorithm: object,
    digest: object,
    *,
    field_name: str,
) -> None:
    """Validate one canonical SHA-256 lineage digest."""
    if not isinstance(algorithm, DigestAlgorithm):
        raise TypeError(f"{field_name} algorithm must be a DigestAlgorithm.")
    if not isinstance(digest, str):
        raise TypeError(f"{field_name} digest must be a string.")
    if (
        algorithm is DigestAlgorithm.SHA256
        and _SHA256_HEX_PATTERN.fullmatch(digest) is None
    ):
        raise InvalidDerivedRecordError(
            f"{field_name} digest must be 64 lowercase hexadecimal characters."
        )


@dataclass(frozen=True, slots=True)
class DerivedRecordKind:
    """Validated, extensible kind of immutable derived record."""

    value: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the record kind value against the slug pattern."""
        if not isinstance(self.value, str):
            raise TypeError("record_kind must be a string.")
        if _RECORD_KIND_PATTERN.fullmatch(self.value) is None:
            raise InvalidDerivedRecordError("record_kind must be a lowercase slug.")

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Reconstruct a record kind from its canonical string value."""
        return cls(value)

    def as_str(self) -> str:
        """Return the canonical record-kind value."""
        return self.value


@dataclass(frozen=True, slots=True)
class OutputDigest:
    """Canonical digest of the immutable domain-owned record payload."""

    algorithm: DigestAlgorithm
    digest: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the output digest against its algorithm."""
        _validate_digest(self.algorithm, self.digest, field_name="output")

    def as_str(self) -> str:
        """Return the algorithm-qualified output digest."""
        return f"{self.algorithm.value}:{self.digest}"


@dataclass(frozen=True, slots=True)
class DerivationFingerprint:
    """Canonical digest describing one complete immutable derivation."""

    algorithm: DigestAlgorithm
    digest: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the derivation fingerprint against its algorithm."""
        _validate_digest(self.algorithm, self.digest, field_name="fingerprint")

    def as_str(self) -> str:
        """Return the algorithm-qualified derivation fingerprint."""
        return f"{self.algorithm.value}:{self.digest}"


@dataclass(frozen=True, slots=True)
class DerivedRecordMetadata:
    """Describe one immutable derived-record version and its direct lineage."""

    record_id: RecordId
    record_kind: DerivedRecordKind
    lineage_schema_version: str
    inputs: LineageInputs
    operation: ProducingOperation
    output_digest: OutputDigest
    derivation_fingerprint: DerivationFingerprint
    supersedes_record_id: RecordId | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the derived record metadata fields."""
        if not isinstance(self.record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        if not isinstance(self.record_kind, DerivedRecordKind):
            raise TypeError("record_kind must be a DerivedRecordKind.")
        validate_lineage_schema_version(self.lineage_schema_version)
        if not isinstance(self.inputs, LineageInputs):
            raise TypeError("inputs must be LineageInputs.")
        if not isinstance(self.operation, ProducingOperation):
            raise TypeError("operation must be a ProducingOperation.")
        if not isinstance(self.output_digest, OutputDigest):
            raise TypeError("output_digest must be an OutputDigest.")
        if not isinstance(self.derivation_fingerprint, DerivationFingerprint):
            raise TypeError("derivation_fingerprint must be a DerivationFingerprint.")
        self._validate_supersession()

    def _validate_supersession(self) -> None:
        """Validate the locally knowable supersession invariants."""
        if self.supersedes_record_id is not None and not isinstance(
            self.supersedes_record_id,
            RecordId,
        ):
            raise TypeError("supersedes_record_id must be a RecordId or None.")
        if self.supersedes_record_id == self.record_id:
            raise InvalidDerivedRecordError("a derived record cannot supersede itself.")
