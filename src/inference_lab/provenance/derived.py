"""Immutable metadata and version semantics for derived records."""

import hashlib
import json
from dataclasses import dataclass
from typing import Self

from inference_lab.provenance.errors import InvalidDerivedRecordError
from inference_lab.provenance.identifiers import RecordId
from inference_lab.provenance.lineage import LineageInputs
from inference_lab.provenance.operations import DigestAlgorithm, ProducingOperation
from inference_lab.provenance.utils import (
    is_lowercase_slug,
    is_sha256_hex_digest,
)
from inference_lab.provenance.versioning import validate_lineage_schema_version


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
    if algorithm is DigestAlgorithm.SHA256 and not is_sha256_hex_digest(digest):
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
        if not is_lowercase_slug(self.value):
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


def build_derivation_fingerprint(
    *,
    record_kind: DerivedRecordKind,
    lineage_schema_version: str,
    inputs: LineageInputs,
    operation: ProducingOperation,
    output_digest: OutputDigest,
) -> DerivationFingerprint:
    """Build the deterministic fingerprint for one semantic derivation."""
    if not isinstance(record_kind, DerivedRecordKind):
        raise TypeError("record_kind must be a DerivedRecordKind.")
    validate_lineage_schema_version(lineage_schema_version)
    if not isinstance(inputs, LineageInputs):
        raise TypeError("inputs must be LineageInputs.")
    if not isinstance(operation, ProducingOperation):
        raise TypeError("operation must be a ProducingOperation.")
    if not isinstance(output_digest, OutputDigest):
        raise TypeError("output_digest must be an OutputDigest.")

    payload = {
        "inputs": {
            "parent_records": [
                {
                    "input_ordinal": parent.input_ordinal,
                    "input_role": (
                        parent.input_role.as_str()
                        if parent.input_role is not None
                        else None
                    ),
                    "record_id": parent.record_id.as_str(),
                }
                for parent in inputs.parent_records
            ],
            "source_anchors": [
                {
                    "document_version_id": anchor.document_version_id.as_str(),
                    "input_ordinal": anchor.input_ordinal,
                    "input_role": (
                        anchor.input_role.as_str()
                        if anchor.input_role is not None
                        else None
                    ),
                    "source_span_id": anchor.source_span_id.as_str(),
                }
                for anchor in inputs.source_anchors
            ],
        },
        "lineage_schema_version": lineage_schema_version,
        "operation": {
            "components": [
                {
                    "artefact_identity": component.artefact_identity,
                    "artefact_revision": component.artefact_revision,
                    "contract_version": component.contract_version,
                    "implementation": component.implementation,
                    "role": component.role.as_str(),
                }
                for component in operation.components
            ],
            "configuration_identity": {
                "algorithm": operation.configuration_identity.algorithm.value,
                "digest": operation.configuration_identity.digest,
            },
            "contract_version": operation.contract_version,
            "implementation": operation.implementation,
            "kind": operation.kind.as_str(),
        },
        "output_digest": {
            "algorithm": output_digest.algorithm.value,
            "digest": output_digest.digest,
        },
        "record_kind": record_kind.as_str(),
    }
    canonical_bytes = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return DerivationFingerprint(
        algorithm=DigestAlgorithm.SHA256,
        digest=hashlib.sha256(canonical_bytes).hexdigest(),
    )


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
        expected_fingerprint = build_derivation_fingerprint(
            record_kind=self.record_kind,
            lineage_schema_version=self.lineage_schema_version,
            inputs=self.inputs,
            operation=self.operation,
            output_digest=self.output_digest,
        )
        if self.derivation_fingerprint != expected_fingerprint:
            raise InvalidDerivedRecordError(
                "derivation_fingerprint does not match derived-record metadata."
            )
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
