"""Immutable storage contracts for parser outcomes and bounded failures."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from inference_lab._validation import is_canonical_semantic_version
from inference_lab.provenance import DocumentId, DocumentVersionId, RecordId
from inference_lab.storage.producers import ProducerIdentity
from inference_lab.storage.source_versioning import (
    CURRENT_SOURCE_RECORD_SCHEMA_VERSION,
    validate_source_record_schema_version,
)


class ParseStatus(StrEnum):
    """Closed lifecycle states for a persisted parse attempt."""

    SUCCEEDED = "succeeded"
    COMPLETED_WITH_RECOVERABLE_FAILURES = "completed_with_recoverable_failures"
    FAILED = "failed"


class ParseFailureCategory(StrEnum):
    """Bounded parser failure categories safe for durable storage."""

    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_FORMAT = "unsupported_format"
    INVALID_ENCODING = "invalid_encoding"
    DECODE_FAILED = "decode_failed"
    MALFORMED_SOURCE = "malformed_source"
    MALFORMED_UNIT = "malformed_unit"
    UNSUPPORTED_STRUCTURE = "unsupported_structure"
    INVALID_OUTPUT = "invalid_output"
    SOURCE_TOO_LARGE = "source_too_large"


class ParseFailureSeverity(StrEnum):
    """Closed severity values for persisted parser failures."""

    FATAL = "fatal"
    RECOVERABLE = "recoverable"


class ParseFailureStage(StrEnum):
    """Closed parser stages that may own a persisted failure."""

    INPUT = "input"
    DECODE = "decode"
    STRUCTURE = "structure"
    OUTPUT = "output"


class ParseFailureUnitKind(StrEnum):
    """Closed structural unit kinds for recoverable parser failures."""

    PAGE = "page"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    OTHER = "other"


_FAILURE_POLICY: Final[
    dict[ParseFailureCategory, tuple[ParseFailureSeverity, ParseFailureStage]]
] = {
    ParseFailureCategory.INVALID_INPUT: (
        ParseFailureSeverity.FATAL,
        ParseFailureStage.INPUT,
    ),
    ParseFailureCategory.UNSUPPORTED_FORMAT: (
        ParseFailureSeverity.FATAL,
        ParseFailureStage.INPUT,
    ),
    ParseFailureCategory.INVALID_ENCODING: (
        ParseFailureSeverity.FATAL,
        ParseFailureStage.INPUT,
    ),
    ParseFailureCategory.SOURCE_TOO_LARGE: (
        ParseFailureSeverity.FATAL,
        ParseFailureStage.INPUT,
    ),
    ParseFailureCategory.DECODE_FAILED: (
        ParseFailureSeverity.FATAL,
        ParseFailureStage.DECODE,
    ),
    ParseFailureCategory.MALFORMED_SOURCE: (
        ParseFailureSeverity.FATAL,
        ParseFailureStage.STRUCTURE,
    ),
    ParseFailureCategory.MALFORMED_UNIT: (
        ParseFailureSeverity.RECOVERABLE,
        ParseFailureStage.STRUCTURE,
    ),
    ParseFailureCategory.UNSUPPORTED_STRUCTURE: (
        ParseFailureSeverity.RECOVERABLE,
        ParseFailureStage.STRUCTURE,
    ),
    ParseFailureCategory.INVALID_OUTPUT: (
        ParseFailureSeverity.FATAL,
        ParseFailureStage.OUTPUT,
    ),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ParseFailureRecord:
    """Persist one bounded parser failure without source or exception payloads."""

    category: ParseFailureCategory
    severity: ParseFailureSeverity
    stage: ParseFailureStage
    unit_kind: ParseFailureUnitKind | None = None
    unit_index: int | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the failure record is self-consistent."""
        if not isinstance(self.category, ParseFailureCategory):
            raise TypeError("category must be a ParseFailureCategory.")
        if not isinstance(self.severity, ParseFailureSeverity):
            raise TypeError("severity must be a ParseFailureSeverity.")
        if not isinstance(self.stage, ParseFailureStage):
            raise TypeError("stage must be a ParseFailureStage.")
        if self.unit_kind is not None and not isinstance(
            self.unit_kind, ParseFailureUnitKind
        ):
            raise TypeError("unit_kind must be a ParseFailureUnitKind or None.")
        if self.unit_index is not None and (
            not isinstance(self.unit_index, int) or isinstance(self.unit_index, bool)
        ):
            raise TypeError("unit_index must be an integer or None.")
        if self.unit_index is not None and self.unit_index < 0:
            raise ValueError("unit_index must not be negative.")
        self._validate_policy()

    def _validate_policy(self) -> None:
        """Require category, severity, stage, and unit location to agree."""
        expected_severity, expected_stage = _FAILURE_POLICY[self.category]
        if (self.severity, self.stage) != (expected_severity, expected_stage):
            raise ValueError("failure severity and stage must match its category.")
        has_unit = self.unit_kind is not None
        if has_unit != (self.unit_index is not None):
            raise ValueError("failure unit kind and index must be supplied together.")
        if self.severity is ParseFailureSeverity.FATAL and has_unit:
            raise ValueError("fatal failures must not identify a partial unit.")
        if self.severity is ParseFailureSeverity.RECOVERABLE and not has_unit:
            raise ValueError("recoverable failures must identify an omitted unit.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ParseOutcome:
    """Persist one immutable parser outcome and its bounded failures."""

    parse_id: RecordId
    document_id: DocumentId
    document_version_id: DocumentVersionId
    status: ParseStatus
    parser: ProducerIdentity
    normalisation_version: str
    parsed_record_id: RecordId | None
    failures: tuple[ParseFailureRecord, ...]
    ocr_producer: ProducerIdentity | None = None
    record_schema_version: str = CURRENT_SOURCE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the parse outcome is self-consistent."""
        validate_source_record_schema_version(self.record_schema_version)
        if not isinstance(self.parse_id, RecordId):
            raise TypeError("parse_id must be a RecordId.")
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.status, ParseStatus):
            raise TypeError("status must be a ParseStatus.")
        if not isinstance(self.parser, ProducerIdentity):
            raise TypeError("parser must be a ProducerIdentity.")
        if not isinstance(self.normalisation_version, str):
            raise TypeError("normalisation_version must be a string.")
        if not is_canonical_semantic_version(self.normalisation_version):
            raise ValueError(
                "normalisation_version must be a canonical semantic version."
            )
        if self.parsed_record_id is not None and not isinstance(
            self.parsed_record_id, RecordId
        ):
            raise TypeError("parsed_record_id must be a RecordId or None.")
        if not isinstance(self.failures, tuple):
            raise TypeError("failures must be a tuple.")
        if any(not isinstance(value, ParseFailureRecord) for value in self.failures):
            raise TypeError("failures must contain only ParseFailureRecord values.")
        if self.ocr_producer is not None and not isinstance(
            self.ocr_producer, ProducerIdentity
        ):
            raise TypeError("ocr_producer must be a ProducerIdentity or None.")
        self._validate_status()

    def _validate_status(self) -> None:
        """Validate parsed-record and failure relationships for the status."""
        if self.status is ParseStatus.SUCCEEDED:
            if self.parsed_record_id is None or self.failures:
                raise ValueError(
                    "a successful parse requires a representation and no failures."
                )
            return
        if self.status is ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES:
            if self.parsed_record_id is None or not self.failures:
                raise ValueError(
                    "a recoverable parse requires a representation and failures."
                )
            if any(
                failure.severity is ParseFailureSeverity.FATAL
                for failure in self.failures
            ):
                raise ValueError("a recoverable parse cannot contain fatal failures.")
            return
        if self.parsed_record_id is not None or not self.failures:
            raise ValueError("a failed parse requires failures and no representation.")
        if not any(
            failure.severity is ParseFailureSeverity.FATAL for failure in self.failures
        ):
            raise ValueError("a failed parse requires at least one fatal failure.")
