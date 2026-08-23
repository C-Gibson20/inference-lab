"""Result contracts for parser success and failure outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, get_args

from inference_lab.ingest.errors import ParserFailure

if TYPE_CHECKING:
    from inference_lab.ingest.parser import ParsedDocument

ParseStatus = Literal[
    "succeeded",
    "completed_with_recoverable_failures",
    "failed",
]

_PARSE_STATUSES: Final[frozenset[ParseStatus]] = frozenset(get_args(ParseStatus))


def _validate_success(
    document: ParsedDocument | None,
    failures: tuple[ParserFailure, ...],
) -> None:
    """Validate the invariants of a fully successful parse."""
    if document is None:
        raise ValueError("successful_parse_requires_document")
    if failures:
        raise ValueError("successful_parse_must_not_have_failures")


def _validate_recoverable_completion(
    document: ParsedDocument | None,
    failures: tuple[ParserFailure, ...],
) -> None:
    """Validate a parse completed with recoverable failures."""
    if document is None:
        raise ValueError("partial_parse_requires_document")
    if not failures:
        raise ValueError("partial_parse_requires_failures")
    if any(failure.severity == "fatal" for failure in failures):
        raise ValueError("partial_parse_must_not_have_fatal_failures")


def _validate_failure(
    document: ParsedDocument | None,
    failures: tuple[ParserFailure, ...],
) -> None:
    """Validate the invariants of a failed parse."""
    if document is not None:
        raise ValueError("failed_parse_must_not_have_document")
    if not failures:
        raise ValueError("failed_parse_requires_failures")
    if not any(failure.severity == "fatal" for failure in failures):
        raise ValueError("failed_parse_requires_fatal_failure")


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    """Represent complete, partial, or failed parser work."""

    status: ParseStatus
    document: ParsedDocument | None
    failures: tuple[ParserFailure, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Prevent inconsistent success, partial, and failure states."""
        from inference_lab.ingest.parser import ParsedDocument

        if not isinstance(self.status, str) or self.status not in _PARSE_STATUSES:
            raise ValueError("invalid_parse_status")
        if self.document is not None and not isinstance(self.document, ParsedDocument):
            raise ValueError("invalid_parsed_document")
        if not isinstance(self.failures, tuple):
            raise ValueError("parse_failures_must_be_tuple")
        if not all(isinstance(failure, ParserFailure) for failure in self.failures):
            raise ValueError("invalid_parser_failure")

        if self.status == "succeeded":
            _validate_success(self.document, self.failures)
        elif self.status == "completed_with_recoverable_failures":
            _validate_recoverable_completion(self.document, self.failures)
        else:
            _validate_failure(self.document, self.failures)
