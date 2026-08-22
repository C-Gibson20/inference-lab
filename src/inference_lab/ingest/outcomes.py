"""Result contracts for parser success and failure outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from inference_lab.ingest.errors import ParserFailure

if TYPE_CHECKING:
    from inference_lab.ingest.parser import ParsedDocument

ParseStatus = Literal[
    "succeeded",
    "completed_with_recoverable_failures",
    "failed",
]


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
        if self.status == "succeeded":
            if self.document is None:
                raise ValueError("successful_parse_requires_document")
            if self.failures:
                raise ValueError("successful_parse_must_not_have_failures")

        elif self.status == "completed_with_recoverable_failures":
            if self.document is None:
                raise ValueError("partial_parse_requires_document")
            if not self.failures:
                raise ValueError("partial_parse_requires_failures")
            if any(failure.severity == "fatal" for failure in self.failures):
                raise ValueError("partial_parse_must_not_have_fatal_failures")

        elif self.status == "failed":
            if self.document is not None:
                raise ValueError("failed_parse_must_not_have_document")
            if not self.failures:
                raise ValueError("failed_parse_requires_failures")
            if not any(failure.severity == "fatal" for failure in self.failures):
                raise ValueError("failed_parse_requires_fatal_failure")
