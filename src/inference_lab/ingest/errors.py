"""Error classes for handling parsing and decoding failures."""

from typing import Literal
from dataclasses import dataclass

ParserFailureCategory = Literal[
    "invalid_input",
    "unsupported_format",
    "invalid_encoding",
    "decode_failed",
    "malformed_source",
    "malformed_unit",
    "unsupported_structure",
    "invalid_output",
    "source_too_large",
]

ParserFailureSeverity = Literal["fatal", "recoverable"]

ParserFailureStage = Literal[
    "input",
    "decode",
    "structure",
    "output",
]


class ParserDecodeError(UnicodeError):
    """Represent a source decoding failure without exposing source content."""

    category: Literal["decode_failed"] = "decode_failed"

    def __init__(self, encoding: str) -> None:
        """Create a safe decoding error identified by its encoding."""
        super().__init__("decode_failed")
        self.encoding = encoding


@dataclass(frozen=True, slots=True)
class ParserFailure:
    """Describe one parser failure without retaining sensitive source data."""

    category: ParserFailureCategory
    severity: ParserFailureSeverity
    stage: ParserFailureStage
    unit_kind: str | None = None
    unit_index: int | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate failure identity and location metadata."""

        if self.unit_kind is not None and not self.unit_kind.strip():
            raise ValueError("failure_unit_kind_must_not_be_empty")

        if self.unit_index is not None:
            if not isinstance(self.unit_index, int) or isinstance(
                self.unit_index, bool
            ):
                raise ValueError("failure_unit_index_must_be_integer")
            if self.unit_index < 0:
                raise ValueError("failure_unit_index_must_be_non_negative")

        if self.severity == "fatal" and self.unit_index is not None:
            raise ValueError("fatal_failure_must_not_identify_partial_unit")
