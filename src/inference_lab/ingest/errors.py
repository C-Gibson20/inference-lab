"""Error classes for handling parsing and decoding failures."""

from dataclasses import dataclass
from typing import Literal

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

ParserFailureUnitKind = Literal[
    "page",
    "section",
    "paragraph",
    "table",
    "figure_caption",
    "other",
]

_FAILURE_POLICIES: dict[
    ParserFailureCategory,
    tuple[ParserFailureSeverity, ParserFailureStage],
] = {
    "invalid_input": ("fatal", "input"),
    "unsupported_format": ("fatal", "input"),
    "invalid_encoding": ("fatal", "input"),
    "source_too_large": ("fatal", "input"),
    "decode_failed": ("fatal", "decode"),
    "malformed_source": ("fatal", "structure"),
    "malformed_unit": ("recoverable", "structure"),
    "unsupported_structure": ("recoverable", "structure"),
    "invalid_output": ("fatal", "output"),
}

_FAILURE_UNIT_KINDS: frozenset[str] = frozenset(
    {"page", "section", "paragraph", "table", "figure_caption", "other"}
)


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
    unit_kind: ParserFailureUnitKind | None = None
    unit_index: int | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate failure identity and location metadata."""

        if not isinstance(self.category, str) or self.category not in _FAILURE_POLICIES:
            raise ValueError("invalid_parser_failure_category")
        if not isinstance(self.severity, str) or self.severity not in {
            "fatal",
            "recoverable",
        }:
            raise ValueError("invalid_parser_failure_severity")
        if not isinstance(self.stage, str) or self.stage not in {
            "input",
            "decode",
            "structure",
            "output",
        }:
            raise ValueError("invalid_parser_failure_stage")

        expected_severity, expected_stage = _FAILURE_POLICIES[self.category]
        if self.severity != expected_severity:
            raise ValueError("failure_severity_does_not_match_category")
        if self.stage != expected_stage:
            raise ValueError("failure_stage_does_not_match_category")

        if self.unit_kind is not None and (
            not isinstance(self.unit_kind, str)
            or self.unit_kind not in _FAILURE_UNIT_KINDS
        ):
            raise ValueError("invalid_failure_unit_kind")

        if self.unit_index is not None:
            if not isinstance(self.unit_index, int) or isinstance(
                self.unit_index, bool
            ):
                raise ValueError("failure_unit_index_must_be_integer")
            if self.unit_index < 0:
                raise ValueError("failure_unit_index_must_be_non_negative")

        has_unit_kind = self.unit_kind is not None
        has_unit_index = self.unit_index is not None
        if has_unit_kind != has_unit_index:
            raise ValueError("failure_unit_location_must_be_complete")
        if self.severity == "fatal" and has_unit_kind:
            raise ValueError("fatal_failure_must_not_identify_partial_unit")
        if self.severity == "recoverable" and not has_unit_kind:
            raise ValueError("recoverable_failure_requires_unit_location")
