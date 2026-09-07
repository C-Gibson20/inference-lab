"""Versioned parser capability declarations used by corpus audits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from inference_lab._validation import is_canonical_semantic_version
from inference_lab.evaluation.audit_errors import (
    AuditContractError,
    AuditErrorCategory,
)
from inference_lab.ingest.parser import ParsedDocument
from inference_lab.ingest.versioning import (
    CURRENT_PARSER_VERSION,
    ParserVersion,
)

_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_STRUCTURE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "page",
        "section",
        "paragraph",
        "table",
        "figure_caption",
        "figure",
        "claim",
        "example",
        "reference",
        "other",
    }
)


def _validate_semantic_version(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_canonical_semantic_version(value):
        raise ValueError(f"{field_name} must be a canonical semantic version.")


@dataclass(frozen=True, slots=True)
class ParserCapabilityProfile:
    """Declare which observations one complete parser version can make."""

    parser_version: ParserVersion
    capability_version: str
    language: bool
    page_count: bool
    page_text: bool
    character_count: bool
    token_count: bool
    acquisition_statuses: frozenset[str]
    layout: bool
    bounding_boxes: bool
    source_spans: bool
    structure_kinds: frozenset[str]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.parser_version, ParserVersion):
            raise TypeError("parser_version must be a ParserVersion.")
        _validate_semantic_version(
            self.capability_version,
            field_name="capability_version",
        )
        for field_name in (
            "language",
            "page_count",
            "page_text",
            "character_count",
            "token_count",
            "layout",
            "bounding_boxes",
            "source_spans",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean.")
        if not isinstance(self.acquisition_statuses, frozenset):
            raise TypeError("acquisition_statuses must be a frozenset.")
        allowed_statuses = {"text_native", "scanned", "mixed", "unknown"}
        if not self.acquisition_statuses <= allowed_statuses:
            raise ValueError("acquisition_statuses contains an unsupported value.")
        if not isinstance(self.structure_kinds, frozenset):
            raise TypeError("structure_kinds must be a frozenset.")
        if not self.structure_kinds <= _STRUCTURE_KINDS:
            raise ValueError("structure_kinds contains an unsupported value.")
        if self.bounding_boxes and not self.layout:
            raise ValueError("bounding_boxes_require_layout")
        if "page" in self.structure_kinds and not self.page_text:
            raise ValueError("page_structure_requires_page_text")
        if self.page_text and not self.page_count:
            raise ValueError("page_text_requires_page_count")

    @property
    def value(self) -> str:
        """Return the complete parser capability identity."""
        return f"{self.parser_version.value}/capability-{self.capability_version}"


CURRENT_PARSER_CAPABILITY_PROFILE: Final = ParserCapabilityProfile(
    parser_version=CURRENT_PARSER_VERSION,
    capability_version="1.0.0",
    language=False,
    page_count=False,
    page_text=False,
    character_count=True,
    token_count=False,
    acquisition_statuses=frozenset({"text_native"}),
    layout=False,
    bounding_boxes=False,
    source_spans=True,
    structure_kinds=frozenset({"paragraph"}),
)

PARSER_CAPABILITY_PROFILES: Final = MappingProxyType(
    {CURRENT_PARSER_VERSION.value: CURRENT_PARSER_CAPABILITY_PROFILE}
)


def parser_capability_profile(
    parser_version: ParserVersion,
) -> ParserCapabilityProfile | None:
    """Return the registered profile for a complete parser version."""
    if not isinstance(parser_version, ParserVersion):
        raise TypeError("parser_version must be a ParserVersion.")
    return PARSER_CAPABILITY_PROFILES.get(parser_version.value)


def validate_parsed_document_capabilities(
    document: ParsedDocument,
    profile: ParserCapabilityProfile,
) -> None:
    """Reject parser output that contradicts its declared capabilities."""
    if not isinstance(document, ParsedDocument):
        raise TypeError("document must be a ParsedDocument.")
    if not isinstance(profile, ParserCapabilityProfile):
        raise TypeError("profile must be a ParserCapabilityProfile.")
    if document.parser_version != profile.parser_version:
        raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)

    if not profile.language and document.language is not None:
        raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)
    if not profile.page_count and document.page_count is not None:
        raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)
    if not profile.token_count and document.token_count is not None:
        raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)
    if not profile.source_spans and document.blocks:
        raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)
    if not profile.layout and document.page_layouts:
        raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)
    if not profile.bounding_boxes and any(
        block.source_span.bounding_box is not None for block in document.blocks
    ):
        raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)

    for block in document.blocks:
        if block.kind not in profile.structure_kinds:
            raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)

    collection_by_kind = {
        "page": document.pages,
        "section": document.sections,
        "paragraph": document.paragraphs,
    }
    for kind, collection in collection_by_kind.items():
        if kind in profile.structure_kinds:
            if collection is None:
                raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)
        elif collection is not None:
            raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)
