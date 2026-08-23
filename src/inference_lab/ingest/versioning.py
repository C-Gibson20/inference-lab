"""Represent the version of the parser used to process a document."""

import re
import uuid
from dataclasses import dataclass

from inference_lab.provenance import DocumentVersionId, SourceSpanId

_IMPLEMENTATION_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


@dataclass(frozen=True, slots=True)
class ParserVersion:
    """Represent the version of the parser used to process a document."""

    implementation: str
    contract_version: str
    normalisation_version: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Require unambiguous canonical parser-version components."""

        if not isinstance(
            self.implementation, str
        ) or not _IMPLEMENTATION_PATTERN.fullmatch(self.implementation):
            raise ValueError("parser_implementation_must_be_slug")
        if not isinstance(
            self.contract_version, str
        ) or not _SEMANTIC_VERSION_PATTERN.fullmatch(self.contract_version):
            raise ValueError("parser_contract_version_must_be_semantic_version")
        if not isinstance(
            self.normalisation_version, str
        ) or not _SEMANTIC_VERSION_PATTERN.fullmatch(self.normalisation_version):
            raise ValueError("normalisation_version_must_be_semantic_version")

    @property
    def value(self) -> str:
        """Return the canonical serialized parser-version identifier."""
        return (
            f"{self.implementation}/"
            f"contract-{self.contract_version}/"
            f"normalization-{self.normalisation_version}"
        )


CURRENT_PARSER_VERSION = ParserVersion(
    implementation="text-native",
    contract_version="1.0.0",
    normalisation_version="1.0.0",
)


def generate_source_span_id(
    document_version_id: DocumentVersionId,
    *,
    parser_version: ParserVersion,
    ordinal: int,
    kind: str,
    start_offset: int,
    end_offset: int,
) -> SourceSpanId:
    """Create a stable source-span ID for one document-version representation."""

    if not isinstance(document_version_id, DocumentVersionId):
        raise ValueError("invalid_document_version_id")
    if not isinstance(parser_version, ParserVersion):
        raise ValueError("invalid_parser_version")
    if not _is_integer(ordinal):
        raise ValueError("source_span_ordinal_must_be_integer")
    if not isinstance(kind, str):
        raise ValueError("source_span_kind_must_be_string")
    if not kind.strip():
        raise ValueError("source_span_kind_must_not_be_empty")
    if not _is_integer(start_offset) or not _is_integer(end_offset):
        raise ValueError("source_span_offsets_must_be_integers")
    if ordinal < 0:
        raise ValueError("source_span_ordinal_must_be_non_negative")
    if start_offset < 0:
        raise ValueError("source_span_start_must_be_non_negative")
    if end_offset < start_offset:
        raise ValueError("source_span_end_must_not_precede_start")
    if not kind:
        raise ValueError("source_span_kind_must_not_be_empty")

    name = (
        f"parser={parser_version.value};ordinal={ordinal};kind={kind};"
        f"start={start_offset};end={end_offset}"
    )
    namespace = uuid.UUID(document_version_id.as_str())
    generated = uuid.uuid5(namespace, name)
    return SourceSpanId.from_str(str(generated))


def _is_integer(value: object) -> bool:
    """Return whether a runtime value is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)
