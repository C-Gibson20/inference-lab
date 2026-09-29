"""Represent the version of the parser used to process a document."""

import uuid
from dataclasses import dataclass

from inference_lab._validation import (
    is_canonical_semantic_version,
    is_lowercase_slug,
    is_strict_integer,
)
from inference_lab.provenance import DocumentVersionId, SourceSpanId


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
        if not is_lowercase_slug(self.implementation):
            raise ValueError("parser_implementation_must_be_slug")
        if not is_canonical_semantic_version(self.contract_version):
            raise ValueError("parser_contract_version_must_be_semantic_version")
        if not is_canonical_semantic_version(self.normalisation_version):
            raise ValueError("normalisation_version_must_be_semantic_version")

    @property
    def value(self) -> str:
        """Return the canonical serialised parser-version identifier."""
        return (
            f"{self.implementation}/"
            f"contract-{self.contract_version}/"
            f"normalisation-{self.normalisation_version}"
        )


CURRENT_PARSER_VERSION = ParserVersion(
    implementation="text-native",
    contract_version="4.0.0",
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
    if not is_strict_integer(ordinal):
        raise ValueError("source_span_ordinal_must_be_integer")
    if not isinstance(kind, str):
        raise ValueError("source_span_kind_must_be_string")
    if not kind.strip():
        raise ValueError("source_span_kind_must_not_be_empty")
    if not is_strict_integer(start_offset) or not is_strict_integer(end_offset):
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
