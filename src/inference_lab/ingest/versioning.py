"""Represent the version of the parser used to process a document."""

import uuid
from dataclasses import dataclass

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
        """Reject incomplete parser-version components."""

        if not isinstance(self.implementation, str) or not self.implementation.strip():
            raise ValueError("parser_implementation_must_not_be_empty")
        if (
            not isinstance(self.contract_version, str)
            or not self.contract_version.strip()
        ):
            raise ValueError("parser_contract_version_must_not_be_empty")
        if (
            not isinstance(self.normalisation_version, str)
            or not self.normalisation_version.strip()
        ):
            raise ValueError("normalisation_version_must_not_be_empty")

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
    ordinal: int,
    kind: str,
    start_offset: int,
    end_offset: int,
) -> SourceSpanId:
    """Create a stable source-span ID for one document-version representation."""

    if not isinstance(document_version_id, DocumentVersionId):
        raise ValueError("invalid_document_version_id")
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

    name = f"ordinal={ordinal};kind={kind};start={start_offset};end={end_offset}"
    namespace = uuid.UUID(document_version_id.as_str())
    generated = uuid.uuid5(namespace, name)
    return SourceSpanId.from_str(str(generated))


def _is_integer(value: object) -> bool:
    """Return whether a runtime value is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)
