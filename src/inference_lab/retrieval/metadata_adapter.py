"""Resolve lexical-candidate metadata through one pinned corpus snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from inference_lab.provenance import SourceSpanId
from inference_lab.retrieval.lexical_store_models import LexicalIndexedChunk
from inference_lab.retrieval.metadata_filters import MetadataValues
from inference_lab.storage import (
    CorpusSnapshot,
    CorruptStorageError,
    ParsedRepresentation,
    ResolvedSourceSpan,
    SourceArtefact,
)


@dataclass(frozen=True, slots=True)
class LexicalCandidateMetadata:
    """Immutable metadata and provenance resolved for one lexical candidate."""

    source_artefact: SourceArtefact
    parsed_representation: ParsedRepresentation
    source_span: ResolvedSourceSpan
    values: MetadataValues

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that all fields are of the expected types."""
        if not isinstance(self.source_artefact, SourceArtefact):
            raise TypeError("source_artefact must be a SourceArtefact.")
        if not isinstance(self.parsed_representation, ParsedRepresentation):
            raise TypeError("parsed_representation must be a ParsedRepresentation.")
        if not isinstance(self.source_span, ResolvedSourceSpan):
            raise TypeError("source_span must be a ResolvedSourceSpan.")
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping.")


def resolve_lexical_candidate_metadata(
    snapshot: CorpusSnapshot,
    candidate: LexicalIndexedChunk,
) -> LexicalCandidateMetadata:
    """Resolve one candidate's source and exact parsed-record metadata.

    A missing reachable record or an ambiguous link is corrupt state, not an
    ordinary missing value.
    """
    if not isinstance(candidate, LexicalIndexedChunk):
        raise TypeError("candidate must be a LexicalIndexedChunk.")
    _validate_snapshot(snapshot)
    if not candidate.source_links:
        raise CorruptStorageError()

    links = candidate.source_links
    if any(link.document_version_id != candidate.document_version_id for link in links):
        raise CorruptStorageError()

    resolved_spans = tuple(
        _resolve_source_span(snapshot, link.source_span_id) for link in links
    )
    first_span = resolved_spans[0]
    if any(
        span.document_id != candidate.document_id
        or span.location.document_version_id != candidate.document_version_id
        or span.location.source_span_id != link.source_span_id
        for link, span in zip(links, resolved_spans, strict=True)
    ):
        raise CorruptStorageError()

    parsed_record_ids = {span.location.parsed_record_id for span in resolved_spans}
    if len(parsed_record_ids) != 1:
        raise CorruptStorageError()
    parsed_record_id = first_span.location.parsed_record_id

    source_artefact = snapshot.get_source_artefact(candidate.document_version_id)
    if source_artefact is None or (
        source_artefact.document_id != candidate.document_id
        or source_artefact.document_version_id != candidate.document_version_id
    ):
        raise CorruptStorageError()

    parsed_representation = snapshot.get_parsed_representation(parsed_record_id)
    if parsed_representation is None or (
        parsed_representation.document_id != candidate.document_id
        or parsed_representation.document_version_id != candidate.document_version_id
        or parsed_representation.record_id != parsed_record_id
    ):
        raise CorruptStorageError()

    values: MetadataValues = MappingProxyType(
        {
            "document-type": source_artefact.media_type,
            "source-byte-count": source_artefact.byte_count,
            "language": parsed_representation.language,
            "page-count": parsed_representation.page_count,
            "character-count": parsed_representation.character_count,
            "token-count": parsed_representation.token_count,
        }
    )
    return LexicalCandidateMetadata(
        source_artefact=source_artefact,
        parsed_representation=parsed_representation,
        source_span=first_span,
        values=values,
    )


def _validate_snapshot(snapshot: CorpusSnapshot) -> None:
    """Require the narrow snapshot read surface used by this adapter."""
    if any(
        not callable(getattr(snapshot, method_name, None))
        for method_name in (
            "get_source_artefact",
            "get_parsed_representation",
            "resolve_source_span",
        )
    ):
        raise TypeError("snapshot must provide corpus snapshot reads.")


def _resolve_source_span(
    snapshot: CorpusSnapshot,
    source_span_id: SourceSpanId,
) -> ResolvedSourceSpan:
    """Resolve one required source span or fail closed as corrupt state."""
    resolved = snapshot.resolve_source_span(source_span_id)
    if resolved is None:
        raise CorruptStorageError()
    return resolved
