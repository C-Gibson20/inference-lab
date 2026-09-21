"""Public contracts for corpus-scoped lexical search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    LexicalMatchMode,
    LexicalQuery,
    RetrievalLimit,
    RetrievalQuery,
)
from inference_lab.retrieval.filters import _FILTER_TYPES, FilterExpression
from inference_lab.storage import (
    BoundingBox,
    ContentOrigin,
    PageLayoutDescriptor,
    StructuralBlockKind,
)


class ServingSearchFailureCategory(StrEnum):
    """Bounded failures raised by the serving search facade."""

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_FILTER = "unsupported_filter"
    CORPUS_NOT_FOUND = "corpus_not_found"
    NO_PUBLISHED_VERSION = "no_published_version"
    HISTORICAL_VERSION_NOT_FOUND = "historical_version_not_found"
    FOREIGN_CORPUS_VERSION = "foreign_corpus_version"
    RETRIEVAL_FAILED = "retrieval_failed"
    INVALID_RETRIEVAL_RESULT = "invalid_retrieval_result"
    CORRUPT_STATE = "corrupt_state"
    STORAGE_UNAVAILABLE = "storage_unavailable"


class ServingSearchError(ValueError):
    """Expose a bounded, payload-free operator-facing search error."""

    def __init__(self, category: ServingSearchFailureCategory) -> None:
        if not isinstance(category, ServingSearchFailureCategory):
            raise TypeError("category must be a ServingSearchFailureCategory.")
        super().__init__(category.value)
        self.category = category


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ServingSearchRequest:
    """Describe one explicit corpus-scoped lexical search."""

    corpus_id: CorpusId
    text: str
    limit: int
    corpus_version: CorpusVersionId | None = None
    filter: FilterExpression | None = None
    mode: LexicalMatchMode = LexicalMatchMode.TERMS

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate request identity and payload-bearing values."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        if not self.text.strip():
            raise ValueError("text must not be blank.")
        if type(self.limit) is not int:
            raise TypeError("limit must be an integer.")
        if self.limit <= 0:
            raise ValueError("limit must be positive.")
        if self.corpus_version is not None and not isinstance(
            self.corpus_version,
            CorpusVersionId,
        ):
            raise TypeError("corpus_version must be a CorpusVersionId or None.")
        if self.filter is not None and not isinstance(self.filter, _FILTER_TYPES):
            raise TypeError("filter must be a filter expression or None.")
        if not isinstance(self.mode, LexicalMatchMode):
            raise TypeError("mode must be a LexicalMatchMode.")

    def retrieval_query(self, corpus_version: CorpusVersionId) -> RetrievalQuery:
        """Build the RET-01 query after serving selects its version."""
        if not isinstance(corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        return RetrievalQuery(
            self.text,
            corpus_version,
            RetrievalLimit(self.limit),
            filter=self.filter,
        )

    def lexical_query(self, corpus_version: CorpusVersionId) -> LexicalQuery:
        """Build the RET-02 request for the selected version."""
        return LexicalQuery(self.retrieval_query(corpus_version), self.mode)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ServingSearchSourceLink:
    """Identify one validated source span linked to a returned passage."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    source_span_id: SourceSpanId

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the source link identity."""
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ServingSearchSourceLocation:
    """A validated clipped range and parser location for one source span."""

    source_span_id: SourceSpanId
    document_id: DocumentId
    document_version_id: DocumentVersionId
    parsed_record_id: RecordId
    start_offset: int
    end_offset: int
    span_start_offset: int
    span_end_offset: int
    content_origin: ContentOrigin
    block_ordinal: int
    block_kind: StructuralBlockKind
    page_number: int | None
    bounding_box: BoundingBox | None
    page_layout: PageLayoutDescriptor | None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate identity, half-open ranges, and optional layout data."""
        for field_name, value, expected_type in (
            ("source_span_id", self.source_span_id, SourceSpanId),
            ("document_id", self.document_id, DocumentId),
            ("document_version_id", self.document_version_id, DocumentVersionId),
            ("parsed_record_id", self.parsed_record_id, RecordId),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{field_name} has an invalid identifier type.")
        for field_name in (
            "start_offset",
            "end_offset",
            "span_start_offset",
            "span_end_offset",
            "block_ordinal",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
        if self.end_offset <= self.start_offset:
            raise ValueError("the clipped source range must be non-empty.")
        if self.span_end_offset <= self.span_start_offset:
            raise ValueError("the source span range must be non-empty.")
        if not (
            self.span_start_offset
            <= self.start_offset
            < self.end_offset
            <= self.span_end_offset
        ):
            raise ValueError("the clipped source range must be inside its span.")
        if not isinstance(self.content_origin, ContentOrigin):
            raise TypeError("content_origin must be a ContentOrigin.")
        if not isinstance(self.block_kind, StructuralBlockKind):
            raise TypeError("block_kind must be a StructuralBlockKind.")
        if self.page_number is not None and (
            type(self.page_number) is not int or self.page_number < 1
        ):
            raise ValueError("page_number must be a positive integer or None.")
        if self.bounding_box is not None and not isinstance(
            self.bounding_box,
            BoundingBox,
        ):
            raise TypeError("bounding_box must be a BoundingBox or None.")
        if self.page_layout is not None and not isinstance(
            self.page_layout,
            PageLayoutDescriptor,
        ):
            raise TypeError("page_layout must be a PageLayoutDescriptor or None.")


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ServingSearchHit:
    """One ordered original-passage result returned by serving."""

    record_id: RecordId
    document_id: DocumentId
    document_version_id: DocumentVersionId
    rank: int
    score: int | float
    score_label: str
    text: str
    source_links: tuple[ServingSearchSourceLink, ...]
    source_locations: tuple[ServingSearchSourceLocation, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the search hit identity and payload-bearing values."""
        if not isinstance(self.record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer.")
        if isinstance(self.score, bool) or type(self.score) not in (int, float):
            raise TypeError("score must be a finite number.")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite.")
        if not isinstance(self.score_label, str) or not self.score_label.strip():
            raise ValueError("score_label must be non-empty.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must not be blank.")
        if not isinstance(self.source_links, tuple) or not self.source_links:
            raise ValueError("source_links must be a non-empty tuple.")
        if any(
            not isinstance(source, ServingSearchSourceLink)
            for source in self.source_links
        ):
            raise TypeError(
                "source_links must contain only ServingSearchSourceLink values."
            )
        if any(
            source.document_id != self.document_id
            or source.document_version_id != self.document_version_id
            for source in self.source_links
        ):
            raise ValueError("source_links must belong to the returned passage.")
        if not isinstance(self.source_locations, tuple):
            raise TypeError("source_locations must be a tuple.")
        if any(
            not isinstance(location, ServingSearchSourceLocation)
            for location in self.source_locations
        ):
            raise TypeError(
                "source_locations must contain only ServingSearchSourceLocation values."
            )
        if self.source_locations:
            link_ids = tuple(link.source_span_id for link in self.source_links)
            location_ids = tuple(
                location.source_span_id for location in self.source_locations
            )
            if location_ids != link_ids:
                raise ValueError(
                    "source_locations must preserve source-link order and identity."
                )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ServingSearchResult:
    """A successful version-pinned search, including successful emptiness."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    requested_limit: int
    hits: tuple[ServingSearchHit, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the search result identity and payload-bearing values."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if type(self.requested_limit) is not int or self.requested_limit <= 0:
            raise ValueError("requested_limit must be a positive integer.")
        if not isinstance(self.hits, tuple):
            raise TypeError("hits must be a tuple.")
        if any(not isinstance(hit, ServingSearchHit) for hit in self.hits):
            raise TypeError("hits must contain only ServingSearchHit values.")
        if tuple(hit.rank for hit in self.hits) != tuple(range(1, len(self.hits) + 1)):
            raise ValueError("hit ranks must be contiguous and ordered.")
        if len(self.hits) > self.requested_limit:
            raise ValueError("hits must not exceed requested_limit.")
        record_ids = tuple(hit.record_id for hit in self.hits)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("hits must not contain duplicate record IDs.")
