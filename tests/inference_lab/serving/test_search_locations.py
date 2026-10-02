"""Tests for fail-closed source-location reconstruction."""

from types import SimpleNamespace
from typing import cast

import pytest

from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import EvidenceSource, LexicalGeneration
from inference_lab.retrieval.lexical_store_models import (
    LexicalIndexedChunk,
    LexicalSourceLink,
)
from inference_lab.serving.search_contracts import (
    ServingSearchError,
    ServingSearchFailureCategory,
)
from inference_lab.serving.search_locations import resolve_source_locations
from inference_lab.storage import (
    ComponentGenerationState,
    ContentOrigin,
    CorpusSnapshot,
    ResolvedSourceSpan,
    SourceLocation,
    StructuralBlockKind,
)

CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
CORPUS_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002")
DOCUMENT_ID = DocumentId.from_str("10000000-0000-4000-8000-000000000001")
DOCUMENT_VERSION_ID = DocumentVersionId.from_str("20000000-0000-4000-8000-000000000001")
RECORD_ID = RecordId.from_str("30000000-0000-4000-8000-000000000001")
PARSED_RECORD_ID = RecordId.from_str("30000000-0000-4000-8000-000000000002")
SOURCE_SPAN_ID = SourceSpanId.from_str("40000000-0000-4000-8000-000000000001")


class _Generation:
    def __init__(self, chunk: LexicalIndexedChunk) -> None:
        self.manifest = SimpleNamespace(
            corpus_id=CORPUS_ID,
            corpus_version=CORPUS_VERSION_ID,
            generation_id="generation-1",
            implementation="sqlite-fts5",
            format_version="1.0.0",
            contract_version="1.0.0",
            configuration_digest="configuration-digest",
            source_manifest_digest="source-manifest-digest",
        )
        self.chunk = chunk

    def __enter__(self) -> "_Generation":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_chunk(self, record_id: RecordId) -> LexicalIndexedChunk | None:
        return self.chunk if record_id == self.chunk.record_id else None


class _Snapshot:
    corpus_id = CORPUS_ID
    version = SimpleNamespace(corpus_version=CORPUS_VERSION_ID)

    def __init__(self, *, missing_span: bool = False) -> None:
        self.manifest = SimpleNamespace(
            component_generations=(
                SimpleNamespace(
                    component_kind=SimpleNamespace(as_str=lambda: "lexical-index"),
                    state=ComponentGenerationState.READY,
                    corpus_id=CORPUS_ID,
                    corpus_version=CORPUS_VERSION_ID,
                    generation_id="generation-1",
                    implementation="sqlite-fts5",
                    format_version="1.0.0",
                    contract_version="1.0.0",
                    configuration_digest="configuration-digest",
                    source_manifest_digest="source-manifest-digest",
                ),
            )
        )
        self.missing_span = missing_span

    def get_document_version(self, document_version_id: DocumentVersionId):
        return SimpleNamespace(document_id=DOCUMENT_ID)

    def resolve_source_span(self, source_span_id: SourceSpanId):
        if self.missing_span:
            return None
        location = SourceLocation(
            source_span_id=source_span_id,
            document_version_id=DOCUMENT_VERSION_ID,
            parsed_record_id=PARSED_RECORD_ID,
            start_offset=0,
            end_offset=13,
            content_origin=ContentOrigin.NATIVE,
        )
        return ResolvedSourceSpan(
            document_id=DOCUMENT_ID,
            location=location,
            page_layout=None,
        )

    def get_parsed_representation(self, record_id: RecordId):
        return SimpleNamespace(
            record_id=record_id,
            document_id=DOCUMENT_ID,
            document_version_id=DOCUMENT_VERSION_ID,
            source_text="alpha passage",
            blocks=(
                SimpleNamespace(
                    ordinal=0,
                    kind=StructuralBlockKind.PARAGRAPH,
                    source_span_id=SOURCE_SPAN_ID,
                ),
            ),
        )


def _chunk() -> LexicalIndexedChunk:
    return LexicalIndexedChunk(
        record_id=RECORD_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        ordinal=0,
        content="alpha passage",
        token_count=2,
        source_start=0,
        source_end=13,
        source_links=(
            LexicalSourceLink(
                source_span_id=SOURCE_SPAN_ID,
                document_version_id=DOCUMENT_VERSION_ID,
                block_ordinal=0,
                block_kind="paragraph",
                source_start=0,
                source_end=13,
            ),
        ),
    )


def test_resolve_source_locations_returns_clipped_unicode_offsets() -> None:
    generation = _Generation(_chunk())

    locations = resolve_source_locations(
        cast(CorpusSnapshot, _Snapshot()),
        cast(LexicalGeneration, generation),
        RECORD_ID,
        "alpha passage",
        (EvidenceSource(DOCUMENT_ID, DOCUMENT_VERSION_ID, SOURCE_SPAN_ID),),
    )

    assert locations[0].start_offset == 0
    assert locations[0].end_offset == 13
    assert locations[0].span_start_offset == 0
    assert locations[0].span_end_offset == 13
    assert locations[0].block_kind is StructuralBlockKind.PARAGRAPH


def test_resolve_source_locations_fails_closed_when_span_is_unreachable() -> None:
    generation = _Generation(_chunk())

    with pytest.raises(ServingSearchError) as error:
        resolve_source_locations(
            cast(CorpusSnapshot, _Snapshot(missing_span=True)),
            cast(LexicalGeneration, generation),
            RECORD_ID,
            "alpha passage",
            (EvidenceSource(DOCUMENT_ID, DOCUMENT_VERSION_ID, SOURCE_SPAN_ID),),
        )

    assert error.value.category is ServingSearchFailureCategory.CORRUPT_STATE
