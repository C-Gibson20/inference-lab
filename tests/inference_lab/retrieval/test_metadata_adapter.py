"""Tests for pinned-snapshot lexical candidate metadata resolution."""

from dataclasses import replace
from typing import cast

import pytest

from inference_lab.provenance import DocumentVersionId, RecordId, SourceSpanId
from inference_lab.retrieval import (
    LexicalCandidateMetadata,
    LexicalIndexedChunk,
    LexicalSourceLink,
    resolve_lexical_candidate_metadata,
)
from inference_lab.storage import (
    CorpusSnapshot,
    CorruptStorageError,
    ParsedRepresentation,
    ResolvedSourceSpan,
    SourceArtefact,
)
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    FIRST_SPAN_ID,
    PARSED_RECORD_ID,
    SECOND_SPAN_ID,
    SOURCE_BYTES,
    SOURCE_HASH,
    SUPERSEDED_RECORD_ID,
    parsed_representation,
)


class _Snapshot:
    """Narrow in-memory snapshot projection used by adapter tests."""

    def __init__(self) -> None:
        representation = parsed_representation()
        self.source_artefact: SourceArtefact | None = SourceArtefact(
            document_id=DOCUMENT_ID,
            document_version_id=DOCUMENT_VERSION_ID,
            content_hash=SOURCE_HASH,
            byte_count=len(SOURCE_BYTES),
            media_type="text/plain",
            storage_key=f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}",
        )
        self.parsed: dict[RecordId, ParsedRepresentation] = {
            representation.record_id: representation
        }
        self.spans = {
            location.source_span_id: ResolvedSourceSpan(
                document_id=DOCUMENT_ID,
                location=location,
                page_layout=None,
            )
            for location in representation.source_locations
        }
        self.requested_parsed_record_ids: list[RecordId] = []

    def get_source_artefact(
        self,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        if document_version_id != DOCUMENT_VERSION_ID:
            return None
        return self.source_artefact

    def get_parsed_representation(
        self,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        self.requested_parsed_record_ids.append(record_id)
        return self.parsed.get(record_id)

    def resolve_source_span(
        self,
        source_span_id: SourceSpanId,
    ) -> ResolvedSourceSpan | None:
        return self.spans.get(source_span_id)


def _resolve(
    snapshot: _Snapshot,
    candidate: LexicalIndexedChunk,
) -> LexicalCandidateMetadata:
    """Call the production adapter through the intentionally narrow fake."""
    return resolve_lexical_candidate_metadata(cast(CorpusSnapshot, snapshot), candidate)


def _candidate(
    *source_links: LexicalSourceLink,
    document_version_id: DocumentVersionId = DOCUMENT_VERSION_ID,
) -> LexicalIndexedChunk:
    return LexicalIndexedChunk(
        record_id=RecordId.from_str("70000000-0000-4000-8000-000000000001"),
        document_id=DOCUMENT_ID,
        document_version_id=document_version_id,
        ordinal=0,
        content="Café",
        token_count=1,
        source_start=0,
        source_end=5,
        source_links=source_links,
    )


def _source_link(
    source_span_id: SourceSpanId = FIRST_SPAN_ID,
    *,
    document_version_id: DocumentVersionId = DOCUMENT_VERSION_ID,
) -> LexicalSourceLink:
    return LexicalSourceLink(
        source_span_id=source_span_id,
        document_version_id=document_version_id,
        block_ordinal=0,
        block_kind="paragraph",
        source_start=0,
        source_end=5,
    )


def test_adapter_resolves_all_phase_five_values_from_one_snapshot() -> None:
    snapshot = _Snapshot()

    resolved = _resolve(
        snapshot,
        _candidate(_source_link()),
    )

    assert isinstance(resolved, LexicalCandidateMetadata)
    assert resolved.source_artefact.media_type == "text/plain"
    assert resolved.source_artefact.byte_count == len(SOURCE_BYTES)
    assert resolved.parsed_representation.record_id == PARSED_RECORD_ID
    assert resolved.source_span.location.source_span_id == FIRST_SPAN_ID
    assert dict(resolved.values) == {
        "character-count": 13,
        "document-type": "text/plain",
        "language": "en",
        "page-count": None,
        "source-byte-count": len(SOURCE_BYTES),
        "token-count": 2,
    }


def test_adapter_uses_exact_span_owning_record_not_another_parser_output() -> None:
    snapshot = _Snapshot()
    superseded = parsed_representation(
        record_id=SUPERSEDED_RECORD_ID,
        supersedes_record_id=PARSED_RECORD_ID,
    )
    snapshot.spans[FIRST_SPAN_ID] = ResolvedSourceSpan(
        document_id=DOCUMENT_ID,
        location=superseded.source_locations[0],
        page_layout=None,
    )
    snapshot.parsed[SUPERSEDED_RECORD_ID] = superseded

    resolved = _resolve(
        snapshot,
        _candidate(_source_link()),
    )

    assert resolved.parsed_representation.record_id == SUPERSEDED_RECORD_ID
    assert snapshot.requested_parsed_record_ids == [SUPERSEDED_RECORD_ID]


def test_adapter_preserves_nullable_missing_values_and_present_zero() -> None:
    snapshot = _Snapshot()
    original = snapshot.parsed[PARSED_RECORD_ID]
    assert original is not None
    snapshot.parsed[PARSED_RECORD_ID] = replace(
        original,
        language=None,
        page_count=0,
        token_count=None,
    )

    resolved = _resolve(
        snapshot,
        _candidate(_source_link()),
    )

    assert resolved.values["language"] is None
    assert resolved.values["page-count"] == 0
    assert resolved.values["token-count"] is None


def test_adapter_accepts_multiple_spans_only_when_they_share_one_record() -> None:
    snapshot = _Snapshot()

    resolved = _resolve(
        snapshot,
        _candidate(_source_link(FIRST_SPAN_ID), _source_link(SECOND_SPAN_ID)),
    )

    assert resolved.parsed_representation.record_id == PARSED_RECORD_ID
    assert resolved.source_span.location.source_span_id == FIRST_SPAN_ID


@pytest.mark.parametrize(
    "candidate",
    (
        _candidate(),
        _candidate(_source_link(document_version_id=DocumentVersionId.generate())),
    ),
)
def test_adapter_rejects_missing_or_foreign_source_links(
    candidate: LexicalIndexedChunk,
) -> None:
    with pytest.raises(CorruptStorageError):
        _resolve(_Snapshot(), candidate)


def test_adapter_rejects_missing_reachable_records() -> None:
    snapshot = _Snapshot()
    snapshot.spans.pop(FIRST_SPAN_ID)

    with pytest.raises(CorruptStorageError):
        _resolve(snapshot, _candidate(_source_link()))

    snapshot = _Snapshot()
    snapshot.source_artefact = None
    with pytest.raises(CorruptStorageError):
        _resolve(snapshot, _candidate(_source_link()))

    snapshot = _Snapshot()
    snapshot.parsed.clear()
    with pytest.raises(CorruptStorageError):
        _resolve(snapshot, _candidate(_source_link()))


def test_adapter_rejects_ambiguous_parsed_records() -> None:
    snapshot = _Snapshot()
    alternate = parsed_representation(record_id=SUPERSEDED_RECORD_ID)
    snapshot.spans[SECOND_SPAN_ID] = ResolvedSourceSpan(
        document_id=DOCUMENT_ID,
        location=alternate.source_locations[1],
        page_layout=None,
    )
    snapshot.parsed[SUPERSEDED_RECORD_ID] = alternate

    with pytest.raises(CorruptStorageError):
        _resolve(
            snapshot,
            _candidate(_source_link(FIRST_SPAN_ID), _source_link(SECOND_SPAN_ID)),
        )
