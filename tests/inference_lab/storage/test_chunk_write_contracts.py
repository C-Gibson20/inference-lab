"""Tests for the typed chunk-set write boundary."""

import pytest

from inference_lab.ingest import (
    ChunkingRequest,
    FakeChunkTokeniser,
    build_chunk_set,
    map_chunk_set,
)
from inference_lab.provenance import CorpusId
from inference_lab.storage import (
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    WriteChunkSetOutcome,
)
from tests.inference_lab.ingest.fixtures import (  # noqa: TID252
    fixture_parsed_document,
)


def _selection() -> ChunkSetSelectionRecord:
    """Build one valid storage selection through the public mapper."""
    chunk_set = build_chunk_set(
        ChunkingRequest(
            corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000302"),
            parsed_document=fixture_parsed_document(),
            tokeniser=FakeChunkTokeniser(),
        )
    )
    return map_chunk_set(chunk_set)


def test_chunk_set_write_request_requires_typed_target_and_selection() -> None:
    selection = _selection()

    request = ChunkSetWriteRequest(
        corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000303"),
        selection=selection,
    )

    assert request.selection is selection
    with pytest.raises(TypeError, match="corpus_id"):
        ChunkSetWriteRequest(corpus_id=object(), selection=selection)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="selection"):
        ChunkSetWriteRequest(corpus_id=request.corpus_id, selection=object())  # type: ignore[arg-type]


def test_write_chunk_set_outcome_requires_strict_replay_boolean() -> None:
    selection = _selection()

    assert WriteChunkSetOutcome(selection=selection).replayed is False
    assert WriteChunkSetOutcome(selection=selection, replayed=True).replayed is True
    with pytest.raises(TypeError, match="replayed"):
        WriteChunkSetOutcome(selection=selection, replayed=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="selection"):
        WriteChunkSetOutcome(selection=object())  # type: ignore[arg-type]
