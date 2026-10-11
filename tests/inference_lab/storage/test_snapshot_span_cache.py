"""Bounded-memory coverage for the snapshot source-location cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_lab.provenance import CorpusId
from inference_lab.retrieval import open_lexical_generation
from inference_lab.storage import open_metadata_repository
from inference_lab.storage._sqlite_corpus_snapshots import (
    DEFAULT_SOURCE_SPAN_CACHE_ENTRIES,
    SQLiteCorpusSnapshot,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS
from tests.inference_lab.retrieval.pipeline_fixtures import (
    PipelineEmbedder,
    publish_corpora,
)

_CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-0000000004a1")


def _snapshot(cache_entries: object) -> SQLiteCorpusSnapshot:
    """Build an unopened snapshot owner with an explicit cache bound."""
    return SQLiteCorpusSnapshot(
        Path("metadata.sqlite3"),
        5_000,
        MIGRATIONS,
        None,
        _CORPUS_ID,
        None,
        source_span_cache_entries=cache_entries,  # type: ignore[arg-type]
    )


def test_the_source_span_cache_bound_is_validated_before_any_read() -> None:
    assert (
        _snapshot(DEFAULT_SOURCE_SPAN_CACHE_ENTRIES).source_span_cache_entries
        == DEFAULT_SOURCE_SPAN_CACHE_ENTRIES
    )
    assert _snapshot(1).source_span_cache_entry_count == 0
    with pytest.raises(ValueError):
        _snapshot(0)
    with pytest.raises(ValueError):
        _snapshot(-1)
    with pytest.raises(TypeError):
        _snapshot(True)
    with pytest.raises(TypeError):
        _snapshot("32")


def test_resolving_spans_across_documents_never_exceeds_the_bound(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "index"
    published = publish_corpora(
        database_path, source_dir, index_dir, PipelineEmbedder()
    )
    corpus_id = next(iter(published))
    # One cached index at a time, so every new document must evict the last.
    repository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        source_span_cache_entries=1,
    )
    try:
        with repository.open_active_corpus_snapshot(corpus_id) as snapshot:
            generation_id = next(
                reference.generation_id
                for reference in snapshot.manifest.component_generations
                if reference.component_kind.as_str() == "lexical-index"
            )
            with open_lexical_generation(index_dir, generation_id) as generation:
                span_ids = [
                    link.source_span_id
                    for chunk in generation.read_chunks()
                    for link in chunk.source_links
                ]

            assert len(span_ids) > 1

            for source_span_id in span_ids:
                assert snapshot.resolve_source_span(source_span_id) is not None
                assert snapshot.source_span_cache_entry_count == 1
    finally:
        repository.close()
