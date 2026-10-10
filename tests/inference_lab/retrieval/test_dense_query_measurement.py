"""Measurement coverage proving dense query work does not grow with the index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from inference_lab.provenance import DocumentId, DocumentVersionId
from inference_lab.retrieval import (
    DenseIndexEntry,
    DenseVector,
    dense_store,
    open_dense_generation,
)
from tests.inference_lab.retrieval.test_dense_store import (
    _VERSION_ONE,
    _build,
    _DeterministicEmbedder,
    _source,
)

_QUERY_LIMIT = 3


def _corpus_of(size: int) -> tuple[tuple[str, DocumentId, DocumentVersionId], ...]:
    """Build one distinct single-chunk document per requested corpus member."""
    return tuple(
        (
            f"passage number {index}",
            DocumentId.from_str(f"00000000-0000-4000-8000-{index + 1:012d}"),
            DocumentVersionId.from_str(f"00000000-0000-4000-9000-{index + 1:012d}"),
        )
        for index in range(size)
    )


def _hydrations_per_query(
    tmp_path: Path,
    size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, int]:
    """Return full rows hydrated and matches returned for one bounded query."""
    embedder = _DeterministicEmbedder()
    outcome = _build(tmp_path, _source(_corpus_of(size), _VERSION_ONE), embedder)

    with open_dense_generation(
        tmp_path / "index",
        outcome.manifest.generation_id,
    ) as generation:
        assert generation.manifest.entry_count == size
        hydrated = 0
        original = dense_store._entry_from_row

        def counting_entry_from_row(
            row: sqlite3.Row, dimensions: int
        ) -> DenseIndexEntry:
            nonlocal hydrated
            hydrated += 1
            return original(row, dimensions)

        monkeypatch.setattr(dense_store, "_entry_from_row", counting_entry_from_row)
        matches = generation.search(DenseVector((1.0, 2.0, 3.0)), limit=_QUERY_LIMIT)
        monkeypatch.undo()
    return hydrated, len(matches)


def test_dense_query_hydrates_only_the_requested_number_of_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_hydrated, small_matches = _hydrations_per_query(
        tmp_path / "small", 4, monkeypatch
    )
    large_hydrated, large_matches = _hydrations_per_query(
        tmp_path / "large", 40, monkeypatch
    )

    # A ten-fold larger index must not hydrate a single extra candidate: only
    # the ranked winners carry content, source links, and vectors into memory.
    assert small_matches == large_matches == _QUERY_LIMIT
    assert small_hydrated == large_hydrated == _QUERY_LIMIT
