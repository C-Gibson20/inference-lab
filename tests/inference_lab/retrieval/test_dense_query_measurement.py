"""Measurement coverage proving dense query work does not grow with the index."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from inference_lab.provenance import DocumentId, DocumentVersionId
from inference_lab.retrieval import (
    DenseIndexEntry,
    DenseVector,
    dense_retrieval,
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


class _RecordingGeneration:
    """Dense generation stub recording the ranking limits it is asked for."""

    def __init__(self, entry_count: int) -> None:
        self.requested_limits: list[int] = []
        self.entry_count = entry_count

    def search(self, query_vector: DenseVector, *, limit: int) -> tuple[int, ...]:
        self.requested_limits.append(limit)
        return tuple(range(min(limit, self.entry_count)))


def _hits_for_surviving(surviving: int):
    """Build a hit assembler where only the deepest candidates pass a filter."""

    def build_hits(
        query: object,
        snapshot: object,
        matches: tuple[int, ...],
        validation: object,
    ) -> tuple[object, ...]:
        passing = [match for match in matches if match >= surviving]
        return tuple(passing[:_QUERY_LIMIT])

    return build_hits


def _window_limits(
    entry_count: int,
    surviving: int,
    validation: object,
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    """Return every ranking limit the executor asked the generation for."""
    generation = _RecordingGeneration(entry_count)
    monkeypatch.setattr(dense_retrieval, "_build_hits", _hits_for_surviving(surviving))
    dense_retrieval._ranked_hits_within_budget(
        cast(Any, generation),
        cast(Any, SimpleNamespace(limit=SimpleNamespace(value=_QUERY_LIMIT))),
        cast(Any, None),
        DenseVector((1.0, 0.0, 0.0)),
        cast(Any, validation),
        entry_count=entry_count,
    )
    return generation.requested_limits


def test_unfiltered_queries_rank_only_the_result_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = _window_limits(1_000, 0, None, monkeypatch)

    assert limits == [_QUERY_LIMIT]


def test_filtered_queries_rank_a_bounded_window_rather_than_the_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = _window_limits(1_000, 0, object(), monkeypatch)

    # Every candidate survives, so the first bounded window is sufficient and
    # the whole generation is never ranked.
    assert limits == [_QUERY_LIMIT * dense_retrieval.FILTERED_WINDOW_FACTOR]
    assert max(limits) < 1_000


def test_a_selective_filter_tops_the_window_up_without_overshooting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only the single deepest candidate survives, so the window must grow
    # all the way to the generation size.
    limits = _window_limits(100, 99, object(), monkeypatch)

    # The window doubles until enough candidates survive, and never exceeds
    # the generation it is ranking.
    assert limits[0] == _QUERY_LIMIT * dense_retrieval.FILTERED_WINDOW_FACTOR
    assert limits == sorted(limits)
    assert limits[-1] == 100
    assert len(limits) > 1
