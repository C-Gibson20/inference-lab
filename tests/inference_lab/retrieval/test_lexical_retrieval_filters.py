"""Tests for pre-limit metadata filtering of ranked lexical matches."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import inference_lab.retrieval.lexical_retrieval as lexical_retrieval
from inference_lab.profiling import (
    FailureEvent,
    ProfilingContext,
    ProfilingSink,
    StageTiming,
)
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    ExactFilter,
    InvalidMetadataFilterError,
    LexicalIndexedChunk,
    LexicalMatchMode,
    LexicalQuery,
    LexicalSourceLink,
    RetrievalLimit,
    RetrievalQuery,
    validate_metadata_filter,
)
from inference_lab.retrieval.lexical_store import LexicalGeneration
from inference_lab.retrieval.results import ResultStatus
from inference_lab.storage import (
    ComponentGenerationState,
    CorpusSnapshot,
    CorruptStorageError,
    MetadataRepository,
    RepositoryUnavailableError,
)

CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
CORPUS_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002")


class _RankedGeneration:
    """Expose a bounded ranked stream while recording the request and scan."""

    def __init__(
        self,
        matches: tuple[tuple[LexicalIndexedChunk, int | float], ...],
    ) -> None:
        self.matches = matches
        self.request: LexicalQuery | None = None
        self.visited = 0

    def iter_ranked_matches(self, request: LexicalQuery):
        self.request = request
        for match in self.matches:
            self.visited += 1
            yield match


class _SnapshotContext:
    """Minimal pinned snapshot projection for orchestration tests."""

    def __init__(self, generation_id: str) -> None:
        self.version = SimpleNamespace(corpus_version=CORPUS_VERSION_ID)
        reference = SimpleNamespace(
            component_kind=SimpleNamespace(as_str=lambda: "lexical-index"),
            state=ComponentGenerationState.READY,
            corpus_id=CORPUS_ID,
            corpus_version=CORPUS_VERSION_ID,
            generation_id=generation_id,
            implementation="sqlite-fts5",
            format_version="1.0.0",
            contract_version="1.0.0",
            configuration_digest="configuration-digest",
            source_manifest_digest="source-manifest-digest",
        )
        self.manifest = SimpleNamespace(component_generations=(reference,))
        self.closed = False

    def __enter__(self) -> "_SnapshotContext":
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def get_source_artefact(self, *_: object) -> None:
        return None

    def get_parsed_representation(self, *_: object) -> None:
        return None

    def resolve_source_span(self, *_: object) -> None:
        return None


class _GenerationContext(_RankedGeneration):
    """Context-managed ranked generation projection for orchestration tests."""

    def __init__(
        self,
        matches: tuple[tuple[LexicalIndexedChunk, int | float], ...],
        generation_id: str,
    ) -> None:
        super().__init__(matches)
        self.closed = False
        self.manifest = SimpleNamespace(
            corpus_id=CORPUS_ID,
            corpus_version=CORPUS_VERSION_ID,
            generation_id=generation_id,
            implementation="sqlite-fts5",
            format_version="1.0.0",
            contract_version="1.0.0",
            configuration_digest="configuration-digest",
            source_manifest_digest="source-manifest-digest",
            configuration=SimpleNamespace(
                score_label="sqlite_fts5_negated_bm25_v1",
                identifier_score_label="sqlite_exact_identifier_occurrences_v1",
            ),
        )

    def __enter__(self) -> "_GenerationContext":
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True
        return None

    def search(self, request: LexicalQuery):
        return self.matches[: request.query.limit.value]


class _Repository:
    """Repository projection that owns one snapshot context."""

    def __init__(self, snapshot: _SnapshotContext) -> None:
        self.snapshot = snapshot

    def open_corpus_snapshot(self, *_: object) -> _SnapshotContext:
        return self.snapshot


class _CollectingSink:
    """Collect profiling records in emission order for orchestration tests."""

    def __init__(self) -> None:
        self.records: list[object] = []

    def emit(self, record: object) -> None:
        self.records.append(record)


def _profiling(sink: object, *, run_id: str = "run-filter-profile") -> ProfilingContext:
    """Build a deterministic profiling context for one orchestration test."""
    return ProfilingContext.create(
        run_id=run_id,
        sink=cast(ProfilingSink, sink),
        clock=iter(range(0, 100_000_000, 10_000_000)).__next__,
        observed_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
    )


def _metadata_completions(sink: _CollectingSink) -> list[StageTiming]:
    """Return terminal timing records for the metadata stage."""
    return [
        record
        for record in sink.records
        if isinstance(record, StageTiming)
        and record.event == "stage_completed"
        and record.pipeline_stage == "metadata_filter"
    ]


def _metadata_failures(sink: _CollectingSink) -> list[FailureEvent]:
    """Return failure records emitted by the metadata stage."""
    return [
        record
        for record in sink.records
        if isinstance(record, FailureEvent)
        and record.pipeline_stage == "metadata_filter"
    ]


def _candidate(ordinal: int) -> LexicalIndexedChunk:
    document_id = DocumentId.from_str(f"10000000-0000-4000-8000-{ordinal:012d}")
    document_version_id = DocumentVersionId.from_str(
        f"20000000-0000-4000-8000-{ordinal:012d}"
    )
    source_span_id = SourceSpanId.from_str(f"30000000-0000-4000-8000-{ordinal:012d}")
    return LexicalIndexedChunk(
        record_id=RecordId.from_str(f"40000000-0000-4000-8000-{ordinal:012d}"),
        document_id=document_id,
        document_version_id=document_version_id,
        ordinal=0,
        content=f"candidate {ordinal}",
        token_count=2,
        source_start=0,
        source_end=10,
        source_links=(
            LexicalSourceLink(
                source_span_id,
                document_version_id,
                0,
                "paragraph",
                0,
                10,
            ),
        ),
    )


def _query(
    *,
    text: str = "common",
    limit: int = 2,
    filter_expression: ExactFilter | None = ExactFilter("language", "en"),
) -> RetrievalQuery:
    return RetrievalQuery(
        text,
        CORPUS_VERSION_ID,
        RetrievalLimit(limit),
        filter=filter_expression,
    )


def test_filtering_scans_lower_ranked_matches_before_final_limit(monkeypatch) -> None:
    candidates = tuple(_candidate(ordinal) for ordinal in range(1, 5))
    generation = _RankedGeneration(
        tuple((candidate, float(index)) for index, candidate in enumerate(candidates))
    )
    metadata = {
        candidate.record_id: {"language": "fr" if index < 2 else "en"}
        for index, candidate in enumerate(candidates)
    }

    monkeypatch.setattr(
        lexical_retrieval,
        "resolve_lexical_candidate_metadata",
        lambda _snapshot, candidate: SimpleNamespace(
            values=metadata[candidate.record_id]
        ),
    )
    query = _query()
    lexical_request = LexicalQuery(
        RetrievalQuery("common", CORPUS_VERSION_ID, RetrievalLimit(2))
    )

    matches = lexical_retrieval._filter_ranked_matches(
        cast(LexicalGeneration, generation),
        lexical_request,
        query,
        validate_metadata_filter(cast(ExactFilter, query.filter)),
        cast(CorpusSnapshot, object()),
    )

    assert tuple(match[0].record_id for match in matches) == tuple(
        candidate.record_id for candidate in candidates[2:]
    )
    assert generation.visited == 4


def test_execute_filtered_retrieval_uses_pinned_generation_and_preserves_hits(
    monkeypatch,
) -> None:
    candidates = tuple(_candidate(ordinal) for ordinal in range(1, 5))
    generation = _GenerationContext(
        tuple((candidate, float(index)) for index, candidate in enumerate(candidates)),
        "generation-1",
    )
    snapshot = _SnapshotContext("generation-1")
    repository = _Repository(snapshot)
    metadata = {
        candidate.record_id: {"language": "fr" if index < 2 else "en"}
        for index, candidate in enumerate(candidates)
    }
    monkeypatch.setattr(
        lexical_retrieval,
        "open_lexical_generation",
        lambda *_: generation,
    )
    monkeypatch.setattr(
        lexical_retrieval,
        "resolve_lexical_candidate_metadata",
        lambda _snapshot, candidate: SimpleNamespace(
            values=metadata[candidate.record_id]
        ),
    )
    query = _query()
    lexical_request = LexicalQuery(
        RetrievalQuery("common", CORPUS_VERSION_ID, RetrievalLimit(2))
    )

    result = lexical_retrieval._execute_lexical_retrieval(
        cast(MetadataRepository, repository),
        Path("indexes"),
        CORPUS_ID,
        query,
        lexical_request,
    )

    assert result.status is ResultStatus.COMPLETED
    assert tuple(hit.candidate.record_id for hit in result.hits) == tuple(
        candidate.record_id for candidate in candidates[2:]
    )
    assert tuple(hit.rank for hit in result.hits) == (1, 2)
    assert tuple(hit.contributions[0].raw_score for hit in result.hits) == (2.0, 3.0)
    assert result.hits[0].contributions[0].score_label == (
        "sqlite_fts5_negated_bm25_v1"
    )
    assert generation.visited == 4
    assert snapshot.closed is True


def test_profiled_unavailable_dependency_returns_safe_failed_result() -> None:
    query = _query(
        text="private-query-marker",
        filter_expression=ExactFilter("language", "private-filter-marker"),
    )

    class _UnavailableRepository:
        def open_corpus_snapshot(self, *_: object) -> None:
            raise RepositoryUnavailableError()

    sink = _CollectingSink()
    result = lexical_retrieval.execute_lexical_retrieval(
        cast(MetadataRepository, _UnavailableRepository()),
        Path("private-index-path"),
        CORPUS_ID,
        query,
        profiling=_profiling(sink),
    )

    assert result.status is ResultStatus.FAILED
    assert result.hits == ()
    assert _metadata_completions(sink)[0].status == "failed"
    assert _metadata_completions(sink)[0].items_processed is None
    assert _metadata_completions(sink)[0].records_emitted is None
    failures = _metadata_failures(sink)
    assert len(failures) == 1
    assert failures[0].failure_category == "dependency"
    assert failures[0].failure_code == "metadata_dependency_unavailable"
    assert "private-query-marker" not in repr(sink.records)
    assert "private-filter-marker" not in repr(sink.records)
    assert "private-index-path" not in repr(sink.records)


def test_profiled_corrupt_metadata_is_internal_failure_without_payloads(
    monkeypatch,
) -> None:
    candidate = _candidate(1)
    generation = _GenerationContext(((candidate, 1.0),), "generation-1")
    snapshot = _SnapshotContext("generation-1")
    raw_error_marker = "private-corruption-details"
    monkeypatch.setattr(
        lexical_retrieval, "open_lexical_generation", lambda *_: generation
    )

    def raise_corruption(_snapshot: object, _candidate: object) -> object:
        raise CorruptStorageError()

    monkeypatch.setattr(
        lexical_retrieval, "resolve_lexical_candidate_metadata", raise_corruption
    )
    sink = _CollectingSink()

    result = lexical_retrieval.execute_lexical_retrieval(
        cast(MetadataRepository, _Repository(snapshot)),
        Path("private-index-path"),
        CORPUS_ID,
        _query(text=raw_error_marker),
        profiling=_profiling(sink),
    )

    assert result.status is ResultStatus.FAILED
    assert _metadata_completions(sink)[0].status == "failed"
    failure = _metadata_failures(sink)[0]
    assert failure.failure_category == "internal"
    assert failure.failure_code == "metadata_filter_failed"
    assert raw_error_marker not in repr(sink.records)
    assert "private-index-path" not in repr(sink.records)


def test_execute_filtered_retrieval_returns_successful_empty_result(
    monkeypatch,
) -> None:
    candidate = _candidate(1)
    generation = _GenerationContext(((candidate, 1.0),), "generation-1")
    snapshot = _SnapshotContext("generation-1")
    monkeypatch.setattr(
        lexical_retrieval,
        "open_lexical_generation",
        lambda *_: generation,
    )
    monkeypatch.setattr(
        lexical_retrieval,
        "resolve_lexical_candidate_metadata",
        lambda _snapshot, _candidate: SimpleNamespace(values={"language": "fr"}),
    )
    query = _query(limit=1)

    result = lexical_retrieval._execute_lexical_retrieval(
        cast(MetadataRepository, _Repository(snapshot)),
        Path("indexes"),
        CORPUS_ID,
        query,
        LexicalQuery(RetrievalQuery("common", CORPUS_VERSION_ID, RetrievalLimit(1))),
    )

    assert result.status is ResultStatus.COMPLETED
    assert result.hits == ()
    assert result.failures == ()
    assert snapshot.closed is True


def test_filtered_identifier_requests_keep_identifier_mode(monkeypatch) -> None:
    candidate = _candidate(1)
    generation = _RankedGeneration(((candidate, 3),))
    query = _query(text="10.1234/ABC-7", limit=1)
    lexical_request = LexicalQuery(
        RetrievalQuery("10.1234/ABC-7", CORPUS_VERSION_ID, RetrievalLimit(1)),
        LexicalMatchMode.IDENTIFIER,
    )
    monkeypatch.setattr(
        lexical_retrieval,
        "resolve_lexical_candidate_metadata",
        lambda _snapshot, _candidate: SimpleNamespace(values={"language": "en"}),
    )

    matches = lexical_retrieval._filter_ranked_matches(
        cast(LexicalGeneration, generation),
        lexical_request,
        query,
        validate_metadata_filter(cast(ExactFilter, query.filter)),
        cast(CorpusSnapshot, object()),
    )

    assert generation.request is lexical_request
    assert generation.request is not None
    assert generation.request.identifier == "10.1234/ABC-7"
    assert matches == ((candidate, 3),)


def test_filtered_hits_rebuild_contiguous_ranks_and_preserve_provenance() -> None:
    first = _candidate(1)
    second = _candidate(2)
    lexical_request = LexicalQuery(
        RetrievalQuery("common", CORPUS_VERSION_ID, RetrievalLimit(2))
    )

    hits = lexical_retrieval._build_hits(
        lexical_request,
        ((first, 0.25), (second, 0.5)),
    )

    assert tuple(hit.rank for hit in hits) == (1, 2)
    assert tuple(hit.candidate.record_id for hit in hits) == (
        first.record_id,
        second.record_id,
    )
    assert hits[0].candidate.kind is CandidateKind.ORIGINAL_PASSAGE
    assert (
        hits[0].candidate.sources[0].source_span_id
        == first.source_links[0].source_span_id
    )
    assert hits[0].contributions[0].raw_score == 0.25


def test_invalid_metadata_filter_is_rejected_before_storage_execution() -> None:
    query = _query(filter_expression=ExactFilter("assignee", "private"))

    class _Repository:
        def open_corpus_snapshot(self, *_: object) -> None:
            raise AssertionError("storage must not be opened for an invalid filter")

    with pytest.raises(InvalidMetadataFilterError):
        lexical_retrieval._execute_lexical_retrieval(
            cast(MetadataRepository, _Repository()),
            Path("indexes"),
            CORPUS_ID,
            query,
            LexicalQuery(
                RetrievalQuery("common", CORPUS_VERSION_ID, RetrievalLimit(2))
            ),
        )
