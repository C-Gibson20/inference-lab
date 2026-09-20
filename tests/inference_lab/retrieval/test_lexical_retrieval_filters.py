"""Tests for pre-limit metadata filtering of ranked lexical matches."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import inference_lab.retrieval.lexical_retrieval as lexical_retrieval
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
    EvidenceResult,
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


class _GenerationContext(_RankedGeneration):
    """Context-managed ranked generation projection for orchestration tests."""

    def __init__(
        self,
        matches: tuple[tuple[LexicalIndexedChunk, int | float], ...],
        generation_id: str,
    ) -> None:
        super().__init__(matches)
        self.manifest = SimpleNamespace(
            corpus_id=CORPUS_ID,
            corpus_version=CORPUS_VERSION_ID,
            generation_id=generation_id,
            implementation="sqlite-fts5",
            format_version="1.0.0",
            contract_version="1.0.0",
            configuration_digest="configuration-digest",
            source_manifest_digest="source-manifest-digest",
        )

    def __enter__(self) -> "_GenerationContext":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _Repository:
    """Repository projection that owns one snapshot context."""

    def __init__(self, snapshot: _SnapshotContext) -> None:
        self.snapshot = snapshot

    def open_corpus_snapshot(self, *_: object) -> _SnapshotContext:
        return self.snapshot


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
    assert generation.visited == 4
    assert snapshot.closed is True


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


def test_execute_filtered_retrieval_fails_closed_on_corrupt_metadata(
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
        lambda _snapshot, _candidate: (_ for _ in ()).throw(CorruptStorageError()),
    )
    query = _query(limit=1)

    result = lexical_retrieval._execute_lexical_retrieval(
        cast(MetadataRepository, _Repository(snapshot)),
        Path("indexes"),
        CORPUS_ID,
        query,
        LexicalQuery(RetrievalQuery("common", CORPUS_VERSION_ID, RetrievalLimit(1))),
    )

    assert result.status is ResultStatus.FAILED
    assert result.hits == ()
    assert result.failures[0].category.value == "channel_unavailable"
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


def test_orchestration_accepts_original_filtered_retrieval_query(monkeypatch) -> None:
    query = _query()
    captured: dict[str, object] = {}

    class _Repository:
        def open_corpus_snapshot(self, *_: object) -> None:
            raise AssertionError(
                "filtered request should be delegated before opening storage"
            )

    def fake_execute(
        _repository: object,
        _index_dir: Path,
        _corpus_id: CorpusId,
        captured_query: RetrievalQuery,
        lexical_request: LexicalQuery,
    ) -> EvidenceResult:
        captured["query"] = captured_query
        captured["lexical_request"] = lexical_request
        return EvidenceResult.from_query(
            captured_query,
            observed_corpus_version=CORPUS_VERSION_ID,
            hits=(),
            status=ResultStatus.COMPLETED,
            failures=(),
        )

    monkeypatch.setattr(lexical_retrieval, "_execute_lexical_retrieval", fake_execute)
    result = lexical_retrieval.execute_lexical_retrieval(
        cast(MetadataRepository, _Repository()),
        Path("indexes"),
        CORPUS_ID,
        query,
    )

    assert result.status.value == "completed"
    assert captured["query"] is query
    assert isinstance(captured["lexical_request"], LexicalQuery)
    assert cast(LexicalQuery, captured["lexical_request"]).query.filter is None


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
