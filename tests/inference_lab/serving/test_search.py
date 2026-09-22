"""Tests for the corpus-scoped serving search facade core."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from inference_lab.profiling import ProfilingContext, ProfilingSink, StageTiming
from inference_lab.provenance import (
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    ExactFilter,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
)
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.serving.search import ServingSearchRunner
from inference_lab.serving.search_contracts import (
    ServingSearchError,
    ServingSearchFailureCategory,
    ServingSearchRequest,
    ServingSearchSourceLocation,
)
from inference_lab.settings import (
    ModelPaths,
    RuntimeConfig,
    Settings,
    StorageConfig,
)
from inference_lab.storage import (
    ComponentGenerationState,
    ContentOrigin,
    MetadataRepository,
    StructuralBlockKind,
)

CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
OTHER_CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000002")
CORPUS_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000003")
OTHER_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000004")
DOCUMENT_ID = DocumentId.from_str("10000000-0000-4000-8000-000000000001")
DOCUMENT_VERSION_ID = DocumentVersionId.from_str("20000000-0000-4000-8000-000000000001")
RECORD_ID = RecordId.from_str("30000000-0000-4000-8000-000000000001")
SOURCE_SPAN_ID = SourceSpanId.from_str("40000000-0000-4000-8000-000000000001")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        models=ModelPaths(
            embedder_path=tmp_path / "models" / "embedder",
            reranker_path=tmp_path / "models" / "reranker",
            generator_path=tmp_path / "models" / "generator",
        ),
        runtime=RuntimeConfig(vram_budget_gb=24),
        storage=StorageConfig(
            index_dir=tmp_path / "index",
            metadata_db_path=tmp_path / "metadata.sqlite3",
            source_dir=tmp_path / "source-data",
        ),
    )


class _Snapshot:
    def __init__(self, corpus_version: CorpusVersionId) -> None:
        self.corpus_id = CORPUS_ID
        self.version = SimpleNamespace(corpus_version=corpus_version)
        self.manifest = SimpleNamespace(
            component_generations=(
                SimpleNamespace(
                    component_kind=SimpleNamespace(as_str=lambda: "lexical-index"),
                    state=ComponentGenerationState.READY,
                    corpus_id=CORPUS_ID,
                    corpus_version=corpus_version,
                    generation_id="generation-1",
                ),
            )
        )
        self.closed = False

    def __enter__(self) -> "_Snapshot":
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def get_document_version(self, document_version_id: DocumentVersionId):
        if document_version_id != DOCUMENT_VERSION_ID:
            return None
        return SimpleNamespace(document_id=DOCUMENT_ID)


class _Repository:
    def __init__(
        self,
        *,
        active: _Snapshot | None,
        corpora: tuple[CorpusMetadata, ...] = (),
        versions: dict[tuple[CorpusId, CorpusVersionId], object] | None = None,
    ) -> None:
        self.active = active
        self.corpora = corpora
        self.versions = {} if versions is None else versions
        self.open_active_count = 0
        self.open_historical_count = 0

    def get_corpus(self, corpus_id: CorpusId):
        return next(
            (corpus for corpus in self.corpora if corpus.corpus_id == corpus_id),
            None,
        )

    def list_corpora(self) -> tuple[CorpusMetadata, ...]:
        return self.corpora

    def get_corpus_version(self, corpus_id: CorpusId, corpus_version: CorpusVersionId):
        return self.versions.get((corpus_id, corpus_version))

    def open_active_corpus_snapshot(self, corpus_id: CorpusId) -> _Snapshot:
        self.open_active_count += 1
        if self.active is None:
            from inference_lab.storage import StorageConflictError

            raise StorageConflictError()
        return self.active

    def open_corpus_snapshot(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> _Snapshot:
        self.open_historical_count += 1
        return cast(_Snapshot, self.active)


class _CollectingSink:
    def __init__(self) -> None:
        self.records: list[object] = []

    def emit(self, record: object) -> None:
        self.records.append(record)


class _Generation:
    def __init__(self) -> None:
        self.closed = False

    def __enter__(self) -> "_Generation":
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True


def _stub_generation(monkeypatch) -> _Generation:
    generation = _Generation()
    monkeypatch.setattr(
        "inference_lab.serving.search.open_lexical_generation",
        lambda *_args: generation,
    )
    return generation


def _request(**kwargs: object) -> ServingSearchRequest:
    values = {
        "corpus_id": CORPUS_ID,
        "text": "alpha",
        "limit": 5,
    }
    values.update(kwargs)
    return ServingSearchRequest(**values)  # type: ignore[arg-type]


def _empty_result(query: RetrievalQuery) -> EvidenceResult:
    return EvidenceResult.from_query(
        query,
        observed_corpus_version=query.corpus_version,
        hits=(),
        status=ResultStatus.COMPLETED,
        failures=(),
    )


def _hit_result(query: RetrievalQuery) -> EvidenceResult:
    source = EvidenceSource(
        DOCUMENT_ID,
        DOCUMENT_VERSION_ID,
        SOURCE_SPAN_ID,
    )
    hit = EvidenceHit(
        candidate=RetrievalCandidate(
            record_id=RECORD_ID,
            kind=CandidateKind.ORIGINAL_PASSAGE,
            content="alpha passage",
            sources=(source,),
        ),
        contributions=(
            ChannelScore(
                RetrievalChannel.LEXICAL,
                rank=1,
                raw_score=1.5,
                score_label="sqlite_fts5_negated_bm25_v1",
            ),
        ),
        rank=1,
    )
    return EvidenceResult.from_query(
        query,
        observed_corpus_version=query.corpus_version,
        hits=(hit,),
        status=ResultStatus.COMPLETED,
        failures=(),
    )


def test_active_search_returns_resolved_version_and_successful_empty_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot = _Snapshot(CORPUS_VERSION_ID)
    repository = _Repository(
        active=snapshot,
        corpora=(CorpusMetadata(CORPUS_ID, "Corpus A"),),
    )
    generation = _stub_generation(monkeypatch)

    def execute(
        _repository,
        _index_dir,
        _corpus_id,
        retrieval_query,
        *,
        mode,
        profiling,
        parent_stage_id,
        snapshot,
        generation,
    ):
        assert snapshot is repository.active
        return _empty_result(retrieval_query)

    monkeypatch.setattr(
        "inference_lab.serving.search.execute_lexical_retrieval", execute
    )
    result = ServingSearchRunner(
        cast(MetadataRepository, repository),
        _settings(tmp_path),
    ).search(_request())

    assert result.corpus_id == CORPUS_ID
    assert result.corpus_version == CORPUS_VERSION_ID
    assert result.hits == ()
    assert repository.open_active_count == 1
    assert snapshot.closed is True
    assert generation.closed is True


def test_search_converts_valid_original_passage_hit(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = _Snapshot(CORPUS_VERSION_ID)
    repository = _Repository(
        active=snapshot,
        corpora=(CorpusMetadata(CORPUS_ID, "Corpus A"),),
    )
    generation = _stub_generation(monkeypatch)

    def execute(
        _repository,
        _index_dir,
        _corpus_id,
        retrieval_query,
        *,
        mode,
        profiling,
        parent_stage_id,
        snapshot,
        generation,
    ):
        return _hit_result(retrieval_query)

    monkeypatch.setattr(
        "inference_lab.serving.search.execute_lexical_retrieval", execute
    )
    monkeypatch.setattr(
        "inference_lab.serving.search.resolve_source_locations",
        lambda *_args: (
            ServingSearchSourceLocation(
                source_span_id=SOURCE_SPAN_ID,
                document_id=DOCUMENT_ID,
                document_version_id=DOCUMENT_VERSION_ID,
                parsed_record_id=RecordId.from_str(
                    "30000000-0000-4000-8000-000000000002"
                ),
                start_offset=0,
                end_offset=13,
                span_start_offset=0,
                span_end_offset=13,
                content_origin=ContentOrigin.NATIVE,
                block_ordinal=0,
                block_kind=StructuralBlockKind.PARAGRAPH,
                page_number=None,
                bounding_box=None,
                page_layout=None,
            ),
        ),
    )
    result = ServingSearchRunner(
        cast(MetadataRepository, repository),
        _settings(tmp_path),
    ).search(_request())

    assert result.hits[0].record_id == RECORD_ID
    assert result.hits[0].document_id == DOCUMENT_ID
    assert result.hits[0].document_version_id == DOCUMENT_VERSION_ID
    assert result.hits[0].score == 1.5
    assert result.hits[0].source_links[0].source_span_id == SOURCE_SPAN_ID
    assert generation.closed is True


def test_search_preserves_fail_closed_location_error(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = _Snapshot(CORPUS_VERSION_ID)
    repository = _Repository(
        active=snapshot,
        corpora=(CorpusMetadata(CORPUS_ID, "Corpus A"),),
    )
    generation = _stub_generation(monkeypatch)
    monkeypatch.setattr(
        "inference_lab.serving.search.execute_lexical_retrieval",
        lambda _repository, _index_dir, _corpus_id, retrieval_query, **_kwargs: (
            _hit_result(retrieval_query)
        ),
    )

    def corrupt_location(*_args: object) -> None:
        raise ServingSearchError(ServingSearchFailureCategory.CORRUPT_STATE)

    monkeypatch.setattr(
        "inference_lab.serving.search.resolve_source_locations", corrupt_location
    )
    with pytest.raises(ServingSearchError) as error:
        ServingSearchRunner(
            cast(MetadataRepository, repository), _settings(tmp_path)
        ).search(_request())
    assert error.value.category is ServingSearchFailureCategory.CORRUPT_STATE
    assert generation.closed is True


def test_search_distinguishes_missing_corpus_and_unpublished_corpus(
    tmp_path: Path,
) -> None:
    missing = _Repository(active=None)
    with pytest.raises(ServingSearchError) as missing_error:
        ServingSearchRunner(
            cast(MetadataRepository, missing),
            _settings(tmp_path),
        ).search(_request())
    assert missing_error.value.category is ServingSearchFailureCategory.CORPUS_NOT_FOUND

    unpublished = _Repository(
        active=None,
        corpora=(CorpusMetadata(CORPUS_ID, "Corpus A"),),
    )
    with pytest.raises(ServingSearchError) as unpublished_error:
        ServingSearchRunner(
            cast(MetadataRepository, unpublished),
            _settings(tmp_path),
        ).search(_request())
    assert (
        unpublished_error.value.category
        is ServingSearchFailureCategory.NO_PUBLISHED_VERSION
    )


def test_search_distinguishes_foreign_historical_version(tmp_path: Path) -> None:
    repository = _Repository(
        active=None,
        corpora=(
            CorpusMetadata(CORPUS_ID, "Corpus A"),
            CorpusMetadata(OTHER_CORPUS_ID, "Corpus B"),
        ),
        versions={(OTHER_CORPUS_ID, OTHER_VERSION_ID): object()},
    )
    request = _request(corpus_version=OTHER_VERSION_ID)

    with pytest.raises(ServingSearchError) as error:
        ServingSearchRunner(
            cast(MetadataRepository, repository),
            _settings(tmp_path),
        ).search(request)

    assert error.value.category is ServingSearchFailureCategory.FOREIGN_CORPUS_VERSION
    assert repository.open_historical_count == 0


def test_search_rejects_unsupported_filter_before_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repository = _Repository(
        active=_Snapshot(CORPUS_VERSION_ID),
        corpora=(CorpusMetadata(CORPUS_ID, "Corpus A"),),
    )
    request = _request(filter=ExactFilter("assignee", "private-marker"))
    monkeypatch.setattr(
        "inference_lab.serving.search.execute_lexical_retrieval",
        lambda *_args, **_kwargs: pytest.fail("unsupported filters must not execute"),
    )

    with pytest.raises(ServingSearchError) as error:
        ServingSearchRunner(
            cast(MetadataRepository, repository),
            _settings(tmp_path),
        ).search(request)

    assert error.value.category is ServingSearchFailureCategory.UNSUPPORTED_FILTER
    assert repository.open_active_count == 0


@pytest.mark.parametrize("condition", ["cold", "warm"])
def test_search_profiles_serving_as_parent_of_lexical_retrieval(
    monkeypatch,
    tmp_path: Path,
    condition: str,
) -> None:
    snapshot = _Snapshot(CORPUS_VERSION_ID)
    repository = _Repository(
        active=snapshot,
        corpora=(CorpusMetadata(CORPUS_ID, "Corpus A"),),
    )
    _stub_generation(monkeypatch)
    sink = _CollectingSink()
    profiling = ProfilingContext.create(
        run_id="run-serving-search",
        request_id="request-1",
        condition=condition,  # type: ignore[arg-type]
        sink=cast(ProfilingSink, sink),
        clock=iter(range(0, 100_000_000, 10_000_000)).__next__,
        observed_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
    )

    def execute(
        _repository,
        _index_dir,
        corpus_id,
        retrieval_query,
        *,
        mode,
        profiling,
        parent_stage_id,
        snapshot,
        generation,
    ):
        with profiling.observed_stage(
            stage="lexical_retrieval",
            stage_id=profiling.stage_id("lexical_retrieval", "request"),
            document_id=None,
            corpus_id=corpus_id.as_str(),
            corpus_version=retrieval_query.corpus_version.as_str(),
            parent_stage_id=parent_stage_id,
            module="inference_lab.retrieval",
        ) as stage:
            stage.set_counts(records_emitted=0)
        return _empty_result(retrieval_query)

    monkeypatch.setattr(
        "inference_lab.serving.search.execute_lexical_retrieval", execute
    )
    result = ServingSearchRunner(
        cast(MetadataRepository, repository),
        _settings(tmp_path),
    ).search(_request(), profiling=profiling)

    assert result.hits == ()
    completions = [
        record
        for record in sink.records
        if isinstance(record, StageTiming) and record.event == "stage_completed"
    ]
    assert [record.pipeline_stage for record in completions] == [
        "lexical_retrieval",
        "serving",
    ]
    assert completions[0].parent_stage_id == ("run-serving-search:serving:request")
    assert completions[1].stage_id == "run-serving-search:serving:request"
    assert completions[1].records_emitted == 0
    assert completions[1].condition == condition
    assert all(record.request_id == "request-1" for record in completions)
