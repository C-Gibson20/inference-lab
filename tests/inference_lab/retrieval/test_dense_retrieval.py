"""Pinned-snapshot execution tests for dense retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from inference_lab.backends.runtime import (
    BackendMetadata,
    BackendOperation,
    EmbeddingProfile,
    EmbeddingRequest,
    EmbeddingResult,
)
from inference_lab.ingest import BaselineChunkGenerationManifest
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
)
from inference_lab.retrieval import (
    DenseIndexCompatibility,
    execute_dense_retrieval,
)
from inference_lab.retrieval.dense import DenseIndexEntry
from inference_lab.retrieval.dense_store import (
    build_dense_generation,
    open_dense_generation,
)
from inference_lab.retrieval.filters import ExactFilter, MissingFilter
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.results import FailureCategory, ResultStatus
from inference_lab.retrieval.values import RetrievalLimit
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    MetadataRepository,
)
from tests.inference_lab.retrieval.lexical_fixtures import (
    CORPUS_ID,
    selection,
)
from tests.inference_lab.retrieval.lexical_fixtures import (
    manifest as make_source_manifest,
)

_VERSION = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000201")
_VERSION_OTHER = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000202")
_DOC_IDS = tuple(
    DocumentId.from_str(f"00000000-0000-4000-8000-{value:012d}")
    for value in (211, 212, 213)
)
_DOC_VERSION_IDS = tuple(
    DocumentVersionId.from_str(f"00000000-0000-4000-8000-{value:012d}")
    for value in (221, 222, 223)
)


class _QueryEmbedder:
    """Deterministic fake that records query-purpose requests."""

    def __init__(self) -> None:
        self.metadata = BackendMetadata(
            backend_name="query-fixture",
            model_name="query-model",
            model_revision="r1",
            supported_operations=(BackendOperation.EMBED,),
            capabilities=(),
            embedding_dimensions=3,
        )
        self.profile = EmbeddingProfile(
            profile_version="1.0.0",
            tokenizer_id="query-tokenizer",
            tokenizer_revision="r1",
            query_prefix="query: ",
            document_prefix="",
            pooling_strategy="mean",
            input_normalisation="identity",
            max_input_tokens=256,
            over_limit_policy="truncate_tail",
            l2_normalised=False,
            inference_dtype="float32",
            device_type="cpu",
        )
        self.requests: list[EmbeddingRequest] = []
        self.fail = False
        self.is_ready = True
        self.is_closed = False
        self.output_metadata = self.metadata
        self.output_profile = self.profile

    def load(self) -> None:
        self.is_ready = True

    def close(self) -> None:
        self.is_ready = False
        self.is_closed = True

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("private query embedding marker")
        vectors = tuple(self._vector(text) for text in request.input_texts)
        return EmbeddingResult(vectors, self.output_metadata, self.output_profile)

    @staticmethod
    def _vector(text: str) -> tuple[float, float, float]:
        vectors = {
            "alpha passage": (1.0, 0.0, 0.0),
            "beta passage": (0.8, 0.6, 0.0),
            "gamma passage": (0.0, 1.0, 0.0),
            "nearest": (1.0, 0.0, 0.0),
            "tie": (1.0, 0.0, 0.0),
        }
        if text in vectors:
            return vectors[text]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return float(digest[0] + 1), float(digest[1] + 1), float(digest[2] + 1)


class _Snapshot:
    """Small snapshot projection with explicit version and source visibility."""

    def __init__(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        generation_reference: ComponentGenerationReference,
        entries: tuple[DenseIndexEntry, ...],
    ) -> None:
        self.corpus_id = corpus_id
        self.version = SimpleNamespace(corpus_version=corpus_version)
        self.manifest = SimpleNamespace(component_generations=(generation_reference,))
        self.visible_versions = {
            entry.candidate.sources[0].document_version_id for entry in entries
        }
        self._artefacts: dict[DocumentVersionId, SimpleNamespace] = {}
        self._parsed: dict[RecordId, SimpleNamespace] = {}
        self._spans: dict[object, SimpleNamespace] = {}
        for entry in entries:
            source = entry.candidate.sources[0]
            parsed_record_id = RecordId.from_str(
                f"00000000-0000-4000-8000-{source.document_id.as_str()[-12:]}"
            )
            self._artefacts[source.document_version_id] = SimpleNamespace(
                document_id=source.document_id,
                document_version_id=source.document_version_id,
                media_type="text/plain",
                byte_count=100 + int(source.document_id.as_str()[-2:]),
            )
            self._parsed[parsed_record_id] = SimpleNamespace(
                document_id=source.document_id,
                document_version_id=source.document_version_id,
                record_id=parsed_record_id,
                language={
                    _DOC_IDS[0]: "fr",
                    _DOC_IDS[1]: "en",
                    _DOC_IDS[2]: "en",
                }[source.document_id],
                page_count=None,
                character_count=len(entry.candidate.content),
                token_count=2,
            )
            for item in entry.candidate.sources:
                self._spans[item.source_span_id] = SimpleNamespace(
                    document_id=item.document_id,
                    location=SimpleNamespace(
                        document_version_id=item.document_version_id,
                        source_span_id=item.source_span_id,
                        parsed_record_id=parsed_record_id,
                    ),
                )

    def __enter__(self) -> _Snapshot:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> SimpleNamespace | None:
        if document_version_id not in self.visible_versions:
            return None
        artefact = self._artefacts.get(document_version_id)
        return (
            None
            if artefact is None
            else SimpleNamespace(document_id=artefact.document_id)
        )

    def get_source_artefact(
        self,
        document_version_id: DocumentVersionId,
    ) -> SimpleNamespace | None:
        return self._artefacts.get(document_version_id)

    def get_parsed_representation(self, record_id: RecordId) -> SimpleNamespace | None:
        return self._parsed.get(record_id)

    def resolve_source_span(self, source_span_id: object) -> SimpleNamespace | None:
        return self._spans.get(source_span_id)


class _Repository:
    """Return the one exact version selected for the test request."""

    def __init__(self, snapshot: _Snapshot) -> None:
        self.snapshot = snapshot

    def open_corpus_snapshot(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> _Snapshot:
        del corpus_id, corpus_version
        return self.snapshot


def _prepared_query(
    tmp_path: Path,
    texts: tuple[str, ...] = ("alpha passage", "beta passage", "gamma passage"),
) -> tuple[_Repository, _Snapshot, _QueryEmbedder, Path]:
    embedder = _QueryEmbedder()
    selections = tuple(
        selection(
            text,
            document_id=document_id,
            document_version_id=document_version_id,
        )[1]
        for text, document_id, document_version_id in zip(
            texts,
            _DOC_IDS,
            _DOC_VERSION_IDS,
            strict=True,
        )
    )
    source_manifest = make_source_manifest(selections, corpus_version=_VERSION)
    compatibility = DenseIndexCompatibility(
        model_metadata=embedder.metadata,
        dimensions=3,
        embedding_profile=embedder.profile,
    )
    index_dir = tmp_path / "index"
    outcome = build_dense_generation(
        index_dir,
        source_manifest,
        embedder,
        metadata_db_path=tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "sources",
        compatibility=compatibility,
    )
    reference = replace(
        outcome.manifest.component_generation(),
        state=ComponentGenerationState.READY,
    )
    with open_dense_generation(index_dir, outcome.manifest.generation_id) as opened:
        entries = opened.iter_entries()
    snapshot = _Snapshot(CORPUS_ID, _VERSION, reference, entries)
    return _Repository(snapshot), snapshot, embedder, index_dir


def _execute(
    repository: _Repository,
    index_dir: Path,
    embedder: _QueryEmbedder,
    *,
    text: str = "nearest",
    limit: int = 3,
    filter_expression: ExactFilter | MissingFilter | None = None,
):
    query = RetrievalQuery(
        text,
        _VERSION,
        RetrievalLimit(limit),
        filter=filter_expression,
    )
    return execute_dense_retrieval(
        cast(MetadataRepository, repository),
        index_dir,
        CORPUS_ID,
        query,
        embedder,
    )


def test_dense_query_returns_exact_neighbours_with_stable_ties(tmp_path: Path) -> None:
    repository, _, embedder, index_dir = _prepared_query(
        tmp_path,
        ("alpha passage", "alpha passage", "gamma passage"),
    )

    result = _execute(repository, index_dir, embedder, text="tie")

    ids = tuple(hit.candidate.record_id.as_str() for hit in result.hits)
    assert ids[:2] == tuple(sorted(ids[:2]))
    assert result.status is ResultStatus.COMPLETED
    assert tuple(hit.contributions[0].rank for hit in result.hits) == (1, 2, 3)
    assert all(
        hit.contributions[0].score_label == "cosine_similarity" for hit in result.hits
    )
    assert all(hit.contributions[0].channel.value == "dense" for hit in result.hits)
    assert embedder.requests[-1].purpose == "query"
    assert embedder.requests[-1].input_texts == ("tie",)
    assert embedder.requests[-1].options.dimensions == 3


def test_dense_query_honours_requested_limit(tmp_path: Path) -> None:
    repository, _, embedder, index_dir = _prepared_query(tmp_path)

    result = _execute(repository, index_dir, embedder, limit=1)

    assert len(result.hits) == 1
    assert result.hits[0].candidate.content == "alpha passage"
    assert result.hits[0].rank == 1


def test_dense_filters_rank_all_candidates_before_applying_limit(
    tmp_path: Path,
) -> None:
    repository, _, embedder, index_dir = _prepared_query(tmp_path)

    result = _execute(
        repository,
        index_dir,
        embedder,
        limit=2,
        filter_expression=ExactFilter("language", "en"),
    )

    assert tuple(hit.candidate.content for hit in result.hits) == (
        "beta passage",
        "gamma passage",
    )
    assert result.status is ResultStatus.COMPLETED


def test_dense_missing_filter_and_empty_results_are_completed(tmp_path: Path) -> None:
    repository, _, embedder, index_dir = _prepared_query(tmp_path)

    result = _execute(
        repository,
        index_dir,
        embedder,
        limit=1,
        filter_expression=MissingFilter("language"),
    )

    assert result.hits == ()
    empty = _execute(
        repository,
        index_dir,
        embedder,
        limit=2,
        filter_expression=ExactFilter("language", "de"),
    )
    assert empty.hits == ()
    assert empty.status is ResultStatus.COMPLETED


def test_query_against_empty_generation_completes_without_embedding(
    tmp_path: Path,
) -> None:
    embedder = _QueryEmbedder()
    seed_selection = selection(
        "seed",
        document_id=_DOC_IDS[0],
        document_version_id=_DOC_VERSION_IDS[0],
    )[1]
    source = BaselineChunkGenerationManifest(
        corpus_id=CORPUS_ID,
        corpus_version=_VERSION,
        configuration_identity=seed_selection.configuration_identity,
        generation_id="empty-dense-query-source",
        entries=(),
    )
    outcome = build_dense_generation(
        tmp_path / "index",
        source,
        embedder,
        metadata_db_path=tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "sources",
        compatibility=DenseIndexCompatibility(
            model_metadata=embedder.metadata,
            dimensions=3,
            embedding_profile=embedder.profile,
        ),
    )
    reference = replace(
        outcome.manifest.component_generation(),
        state=ComponentGenerationState.READY,
    )
    snapshot = _Snapshot(CORPUS_ID, _VERSION, reference, ())

    result = execute_dense_retrieval(
        cast(MetadataRepository, _Repository(snapshot)),
        tmp_path / "index",
        CORPUS_ID,
        RetrievalQuery("nearest", _VERSION, RetrievalLimit(3)),
        embedder,
    )

    assert result.status is ResultStatus.COMPLETED
    assert result.hits == ()
    assert embedder.requests == []


def test_dense_rejects_wrong_snapshot_version_without_hits(tmp_path: Path) -> None:
    repository, snapshot, embedder, index_dir = _prepared_query(tmp_path)
    snapshot.version = SimpleNamespace(corpus_version=_VERSION_OTHER)

    result = _execute(repository, index_dir, embedder)

    assert result.hits == ()
    assert result.failures[0].category is FailureCategory.CORPUS_VERSION_MISMATCH
    assert result.failures[0].channel is None


@pytest.mark.parametrize(
    ("metadata_name", "expected_category"),
    (
        ("foreign_corpus", FailureCategory.INVALID_CHANNEL_RESPONSE),
        ("missing_member", FailureCategory.CHANNEL_UNAVAILABLE),
    ),
)
def test_dense_rejects_cross_corpus_or_nonmember_hits(
    tmp_path: Path,
    metadata_name: str,
    expected_category: FailureCategory,
) -> None:
    repository, snapshot, embedder, index_dir = _prepared_query(tmp_path)
    if metadata_name == "foreign_corpus":
        snapshot.corpus_id = CorpusId.from_str("00000000-0000-4000-8000-000000000299")
    else:
        snapshot.visible_versions.clear()

    result = _execute(repository, index_dir, embedder)

    assert result.hits == ()
    assert result.failures[0].category is expected_category


def test_dense_embedding_failures_are_bounded_and_do_not_leak_query_text(
    tmp_path: Path,
) -> None:
    repository, _, embedder, index_dir = _prepared_query(tmp_path)
    embedder.fail = True

    result = _execute(
        repository,
        index_dir,
        embedder,
        text="secret query marker",
    )

    assert result.status is ResultStatus.FAILED
    assert result.failures[0].category is FailureCategory.CHANNEL_EXECUTION_FAILURE
    assert result.failures[0].retryable
    assert "secret query marker" not in repr(result)


def test_dense_rejects_query_embedding_profile_mismatch(tmp_path: Path) -> None:
    repository, _, embedder, index_dir = _prepared_query(tmp_path)
    embedder.output_profile = replace(embedder.profile, query_prefix="other: ")

    result = _execute(repository, index_dir, embedder)

    assert result.status is ResultStatus.FAILED
    assert result.failures[0].category is FailureCategory.INVALID_CHANNEL_RESPONSE
