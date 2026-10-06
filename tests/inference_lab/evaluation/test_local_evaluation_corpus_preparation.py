"""Acceptance for private, version-pinned corpus preparation."""

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from runpy import run_path
from textwrap import dedent
from types import FunctionType
from typing import cast

import pytest

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingProfile,
    EmbeddingRequest,
    EmbeddingResult,
    RerankRequest,
    RerankResult,
    ScoredCandidate,
)
from inference_lab.evaluation import (
    LocalDenseEvaluationAdapter,
    RetrievalEvaluationExecutionError,
)
from inference_lab.ingest import (
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineExecutionStatus,
    BaselineFailure,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineIngestSummary,
    BaselinePublicationStatus,
    BaselineStage,
    BaselineStageDisposition,
)
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.retrieval import (
    DENSE_COMPONENT_KIND,
    open_dense_generation,
)
from inference_lab.storage import open_metadata_repository
from tests.inference_lab.serving.operator_fixtures import (  # noqa: TID252
    LOCAL_TOKENISER_FIXTURE,
    PROJECT_ROOT,
    environment,
    run_example,
    write_config,
)

_SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/inference_lab/evaluation/prepare_local_evaluation_corpus.py"
)


def _run_with_test_embedder(tmp_path: Path) -> dict[str, object]:
    """Run the injected embedder acceptance in a fresh interpreter."""
    runner = dedent(
        """
        import json
        import os
        import sys
        from runpy import run_path
        from tests.inference_lab.evaluation.test_local_evaluation_corpus_preparation import (
            _EmbeddingFixture,
            _RerankerFixture,
        )
        from tests.inference_lab.evaluation.fixtures import case_payload
        from inference_lab.evaluation import (
            EvaluationCase,
            LocalDenseEvaluationAdapter,
            LocalFusedEvaluationAdapter,
            LocalRerankedEvaluationAdapter,
            RetrievalEvaluationExecutionError,
        )
        from inference_lab.profiling import ProfilingContext
        from inference_lab.provenance import CorpusId, CorpusVersionId
        from inference_lab.retrieval import (
            DENSE_COMPONENT_KIND,
            RetrievalLimit,
            ResultStatus,
            open_dense_generation,
        )
        from inference_lab.retrieval.reranking import (
            EvidenceCount,
            RerankCandidateLimit,
        )
        from inference_lab.settings import get_settings
        from inference_lab.storage import open_metadata_repository

        namespace = run_path(sys.argv[1], run_name="prepare_local_corpus_script")
        backend = _EmbeddingFixture()
        result = namespace["_run"](
            embedding_backend_factory=lambda _path: backend,
        )
        assert backend.is_closed
        settings = get_settings()
        corpus_id = CorpusId.from_str(os.environ["EVALUATION_CORPUS_ID"])
        corpus_version = CorpusVersionId.from_str(
            os.environ["EVALUATION_CORPUS_PROPOSED_VERSION"]
        )
        repository = open_metadata_repository(
            settings.storage.metadata_db_path,
            source_dir=settings.storage.source_dir,
            apply_migrations=False,
        )
        query_backend = _EmbeddingFixture()
        query_backend.load()
        try:
            with repository.open_corpus_snapshot(corpus_id, corpus_version) as snapshot:
                dense_reference = next(
                    item
                    for item in snapshot.manifest.component_generations
                    if item.component_kind == DENSE_COMPONENT_KIND
                )
            with open_dense_generation(
                settings.storage.index_dir, dense_reference.generation_id
            ) as generation:
                source_reference = generation.iter_entries()[0].candidate.sources[0]
            assert source_reference is not None
            adapter = LocalDenseEvaluationAdapter(
                repository,
                settings.storage.index_dir,
                corpus_id,
                corpus_version,
                query_backend,
            )
            case_record = case_payload("case-dense")
            case_record["question"] = "SARM1 literature passage"
            case_record["evidence"][0].update(
                corpus_version=corpus_version.as_str(),
                document_id=source_reference.document_id.as_str(),
                document_version_id=source_reference.document_version_id.as_str(),
                source_span_id=source_reference.source_span_id.as_str(),
            )
            response = adapter.execute(
                EvaluationCase.model_validate(case_record),
                corpus_id=corpus_id,
                corpus_version=corpus_version,
                candidate_depth=RetrievalLimit(5),
                profiling=ProfilingContext.create(),
                profiling_identity="case-dense",
            )
            assert response.result.status is ResultStatus.COMPLETED
            assert response.result.hits
            assert response.system_identity == adapter.system_identity
            fused_adapter = LocalFusedEvaluationAdapter(
                repository,
                settings.storage.index_dir,
                corpus_id,
                corpus_version,
                query_backend,
            )
            fused_response = fused_adapter.execute(
                EvaluationCase.model_validate(case_record),
                corpus_id=corpus_id,
                corpus_version=corpus_version,
                candidate_depth=RetrievalLimit(10),
                profiling=ProfilingContext.create(),
                profiling_identity="case-fused",
            )
            assert fused_response.result.status is ResultStatus.COMPLETED
            assert fused_response.result.hits
            assert fused_response.system_identity == fused_adapter.system_identity
            assert fused_adapter.system_identity.configuration_digest
            assert fused_adapter.fusion_policy.identity
            reranker = _RerankerFixture()
            reranker.load()
            reranked_adapter = LocalRerankedEvaluationAdapter(
                fused_adapter,
                reranker,
                evidence_count=EvidenceCount(5),
                candidate_limit=RerankCandidateLimit(10),
            )
            reranked_response = reranked_adapter.execute(
                EvaluationCase.model_validate(case_record),
                corpus_id=corpus_id,
                corpus_version=corpus_version,
                candidate_depth=RetrievalLimit(10),
                profiling=ProfilingContext.create(),
                profiling_identity="case-reranked",
            )
            assert reranked_response.result.status is ResultStatus.COMPLETED
            assert len(reranked_response.result.hits) <= EvidenceCount(5).value
            assert reranked_response.system_identity == (
                reranked_adapter.system_identity
            )
            assert reranked_adapter.reranking_identity.evidence_count == 5
            reranker.close()
            try:
                fused_adapter.execute(
                    EvaluationCase.model_validate(case_record),
                    corpus_id=CorpusId.generate(),
                    corpus_version=corpus_version,
                    candidate_depth=RetrievalLimit(10),
                    profiling=ProfilingContext.create(),
                    profiling_identity="case-fused-mismatch",
                )
            except RetrievalEvaluationExecutionError as error:
                assert error.category == "adapter_scope_mismatch"
            else:
                raise AssertionError("fused adapter accepted a different corpus")
        finally:
            query_backend.close()
            repository.close()
        print(json.dumps(result, sort_keys=True))
        """
    )
    child_environment = environment({})
    child_environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(PROJECT_ROOT),
            child_environment.get("PYTHONPATH", ""),
        )
        if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", runner, str(_SCRIPT)],
        cwd=tmp_path,
        env=child_environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return cast(dict[str, object], json.loads(completed.stdout.splitlines()[-1]))


def test_preparation_ingests_an_explicit_directory_without_leaking_payloads(
    tmp_path: Path,
) -> None:
    """One private directory becomes one corpus version with bounded output."""
    write_config(tmp_path)
    source = tmp_path / "approved-public-corpus"
    source.mkdir()
    document_text = "SARM1 literature passage " * 180
    (source / "paper.txt").write_text(document_text, encoding="utf-8")
    corpus_id = CorpusId.generate().as_str()
    version = CorpusVersionId.generate().as_str()

    output, stderr = run_example(
        tmp_path,
        _SCRIPT,
        {
            "EVALUATION_CORPUS_SOURCE_DIRECTORY": str(source),
            "EVALUATION_CORPUS_TOKENISER_DIRECTORY": str(LOCAL_TOKENISER_FIXTURE),
            "EVALUATION_CORPUS_ID": corpus_id,
            "EVALUATION_CORPUS_PROPOSED_VERSION": version,
        },
    )

    assert len(output) == 1
    assert {
        key: value
        for key, value in output[0].items()
        if key != "component_generation_ids"
    } == {
        "corpus_id": corpus_id,
        "document_outcome_count": 1,
        "execution_status": "completed",
        "failure_categories": [],
        "proposed_version": version,
        "publication_status": "committed",
        "published_document_count": 1,
        "readiness_failure_categories": [],
        "readiness_status": "ready",
    }
    assert {
        item["component_kind"] for item in output[0]["component_generation_ids"]
    } == {"chunks", "lexical-index"}
    safe_output = json.dumps(output) + stderr
    assert str(source) not in safe_output
    assert document_text not in safe_output
    assert "paper.txt" not in safe_output
    repository = open_metadata_repository(
        tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "source-data",
        apply_migrations=False,
    )
    try:
        with pytest.raises(RetrievalEvaluationExecutionError) as error:
            LocalDenseEvaluationAdapter(
                repository,
                tmp_path / "index",
                CorpusId.from_str(corpus_id),
                CorpusVersionId.from_str(version),
                _EmbeddingFixture(),
            )
        assert error.value.category == "invalid_dense_generation"
    finally:
        repository.close()


def test_readiness_rejects_published_partial_ingest() -> None:
    """A committed version with failed documents is not evaluation-ready."""
    namespace = run_path(str(_SCRIPT), run_name="prepare_local_corpus_script")
    readiness_check = cast(
        Callable[..., tuple[str, ...]],
        namespace["_readiness_failure_categories"],
    )
    corpus_id = CorpusId.generate()
    proposed_version = CorpusVersionId.generate()
    document_id = DocumentId.generate()
    document_version_id = DocumentVersionId.generate()
    failure = BaselineFailure(
        stage=BaselineStage.CHUNK,
        category=BaselineFailureCategory.CHUNK_FAILED,
        scope=BaselineFailureScope.DOCUMENT,
        severity=BaselineFailureSeverity.RECOVERABLE,
        retryable=False,
    )
    outcome = BaselineDocumentOutcome(
        ordinal=0,
        kind=BaselineDocumentOutcomeKind.FAILED,
        document_id=document_id,
        document_version_id=document_version_id,
        parse_disposition=BaselineStageDisposition.COMPLETED,
        chunk_disposition=BaselineStageDisposition.FAILED,
        failures=(failure,),
    )
    summary = BaselineIngestSummary(
        corpus_id=corpus_id,
        proposed_corpus_version=proposed_version,
        expected_corpus_version=None,
        execution_status=BaselineExecutionStatus.COMPLETED_WITH_FAILURES,
        publication_status=BaselinePublicationStatus.COMMITTED,
        document_outcomes=(outcome,),
    )

    categories = readiness_check(
        summary,
        published_document_versions=set(),
        published_component_kinds={"chunks", "lexical-index"},
        include_dense=False,
    )

    assert categories == (
        "ingest_incomplete",
        "document_outcomes_incomplete",
        "snapshot_membership_incomplete",
    )


def test_main_returns_failure_for_incomplete_corpus(capsys) -> None:
    """Expose a bounded readiness result and a failing process status."""
    namespace = run_path(str(_SCRIPT), run_name="prepare_local_corpus_script")
    main = cast(FunctionType, namespace["main"])
    main.__globals__["_run"] = lambda: {
        "readiness_status": "incomplete",
        "readiness_failure_categories": ["ingest_incomplete"],
    }

    assert main(None) == 1
    assert json.loads(capsys.readouterr().out) == {
        "readiness_status": "incomplete",
        "readiness_failure_categories": ["ingest_incomplete"],
    }


class _EmbeddingFixture:
    """Small deterministic embedder for atomic dense publication coverage."""

    metadata = BackendMetadata(
        backend_name="evaluation-test",
        model_name="evaluation-test-model",
        model_revision="r1",
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(BackendCapability.BATCHING,),
        embedding_dimensions=3,
    )
    output_profile = EmbeddingProfile(
        profile_version="1.0.0",
        tokenizer_id="evaluation-test-tokenizer",
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

    def __init__(self) -> None:
        self.is_ready = False
        self.is_closed = False

    def load(self) -> None:
        self.is_ready = True

    def close(self) -> None:
        self.is_ready = False
        self.is_closed = True

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            tuple((1.0, 0.0, 0.0) for _ in request.input_texts),
            self.metadata,
            self.output_profile,
        )


class _RerankerFixture:
    """Deterministic reranker for the corpus-backed adapter acceptance."""

    metadata = BackendMetadata(
        backend_name="evaluation-test-reranker",
        model_name="evaluation-test-reranker-model",
        model_revision="r1",
        supported_operations=(BackendOperation.RERANK,),
        capabilities=(BackendCapability.BATCHING,),
        preprocessing_profile_version="1.0.0",
    )

    def __init__(self) -> None:
        self.is_ready = False
        self.is_closed = False

    def load(self) -> None:
        self.is_ready = True

    def close(self) -> None:
        self.is_ready = False
        self.is_closed = True

    def rerank(self, request: RerankRequest) -> RerankResult:
        selected = request.candidates[: request.top_k]
        total = len(selected)
        return RerankResult(
            tuple(
                ScoredCandidate(
                    candidate_id=candidate.candidate_id,
                    score=float(total - rank + 1),
                    rank=rank,
                )
                for rank, candidate in enumerate(selected, start=1)
            ),
            self.metadata,
        )


def test_dense_index_is_published_with_the_same_immutable_corpus_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The optional local embedding stage joins the lexical publication."""
    write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "approved-public-corpus"
    source.mkdir()
    (source / "paper.txt").write_text(
        "SARM1 literature passage " * 180,
        encoding="utf-8",
    )
    corpus_id = CorpusId.generate()
    version = CorpusVersionId.generate()
    for key, value in {
        "EVALUATION_CORPUS_SOURCE_DIRECTORY": str(source),
        "EVALUATION_CORPUS_TOKENISER_DIRECTORY": str(LOCAL_TOKENISER_FIXTURE),
        "EVALUATION_CORPUS_ID": corpus_id.as_str(),
        "EVALUATION_CORPUS_PROPOSED_VERSION": version.as_str(),
        "EVALUATION_CORPUS_INCLUDE_DENSE": "true",
    }.items():
        monkeypatch.setenv(key, value)
    result = _run_with_test_embedder(tmp_path)
    assert result["publication_status"] == "committed"
    component_rows = cast(list[dict[str, str]], result["component_generation_ids"])
    components = {row["component_kind"]: row["generation_id"] for row in component_rows}
    assert set(components) == {"chunks", "lexical-index", DENSE_COMPONENT_KIND.as_str()}

    settings_repository = open_metadata_repository(
        tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "source-data",
        apply_migrations=False,
    )
    try:
        with settings_repository.open_corpus_snapshot(corpus_id, version) as snapshot:
            assert {
                reference.component_kind.as_str()
                for reference in snapshot.manifest.component_generations
            } == {"chunks", "lexical-index", DENSE_COMPONENT_KIND.as_str()}
        with open_dense_generation(
            tmp_path / "index", components[DENSE_COMPONENT_KIND.as_str()]
        ) as generation:
            assert generation.manifest.corpus_id == corpus_id
            assert generation.manifest.corpus_version == version
            assert generation.manifest.entry_count > 0
        query_backend = _EmbeddingFixture()
        adapter = LocalDenseEvaluationAdapter(
            settings_repository,
            tmp_path / "index",
            corpus_id,
            version,
            query_backend,
        )
        assert (
            adapter.system_identity.components[0].generation_id
            == components[DENSE_COMPONENT_KIND.as_str()]
        )
    finally:
        settings_repository.close()


def test_preparation_fails_closed_when_config_is_absent(
    capsys,
) -> None:
    """Missing deployment configuration produces one fixed safe message."""
    namespace = run_path(str(_SCRIPT), run_name="prepare_local_corpus_script")
    main = cast(Callable[[Sequence[str] | None], int], namespace["main"])
    assert main(None) == 1
    assert capsys.readouterr().err.strip() == (
        "local evaluation corpus preparation failed"
    )
