"""Cross-issue acceptance for the shipped lexical evaluation adapter."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.evaluation import (
    LocalLexicalEvaluationAdapter,
    RetrievalEvaluationExecutionError,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRequest,
    lexical_relationship_snapshot,
    load_dataset,
    retrieval_result_digest,
    run_retrieval_evaluation,
)
from inference_lab.ingest import Chunk
from inference_lab.profiling import ProfilingContext
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import LexicalMatchMode, RetrievalLimit
from inference_lab.storage import (
    ParsedRepresentation,
    ResolvedSourceSpan,
    _sqlite_corpus_snapshots,
    open_metadata_repository,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.evaluation.fixtures import (  # noqa: TID252
    case_payload,
    write_dataset_fixture,
)
from tests.inference_lab.retrieval.lexical_fixtures import (  # noqa: TID252
    CORPUS_ID,
    CORPUS_VERSION_ID,
    publish_query_fixture,
    source_manifest,
)
from tests.inference_lab.serving.operator_fixtures import (  # noqa: TID252
    LOCAL_TOKENISER_FIXTURE,
    example,
    run_example,
    write_config,
)


def _case_for_chunk(case_id: str, question: str, chunk: Chunk) -> dict:
    """Bind one evaluation case to a real persisted lexical source span."""
    payload = case_payload(case_id)
    payload["question"] = question
    payload["evidence"][0].update(
        document_id=chunk.document_id.as_str(),
        document_version_id=chunk.document_version_id.as_str(),
        source_span_id=chunk.source_references[0].source_span_id.as_str(),
    )
    return payload


def test_real_lexical_generation_scores_hit_and_empty_case(tmp_path: Path) -> None:
    """The adapter composes published storage, RET-02, and EVAL-03 scoring."""
    lexical_manifest, first, second = source_manifest()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3", apply_migrations=True
    )
    try:
        publish_query_fixture(repository, tmp_path / "indexes", lexical_manifest)
        dataset = load_dataset(
            write_dataset_fixture(
                tmp_path,
                [
                    _case_for_chunk("case-001", "alpha", first.chunks[0]),
                    _case_for_chunk("case-002", "unfindable", second.chunks[0]),
                ],
            )
        )
        adapter = LocalLexicalEvaluationAdapter(
            repository, tmp_path / "indexes", CORPUS_ID, CORPUS_VERSION_ID
        )
        request = RetrievalEvaluationRequest(
            dataset=dataset,
            selected_case_ids=("case-002", "case-001"),
            selected_split="development",
            corpus_id=CORPUS_ID,
            corpus_version=CORPUS_VERSION_ID,
            policy=RetrievalEvaluationPolicy(
                passage_cutoffs=(1, 5), candidate_depth=10
            ),
            relationship_snapshot=lexical_relationship_snapshot(
                repository, CORPUS_ID, CORPUS_VERSION_ID
            ),
            adapter=adapter,
        )
        run = run_retrieval_evaluation(request)
        assert run.identity.selected_case_ids == ("case-001", "case-002")
        assert run.identity.system_identity == adapter.system_identity
        assert (
            run.case_results[0].hits[0].record_id == first.chunks[0].metadata.record_id
        )
        assert run.case_results[0].document_recall is not None
        assert run.case_results[0].document_recall.value.numerator == 1
        assert run.case_results[1].hits == ()
        assert run.case_results[1].status.value == "completed"
        assert run.case_results[1].document_recall is not None
        assert run.case_results[1].document_recall.value.numerator == 0
        assert run.aggregate.case_count == 2
        component = adapter.system_identity.components[0]
        assert component.generation_id
        assert component.configuration_digest.startswith("sha256:")
        assert str(tmp_path) not in str(adapter.system_identity.as_dict())
        rerun = run_retrieval_evaluation(request)
        assert rerun.identity.run_fingerprint == run.identity.run_fingerprint
        assert retrieval_result_digest(rerun) == retrieval_result_digest(run)
        phrase_adapter = LocalLexicalEvaluationAdapter(
            repository,
            tmp_path / "indexes",
            CORPUS_ID,
            CORPUS_VERSION_ID,
            mode=LexicalMatchMode.EXACT_PHRASE,
        )
        assert (
            phrase_adapter.system_identity.configuration_digest
            != adapter.system_identity.configuration_digest
        )
    finally:
        repository.close()


def test_adapter_rejects_foreign_scope_before_query(tmp_path: Path) -> None:
    """A caller cannot reuse the pinned adapter against a different corpus."""
    lexical_manifest, first, _ = source_manifest()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3", apply_migrations=True
    )
    try:
        publish_query_fixture(repository, tmp_path / "indexes", lexical_manifest)
        adapter = LocalLexicalEvaluationAdapter(
            repository, tmp_path / "indexes", CORPUS_ID, CORPUS_VERSION_ID
        )
        dataset = load_dataset(
            write_dataset_fixture(
                tmp_path,
                [_case_for_chunk("case-001", "alpha", first.chunks[0])],
            )
        )
        with pytest.raises(RetrievalEvaluationExecutionError) as error:
            adapter.execute(
                dataset.cases[0],
                corpus_id=CORPUS_ID,
                corpus_version=type(CORPUS_VERSION_ID).generate(),
                candidate_depth=RetrievalLimit(10),
                profiling=ProfilingContext.create(),
                profiling_identity="case-00000001",
            )
        assert error.value.category == "adapter_scope_mismatch"
    finally:
        repository.close()


def test_adapter_and_serving_facade_find_same_published_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh-process ingest and serving agree with EVAL-03 on one version."""
    write_config(tmp_path)
    create_script = tmp_path / "create-corpus.py"
    ingest_script = tmp_path / "ingest-and-search.py"
    create_script.write_text(example("create"), encoding="utf-8")
    ingest_script.write_text(example("ingest"), encoding="utf-8")
    source = tmp_path / "paper.txt"
    source.write_text("alpha method " * 180, encoding="utf-8")
    created, _ = run_example(
        tmp_path, create_script, {"CORPUS_NAME": "Evaluation corpus"}
    )
    corpus_id = created[0]["corpus_id"]
    version = CORPUS_VERSION_ID.as_str()
    served, _ = run_example(
        tmp_path,
        ingest_script,
        {
            "CORPUS_ID": corpus_id,
            "SOURCE_FILE": str(source),
            "TOKENISER_DIR": str(LOCAL_TOKENISER_FIXTURE),
            "PROPOSED_VERSION": version,
            "QUERY": "alpha method",
        },
    )
    hit = served[1]["hits"][0]
    source_span_id = hit["source_locations"][0]["source_span_id"]
    case = case_payload()
    case["question"] = "alpha method"
    case["evidence"][0].update(
        document_id=hit["document_id"],
        source_span_id=source_span_id,
        corpus_version=version,
    )
    repository = open_metadata_repository(
        tmp_path / "metadata.sqlite3", source_dir=tmp_path / "source-data"
    )
    try:
        original = _sqlite_corpus_snapshots.get_parsed_representation
        representation_reads = 0

        def count_representation_reads(
            connection: sqlite3.Connection,
            selected_corpus_id: CorpusId,
            record_id: RecordId,
        ) -> ParsedRepresentation | None:
            nonlocal representation_reads
            representation_reads += 1
            return original(connection, selected_corpus_id, record_id)

        monkeypatch.setattr(
            _sqlite_corpus_snapshots,
            "get_parsed_representation",
            count_representation_reads,
        )
        selected_corpus = CorpusId.from_str(corpus_id)
        selected_version = CorpusVersionId.from_str(version)
        resolved: ResolvedSourceSpan | None = None
        with repository.open_corpus_snapshot(
            selected_corpus, selected_version
        ) as snapshot:
            resolved = snapshot.resolve_source_span(
                SourceSpanId.from_str(source_span_id)
            )
            assert resolved is not None
            assert (
                snapshot.resolve_source_span(SourceSpanId.from_str(source_span_id))
                == resolved
            )
            assert representation_reads == 1
            matching_memberships = tuple(
                membership
                for membership in snapshot.manifest.document_memberships
                if membership.document_id == DocumentId.from_str(hit["document_id"])
            )
            assert len(matching_memberships) == 1
            case["evidence"][0]["document_version_id"] = matching_memberships[
                0
            ].document_version_id.as_str()
        assert resolved is not None
        with repository.open_corpus_snapshot(
            selected_corpus, selected_version
        ) as next_snapshot:
            assert (
                next_snapshot.resolve_source_span(SourceSpanId.from_str(source_span_id))
                == resolved
            )
            assert representation_reads == 2
        (tmp_path / "dataset").mkdir()
        dataset = load_dataset(
            write_dataset_fixture(
                tmp_path / "dataset",
                [case],
                manifest_overrides={"corpus_id": corpus_id, "corpus_version": version},
            )
        )
        run = run_retrieval_evaluation(
            RetrievalEvaluationRequest(
                dataset=dataset,
                selected_case_ids=("case-001",),
                selected_split="development",
                corpus_id=selected_corpus,
                corpus_version=selected_version,
                policy=RetrievalEvaluationPolicy(
                    passage_cutoffs=(1,), candidate_depth=10
                ),
                relationship_snapshot=lexical_relationship_snapshot(
                    repository, selected_corpus, selected_version
                ),
                adapter=LocalLexicalEvaluationAdapter(
                    repository, tmp_path / "index", selected_corpus, selected_version
                ),
            )
        )
        evaluated_hit = run.case_results[0].hits[0]
        assert evaluated_hit.sources[0].document_id.as_str() == hit["document_id"]
        assert (
            evaluated_hit.sources[0].source_span_id.as_str()
            == hit["source_locations"][0]["source_span_id"]
        )
    finally:
        repository.close()
