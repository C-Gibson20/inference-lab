"""Acceptance tests for the local serving ingest and search facade."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Lock, get_ident
from time import perf_counter_ns
from typing import cast
from unittest.mock import patch

import pytest

from inference_lab.ingest import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_PARSER_VERSION,
    BaselineCheckpoint,
    BaselineCheckpointStatus,
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineFailureCategory,
    BaselineParseRepository,
    BaselineParseRequest,
    BaselineParseResult,
    BaselineProfilingContext,
    BaselineStage,
    BaselineStageDisposition,
    ChunkingPolicy,
    DiscoveredSource,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    FakeChunkTokeniser,
    ParserRegistry,
    StableReadFileSystem,
    execute_baseline_parse,
)
from inference_lab.ingest._chunk_identity import _configuration_identity_from_values
from inference_lab.profiling import FailureEvent, ProfilingSink, StageTiming
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
)
from inference_lab.retrieval import LexicalIndexConfiguration, open_lexical_generation
from inference_lab.retrieval.lexical_store_reader import (
    _sha256_file,
    _validate_database,
)
from inference_lab.serving import (
    LocalServingFacade,
    ServingIngestError,
    ServingIngestFailureCategory,
    ServingIngestRequest,
    ServingSearchError,
    ServingSearchFailureCategory,
    ServingSearchRequest,
)
from inference_lab.serving.ingest_documents import ServingDocumentRunner
from inference_lab.serving.ingest_publication import ServingPublicationCoordinator
from inference_lab.settings import (
    IngestConfig,
    ModelPaths,
    RuntimeConfig,
    Settings,
    StorageConfig,
)
from inference_lab.storage import (
    CorpusUpdateState,
    MetadataRepository,
    StorageConcurrencyError,
    WorkerRepositoryOpener,
    open_metadata_repository,
)


def _settings(
    tmp_path: Path,
    *,
    max_concurrent_documents: int = 1,
) -> Settings:
    """Build isolated local settings for the facade acceptance test."""
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
        ingest=IngestConfig(max_concurrent_documents=max_concurrent_documents),
    )


def _request(
    corpus_id: CorpusId,
    paths: tuple[Path, ...],
    *,
    proposed: CorpusVersionId,
    expected: CorpusVersionId | None,
) -> ServingIngestRequest:
    """Build one explicit corpus-scoped request over local files."""
    return ServingIngestRequest(
        corpus_id=corpus_id,
        discovery_request=DiscoveryRequest(
            corpus_id=corpus_id,
            inputs=tuple(
                DiscoveryInput(path, DiscoveryInputKind.FILE) for path in paths
            ),
        ),
        proposed_corpus_version=proposed,
        expected_corpus_version=expected,
        configuration_identity=_configuration_identity_from_values(
            parser_version=CURRENT_PARSER_VERSION,
            chunker_version=CURRENT_CHUNKER_VERSION,
            tokeniser_identity=FakeChunkTokeniser().identity,
            policy=ChunkingPolicy(min_tokens=1, max_tokens=32),
        ),
        chunking_policy=ChunkingPolicy(min_tokens=1, max_tokens=32),
        tokeniser=FakeChunkTokeniser(),
    )


class _CollectingSink:
    def __init__(self) -> None:
        self.records: list[object] = []

    def emit(self, record: object) -> None:
        self.records.append(record)


def _profiling(sink: _CollectingSink) -> BaselineProfilingContext:
    return BaselineProfilingContext.create(
        run_id="ingest-run",
        request_id="ingest-request",
        condition="cold",
        sink=cast(ProfilingSink, sink),
        clock=iter(range(0, 10_000_000_000, 1_000_000)).__next__,
        observed_at_factory=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )


def _terminal_timings(sink: _CollectingSink) -> list[StageTiming]:
    return [
        record
        for record in sink.records
        if isinstance(record, StageTiming) and record.event == "stage_completed"
    ]


def test_ingest_profiles_discovery_and_work_under_one_request(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "private-source.txt"
    source.write_text("private passage content", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Private corpus", corpus_id=corpus_id)
        sink = _CollectingSink()
        result = facade.ingest(request, profiling=_profiling(sink))
        assert result.execution_status.value == "completed"

        timings = _terminal_timings(sink)
        by_stage = {record.pipeline_stage: record for record in timings}
        request_id = "ingest-run:serving:ingest-request"
        assert {
            "serving",
            "discovery",
            "source_acquisition",
            "parse",
            "chunk",
            "membership",
            "lexical_index",
            "publication",
        } <= by_stage.keys()
        assert by_stage["serving"].stage_id == request_id
        assert by_stage["serving"].items_processed == 1
        assert by_stage["serving"].records_emitted == 1
        assert by_stage["discovery"].items_processed == 1
        assert by_stage["discovery"].records_emitted == 1
        assert all(
            record.parent_stage_id == request_id
            for record in timings
            if record.pipeline_stage
            in {
                "discovery",
                "parse",
                "chunk",
                "membership",
                "lexical_index",
                "publication",
            }
        )
        assert all(record.run_id == "ingest-run" for record in timings)
        assert all(record.request_id == "ingest-request" for record in timings)
        assert all(record.condition == "cold" for record in timings)
        assert str(source) not in repr(sink.records)
        assert "private passage content" not in repr(sink.records)
    finally:
        repository.close()


def test_ingest_profiles_returned_discovery_failure(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (tmp_path / "missing.txt",),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Corpus", corpus_id=corpus_id)
        sink = _CollectingSink()
        summary = facade.ingest(request, profiling=_profiling(sink))
        assert summary.failures[0].stage is BaselineStage.DISCOVERY
        assert summary.failures[0].category is BaselineFailureCategory.DISCOVERY_FAILED
        timings = _terminal_timings(sink)
        assert [(record.pipeline_stage, record.status) for record in timings] == [
            ("discovery", "failed"),
            ("serving", "failed"),
        ]
        failures = [
            record for record in sink.records if isinstance(record, FailureEvent)
        ]
        assert [record.pipeline_stage for record in failures] == [
            "discovery",
            "serving",
        ]
        assert failures[-1].failure_code == "discovery_failed"
        assert failures[-1].failure_category == "dependency"
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("target", "stage", "category"),
    [
        (
            "inference_lab.serving.ingest_discovery.inventory_local_sources",
            BaselineStage.DISCOVERY,
            BaselineFailureCategory.DISCOVERY_FAILED,
        ),
        (
            "inference_lab.serving.ingest.plan_baseline_membership",
            BaselineStage.MEMBERSHIP,
            BaselineFailureCategory.MEMBERSHIP_CONFLICT,
        ),
        (
            "inference_lab.serving.ingest_publication.build_lexical_generation_for_corpus_update",
            BaselineStage.PUBLICATION,
            BaselineFailureCategory.GENERATION_BUILD_FAILED,
        ),
    ],
)
def test_ingest_unexpected_failure_keeps_its_phase(
    tmp_path: Path,
    target: str,
    stage: BaselineStage,
    category: BaselineFailureCategory,
) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("source content", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Corpus", corpus_id=corpus_id)
        sink = _CollectingSink()
        with patch(target, side_effect=RuntimeError("sensitive internal detail")):
            summary = facade.ingest(request, profiling=_profiling(sink))
        assert summary.failures[0].stage is stage
        assert summary.failures[0].category is category
        assert any(
            isinstance(record, FailureEvent)
            and record.pipeline_stage == "serving"
            and record.failure_code == category.value
            for record in sink.records
        )
        if stage is BaselineStage.DISCOVERY:
            assert any(
                isinstance(record, FailureEvent)
                and record.pipeline_stage == "discovery"
                for record in sink.records
            )
        assert "sensitive internal detail" not in repr(sink.records)
    finally:
        repository.close()


def test_ingest_preserves_primary_exception_when_failure_emission_fails(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (tmp_path / "unused.txt",),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Corpus", corpus_id=corpus_id)

        class _BrokenSink(_CollectingSink):
            def emit(self, record: object) -> None:
                if (
                    isinstance(record, StageTiming)
                    and record.event == "stage_completed"
                ):
                    raise RuntimeError("instrumentation unavailable")
                super().emit(record)

        sink = _BrokenSink()
        with patch(
            "inference_lab.serving.ingest.discover_sources",
            side_effect=KeyboardInterrupt(),
        ):
            with pytest.raises(KeyboardInterrupt):
                facade.ingest(request, profiling=_profiling(sink))
        assert any(isinstance(record, FailureEvent) for record in sink.records)
    finally:
        repository.close()


def test_committed_replay_verification_error_is_not_returned_as_unpublished(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("source content", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Corpus", corpus_id=corpus_id)
        assert facade.ingest(request).publication_status.value == "committed"
        with patch.object(
            ServingPublicationCoordinator,
            "verify_published",
            side_effect=RuntimeError("verification unavailable"),
        ):
            with pytest.raises(RuntimeError, match="verification unavailable"):
                facade.ingest(request, profiling=_profiling(_CollectingSink()))
        assert repository.get_active_corpus_version(corpus_id) is not None
    finally:
        repository.close()


def _membership_hashes(
    repository: MetadataRepository,
    request: ServingIngestRequest,
) -> set[str]:
    """Read content hashes through the selected corpus snapshot."""
    version = repository.get_active_corpus_version(request.corpus_id)
    assert version is not None
    hashes: set[str] = set()
    with repository.open_corpus_snapshot(
        request.corpus_id,
        version.corpus_version,
    ) as snapshot:
        for membership in snapshot.manifest.document_memberships:
            document_version = snapshot.get_document_version(
                membership.document_version_id
            )
            assert document_version is not None
            hashes.add(document_version.content_hash.as_str())
    return hashes


def _write_process_config(tmp_path: Path) -> None:
    """Write a minimal config used by the fresh-process query example."""
    config = "\n".join(
        (
            "[models]",
            f"embedder_path = {json.dumps(str(tmp_path / 'models' / 'embedder'))}",
            f"reranker_path = {json.dumps(str(tmp_path / 'models' / 'reranker'))}",
            f"generator_path = {json.dumps(str(tmp_path / 'models' / 'generator'))}",
            "",
            "[runtime]",
            "vram_budget_gb = 24",
            "",
            "[storage]",
            f"index_dir = {json.dumps(str(tmp_path / 'index'))}",
            f"metadata_db_path = {json.dumps(str(tmp_path / 'metadata.sqlite3'))}",
            f"source_dir = {json.dumps(str(tmp_path / 'source-data'))}",
            "",
            "[logging]",
            'level = "CRITICAL"',
            'format = "json"',
            "",
            "[ingest]",
            "max_source_bytes = 52428800",
            "max_concurrent_documents = 1",
            "",
        )
    )
    (tmp_path / "config.toml").write_text(config, encoding="utf-8")


def _run_fresh_process_search(
    tmp_path: Path,
    corpus_id: CorpusId,
    query: str,
    *,
    corpus_version: CorpusVersionId | None = None,
) -> dict[str, object]:
    """Run the documented query example and decode its JSON result."""
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "inference_lab"
        / "serving-search.py"
    )
    command = [
        sys.executable,
        str(script),
        "--corpus-id",
        corpus_id.as_str(),
        "--query",
        query,
        "--limit",
        "5",
    ]
    if corpus_version is not None:
        command.extend(("--corpus-version", corpus_version.as_str()))
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else f"{source_root}{os.pathsep}{existing_pythonpath}"
    )
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    output = completed.stdout.strip().splitlines()
    assert output
    return json.loads(output[-1])


def test_ingest_is_corpus_scoped_and_replays_after_restart(tmp_path: Path) -> None:
    """Publish, restart, search, and replay two isolated corpora."""
    settings = _settings(tmp_path)
    shared = tmp_path / "shared.txt"
    alpha = tmp_path / "alpha.txt"
    beta = tmp_path / "beta.txt"
    shared.write_text("shared source content", encoding="utf-8")
    alpha.write_text("alpha only content", encoding="utf-8")
    beta.write_text("beta only content", encoding="utf-8")

    corpus_a = CorpusId.generate()
    corpus_b = CorpusId.generate()
    request_a = _request(
        corpus_a,
        (shared, alpha),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    request_b = _request(
        corpus_b,
        (shared, beta),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )

    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Corpus A", corpus_id=corpus_a)
        facade.create_corpus("Corpus B", corpus_id=corpus_b)

        first_a = facade.ingest(request_a)
        first_b = facade.ingest(request_b)

        assert first_a.publication_status.value == "committed"
        assert first_b.publication_status.value == "committed"
        assert len(repository.list_corpus_versions(corpus_a)) == 1
        assert len(repository.list_corpus_versions(corpus_b)) == 1
        hashes_a = _membership_hashes(repository, request_a)
        hashes_b = _membership_hashes(repository, request_b)
        assert len(hashes_a) == 2
        assert len(hashes_b) == 2
        assert len(hashes_a & hashes_b) == 1
        assert len(hashes_a - hashes_b) == 1
        assert len(hashes_b - hashes_a) == 1
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(reopened, settings)
        replayed = facade.ingest(request_a)
        assert replayed.publication_status.value == "committed"
        assert replayed.document_outcomes == ()
        assert len(reopened.list_corpus_versions(corpus_a)) == 1

        active_a = reopened.get_active_corpus_version(corpus_a)
        active_b = reopened.get_active_corpus_version(corpus_b)
        assert active_a is not None
        assert active_b is not None

        alpha_result = facade.search(
            ServingSearchRequest(corpus_id=corpus_a, text="alpha", limit=5)
        )
        beta_result = facade.search(
            ServingSearchRequest(corpus_id=corpus_b, text="beta", limit=5)
        )
        historical_result = facade.search(
            ServingSearchRequest(
                corpus_id=corpus_a,
                corpus_version=active_a.corpus_version,
                text="alpha",
                limit=5,
            )
        )
        assert alpha_result.corpus_version == active_a.corpus_version
        assert beta_result.corpus_version == active_b.corpus_version
        assert historical_result.corpus_version == active_a.corpus_version
        assert alpha_result.hits
        assert beta_result.hits
        assert historical_result.hits
        assert all("alpha" in hit.text for hit in alpha_result.hits)
        assert all("beta" in hit.text for hit in beta_result.hits)
        assert all(hit.source_locations for hit in alpha_result.hits)
        assert all(hit.source_locations for hit in beta_result.hits)
        for hit in (*alpha_result.hits, *beta_result.hits):
            for location in hit.source_locations:
                assert location.start_offset < location.end_offset
                assert location.span_start_offset <= location.span_end_offset
    finally:
        reopened.close()

    _write_process_config(tmp_path)
    fresh_alpha = _run_fresh_process_search(
        tmp_path,
        corpus_a,
        "alpha",
        corpus_version=active_a.corpus_version,
    )
    fresh_beta = _run_fresh_process_search(tmp_path, corpus_b, "beta")
    assert fresh_alpha["corpus_id"] == corpus_a.as_str()
    assert fresh_alpha["corpus_version"] == active_a.corpus_version.as_str()
    assert fresh_beta["corpus_id"] == corpus_b.as_str()
    assert fresh_beta["corpus_version"] == active_b.corpus_version.as_str()
    assert fresh_alpha["hits"]
    assert fresh_beta["hits"]


def test_multi_hit_search_validates_one_generation_per_request(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    sources = tuple(tmp_path / f"source-{index}.txt" for index in range(3))
    for index, source in enumerate(sources):
        source.write_text(f"shared-needle evidence {index}", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        sources,
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Search lifecycle corpus", corpus_id=corpus_id)
        assert facade.ingest(request).publication_status.value == "committed"
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
    )
    try:
        facade = LocalServingFacade(reopened, settings)
        search = ServingSearchRequest(
            corpus_id=corpus_id, text="shared-needle", limit=5
        )
        with (
            patch(
                "inference_lab.serving.search.open_lexical_generation",
                wraps=open_lexical_generation,
            ) as opened,
            patch(
                "inference_lab.retrieval.lexical_retrieval.open_lexical_generation",
                side_effect=AssertionError("retrieval must borrow the handle"),
            ),
            patch(
                "inference_lab.retrieval.lexical_store_reader._sha256_file",
                wraps=_sha256_file,
            ) as checksums,
            patch(
                "inference_lab.retrieval.lexical_store_reader._validate_database",
                wraps=_validate_database,
            ) as validations,
        ):
            start = perf_counter_ns()
            first = facade.search(search)
            first_ms = (perf_counter_ns() - start) / 1_000_000
            start = perf_counter_ns()
            repeat = facade.search(search)
            repeat_ms = (perf_counter_ns() - start) / 1_000_000
        assert len(first.hits) == len(repeat.hits) == 3
        assert tuple(hit.record_id for hit in first.hits) == tuple(
            hit.record_id for hit in repeat.hits
        )
        assert opened.call_count == checksums.call_count == validations.call_count == 2
        print(f"first_search_ms={first_ms:.3f} repeat_search_ms={repeat_ms:.3f}")
    finally:
        reopened.close()


def test_search_rejects_corrupt_lexical_database(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("checksum-sensitive evidence", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Corrupt lexical corpus", corpus_id=corpus_id)
        assert facade.ingest(request).publication_status.value == "committed"
        lexical_generation_id = ""
        with repository.open_active_corpus_snapshot(corpus_id) as snapshot:
            lexical_generation_id = next(
                ref.generation_id
                for ref in snapshot.manifest.component_generations
                if ref.component_kind.as_str() == "lexical-index"
            )
        database = (
            settings.storage.index_dir
            / "lexical"
            / lexical_generation_id
            / "index.sqlite3"
        )
        database.write_bytes(database.read_bytes() + b"tampered")
        with pytest.raises(ServingSearchError) as error:
            facade.search(
                ServingSearchRequest(
                    corpus_id=corpus_id, text="checksum-sensitive", limit=5
                )
            )
        assert error.value.category is ServingSearchFailureCategory.CORRUPT_STATE
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("interrupted_stage", "interrupted_status"),
    (
        (BaselineStage.PARSE, BaselineCheckpointStatus.STARTED),
        (BaselineStage.PARSE, BaselineCheckpointStatus.COMPLETED),
        (BaselineStage.CHUNK, BaselineCheckpointStatus.STARTED),
        (BaselineStage.CHUNK, BaselineCheckpointStatus.COMPLETED),
    ),
)
def test_ingest_resumes_each_durable_document_checkpoint_after_restart(
    tmp_path: Path,
    interrupted_stage: BaselineStage,
    interrupted_status: BaselineCheckpointStatus,
) -> None:
    """A process interruption reuses completed work and finishes started work."""
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("resumable lexical evidence", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Resume corpus", corpus_id=corpus_id)
        original = ServingDocumentRunner.write_checkpoint
        interrupted = False

        def interrupt_after_checkpoint(
            runner: ServingDocumentRunner,
            ingest_request: ServingIngestRequest,
            checkpoint: BaselineCheckpoint,
            *,
            expected_status: BaselineCheckpointStatus | None = None,
        ) -> None:
            nonlocal interrupted
            original(
                runner,
                ingest_request,
                checkpoint,
                expected_status=expected_status,
            )
            if (
                not interrupted
                and checkpoint.work_key.stage is interrupted_stage
                and checkpoint.status is interrupted_status
            ):
                interrupted = True
                raise KeyboardInterrupt

        with patch.object(
            ServingDocumentRunner,
            "write_checkpoint",
            interrupt_after_checkpoint,
        ):
            with pytest.raises(KeyboardInterrupt):
                facade.ingest(request)
        assert interrupted
        assert repository.get_active_corpus_version(corpus_id) is None
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
    )
    try:
        facade = LocalServingFacade(reopened, settings)
        summary = facade.ingest(request)
        assert summary.publication_status.value == "committed", summary
        assert len(summary.document_outcomes) == 1
        outcome = summary.document_outcomes[0]
        expected_parse = (
            BaselineStageDisposition.COMPLETED
            if interrupted_stage is BaselineStage.PARSE
            and interrupted_status is BaselineCheckpointStatus.STARTED
            else BaselineStageDisposition.REUSED
        )
        expected_chunk = (
            BaselineStageDisposition.REUSED
            if interrupted_stage is BaselineStage.CHUNK
            and interrupted_status is BaselineCheckpointStatus.COMPLETED
            else BaselineStageDisposition.COMPLETED
        )
        assert outcome.parse_disposition is expected_parse
        assert outcome.chunk_disposition is expected_chunk
        assert len(reopened.list_corpus_versions(corpus_id)) == 1
        result = facade.search(
            ServingSearchRequest(corpus_id=corpus_id, text="resumable", limit=5)
        )
        assert result.hits
    finally:
        reopened.close()


@pytest.mark.parametrize("interrupt_at", ("ready", "published"))
def test_ingest_resumes_publication_after_restart(
    tmp_path: Path,
    interrupt_at: str,
) -> None:
    """Ready work and committed-but-unacknowledged work survive cold restart."""
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("publication resume evidence", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Publication corpus", corpus_id=corpus_id)
        method = "build_and_publish" if interrupt_at == "ready" else "verify_published"
        with patch.object(
            ServingPublicationCoordinator,
            method,
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(KeyboardInterrupt):
                facade.ingest(request)
        assert (repository.get_active_corpus_version(corpus_id) is None) == (
            interrupt_at == "ready"
        )
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
    )
    try:
        summary = LocalServingFacade(reopened, settings).ingest(request)
        assert summary.publication_status.value == "committed"
        assert len(reopened.list_corpus_versions(corpus_id)) == 1
    finally:
        reopened.close()


def test_no_change_committed_replay_uses_parent_version(tmp_path: Path) -> None:
    """A no-change proposal remains replayable after its baseline manifest commits."""
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("unchanged source evidence", encoding="utf-8")
    corpus_id = CorpusId.generate()
    first = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("No-change corpus", corpus_id=corpus_id)
        assert facade.ingest(first).publication_status.value == "committed"
        unchanged = _request(
            corpus_id,
            (source,),
            proposed=CorpusVersionId.generate(),
            expected=first.proposed_corpus_version,
        )
        lexical_dir = settings.storage.index_dir / "lexical"
        before = {entry.name for entry in lexical_dir.iterdir()}
        with patch(
            "inference_lab.serving.ingest_publication."
            "build_lexical_generation_for_corpus_update",
            side_effect=AssertionError("unchanged ingest must not build"),
        ):
            assert facade.ingest(unchanged).publication_status.value == "no_change"
        assert {entry.name for entry in lexical_dir.iterdir()} == before
        assert len(repository.list_corpus_versions(corpus_id)) == 1
        update = repository.get_corpus_update(
            corpus_id, unchanged.proposed_corpus_version
        )
        assert update is not None
        assert update.state is CorpusUpdateState.ABORTED
        assert update.manifest_digest is None
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
    )
    try:
        summary = LocalServingFacade(reopened, settings).ingest(unchanged)
        assert summary.publication_status.value == "no_change"
        assert len(reopened.list_corpus_versions(corpus_id)) == 1
    finally:
        reopened.close()


def test_lexical_config_only_change_builds_new_generation(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("stable source evidence", encoding="utf-8")
    corpus_id = CorpusId.generate()
    first = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Lexical configuration corpus", corpus_id=corpus_id)
        assert facade.ingest(first).publication_status.value == "committed"
        lexical_dir = settings.storage.index_dir / "lexical"
        before = {entry.name for entry in lexical_dir.iterdir()}
        changed = replace(
            _request(
                corpus_id,
                (source,),
                proposed=CorpusVersionId.generate(),
                expected=first.proposed_corpus_version,
            ),
            lexical_configuration=LexicalIndexConfiguration(column_weights=(2.0,)),
        )
        assert facade.ingest(changed).publication_status.value == "committed"
        assert len(repository.list_corpus_versions(corpus_id)) == 2
        assert len({entry.name for entry in lexical_dir.iterdir()} - before) == 1
    finally:
        repository.close()


def test_ready_resume_rejects_staged_membership_drift(tmp_path: Path) -> None:
    """A READY proposal cannot publish a different staged document set."""
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("staged membership evidence", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Staged membership corpus", corpus_id=corpus_id)
        with patch.object(
            ServingPublicationCoordinator,
            "build_and_publish",
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(KeyboardInterrupt):
                facade.ingest(request)
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
    )
    try:
        facade = LocalServingFacade(reopened, settings)
        with patch.object(
            reopened,
            "list_corpus_update_document_memberships",
            return_value=(),
        ):
            with pytest.raises(ServingIngestError) as raised:
                facade.ingest(request)
        assert raised.value.category is ServingIngestFailureCategory.INCOMPATIBLE_RESUME
        assert reopened.get_active_corpus_version(corpus_id) is None
        assert facade.ingest(request).publication_status.value == "committed"
    finally:
        reopened.close()


def test_changed_source_rejects_incompatible_resume(tmp_path: Path) -> None:
    """New source bytes cannot reuse an unfinished proposal's identity."""
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("original content", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Changed source corpus", corpus_id=corpus_id)
        with patch.object(
            ServingPublicationCoordinator,
            "build_and_publish",
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(KeyboardInterrupt):
                facade.ingest(request)
    finally:
        repository.close()

    source.write_text("changed content", encoding="utf-8")
    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
    )
    try:
        with pytest.raises(ServingIngestError) as raised:
            LocalServingFacade(reopened, settings).ingest(request)
        assert raised.value.category is ServingIngestFailureCategory.INCOMPATIBLE_RESUME
        assert reopened.get_active_corpus_version(corpus_id) is None
    finally:
        reopened.close()


def test_changed_version_resumes_against_original_parent(tmp_path: Path) -> None:
    """Rediscovery of an unfinished replacement keeps its replace semantics."""
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("original passage", encoding="utf-8")
    corpus_id = CorpusId.generate()
    first = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Replacement corpus", corpus_id=corpus_id)
        assert facade.ingest(first).publication_status.value == "committed"
        source.write_text("replacement passage", encoding="utf-8")
        replacement = _request(
            corpus_id,
            (source,),
            proposed=CorpusVersionId.generate(),
            expected=first.proposed_corpus_version,
        )
        original = ServingDocumentRunner.write_checkpoint
        interrupted = False

        def interrupt_after_parse_started(
            runner: ServingDocumentRunner,
            ingest_request: ServingIngestRequest,
            checkpoint: BaselineCheckpoint,
            *,
            expected_status: BaselineCheckpointStatus | None = None,
        ) -> None:
            nonlocal interrupted
            original(
                runner,
                ingest_request,
                checkpoint,
                expected_status=expected_status,
            )
            if (
                not interrupted
                and checkpoint.work_key.stage is BaselineStage.PARSE
                and checkpoint.status is BaselineCheckpointStatus.STARTED
            ):
                interrupted = True
                raise KeyboardInterrupt

        with patch.object(
            ServingDocumentRunner,
            "write_checkpoint",
            interrupt_after_parse_started,
        ):
            with pytest.raises(KeyboardInterrupt):
                facade.ingest(replacement)
        assert interrupted
        active = repository.get_active_corpus_version(corpus_id)
        assert active is not None
        assert active.corpus_version == first.proposed_corpus_version
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
    )
    try:
        summary = LocalServingFacade(reopened, settings).ingest(replacement)
        assert summary.publication_status.value == "committed"
        assert (
            summary.document_outcomes[0].kind
            is BaselineDocumentOutcomeKind.CHANGED_VERSION
        )
        assert (
            summary.document_outcomes[0].parse_disposition
            is BaselineStageDisposition.COMPLETED
        )
        active = reopened.get_active_corpus_version(corpus_id)
        assert active is not None
        assert active.corpus_version == replacement.proposed_corpus_version
        assert len(reopened.list_corpus_versions(corpus_id)) == 2
    finally:
        reopened.close()


def test_failed_checkpoint_is_terminal_for_same_proposal(tmp_path: Path) -> None:
    """A failed stage does not silently retry under the same work key."""
    settings = _settings(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("retryable source", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Failed checkpoint corpus", corpus_id=corpus_id)
        original = ServingDocumentRunner.write_checkpoint

        def interrupt_after_failed_checkpoint(
            runner: ServingDocumentRunner,
            ingest_request: ServingIngestRequest,
            checkpoint: BaselineCheckpoint,
            *,
            expected_status: BaselineCheckpointStatus | None = None,
        ) -> None:
            original(
                runner,
                ingest_request,
                checkpoint,
                expected_status=expected_status,
            )
            if checkpoint.status is BaselineCheckpointStatus.FAILED:
                raise KeyboardInterrupt

        with (
            patch(
                "inference_lab.serving.ingest_documents.execute_baseline_parse",
                side_effect=RuntimeError("simulated parser failure"),
            ),
            patch.object(
                ServingDocumentRunner,
                "write_checkpoint",
                interrupt_after_failed_checkpoint,
            ),
        ):
            with pytest.raises(KeyboardInterrupt):
                facade.ingest(request)
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
    )
    try:
        facade = LocalServingFacade(reopened, settings)
        with patch(
            "inference_lab.serving.ingest_documents.execute_baseline_parse",
            side_effect=AssertionError("terminal checkpoint must not reparse"),
        ):
            summary = facade.ingest(request)
        assert summary.document_outcomes[0].kind is BaselineDocumentOutcomeKind.FAILED
        assert (
            summary.document_outcomes[0].parse_disposition
            is BaselineStageDisposition.FAILED
        )
        retry = _request(
            corpus_id,
            (source,),
            proposed=CorpusVersionId.generate(),
            expected=request.proposed_corpus_version,
        )
        assert facade.ingest(retry).publication_status.value == "committed"
    finally:
        reopened.close()


def test_parallel_ingest_owns_connections_and_isolates_corpora(tmp_path: Path) -> None:
    """Two workers use separate repositories and preserve corpus visibility."""
    settings = _settings(tmp_path, max_concurrent_documents=2)
    shared = tmp_path / "shared.txt"
    unique = tmp_path / "unique.txt"
    shared.write_text("shared evidence", encoding="utf-8")
    unique.write_text("unique evidence", encoding="utf-8")
    first_id = CorpusId.generate()
    second_id = CorpusId.generate()
    first = _request(
        first_id,
        (shared, unique),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    second = _request(
        second_id,
        (shared,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("First parallel corpus", corpus_id=first_id)
        facade.create_corpus("Second parallel corpus", corpus_id=second_id)
        barrier = Barrier(2)
        lock = Lock()
        caller_thread = get_ident()
        worker_repositories: list[MetadataRepository] = []
        worker_threads: set[int] = set()
        original = ServingDocumentRunner.process_source

        def observe_worker(
            runner: ServingDocumentRunner,
            ingest_request: ServingIngestRequest,
            source: DiscoveredSource,
            profiling: BaselineProfilingContext,
            parent_memberships: tuple[CorpusDocumentVersionMembership, ...],
            *,
            parent_stage_id: str | None = None,
        ) -> BaselineDocumentOutcome:
            if ingest_request.corpus_id == first_id:
                assert runner._repository is not repository
                with lock:
                    worker_repositories.append(runner._repository)
                    worker_threads.add(get_ident())
                assert get_ident() != caller_thread
                barrier.wait(timeout=10)
            return original(
                runner,
                ingest_request,
                source,
                profiling,
                parent_memberships,
                parent_stage_id=parent_stage_id,
            )

        with patch.object(ServingDocumentRunner, "process_source", observe_worker):
            first_summary = facade.ingest(first)
        assert first_summary.publication_status.value == "committed"
        assert len(first_summary.document_outcomes) == 2
        assert all(
            outcome.kind is not BaselineDocumentOutcomeKind.FAILED
            for outcome in first_summary.document_outcomes
        ), first_summary.document_outcomes
        assert len(worker_threads) == 2
        assert len({id(value) for value in worker_repositories}) == 2
        assert all(value.is_closed for value in worker_repositories)
        assert len(_membership_hashes(repository, first)) == 2

        second_summary = facade.ingest(second)
        assert second_summary.publication_status.value == "committed"
        assert len(_membership_hashes(repository, second)) == 1
        assert facade.search(
            ServingSearchRequest(corpus_id=first_id, text="unique", limit=5)
        ).hits
        assert not facade.search(
            ServingSearchRequest(corpus_id=second_id, text="unique", limit=5)
        ).hits
    finally:
        repository.close()


def test_parallel_ingest_resumes_mixed_document_outcomes(tmp_path: Path) -> None:
    """Parallel parse failure and success survive READY restart together."""
    settings = _settings(tmp_path, max_concurrent_documents=2)
    good = tmp_path / "a-good.txt"
    bad = tmp_path / "z-bad.txt"
    good.write_text("surviving evidence", encoding="utf-8")
    bad.write_text("failed evidence", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (good, bad),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Mixed parallel corpus", corpus_id=corpus_id)
        original_parse = execute_baseline_parse

        def fail_one_parse(
            repository: BaselineParseRepository,
            parse_request: BaselineParseRequest,
            *,
            settings: Settings,
            parser_registry: ParserRegistry,
            filesystem: StableReadFileSystem | None,
            profiling: BaselineProfilingContext,
            parent_stage_id: str | None = None,
        ) -> BaselineParseResult:
            if parse_request.source.ordinal == 1:
                raise RuntimeError("simulated parse failure")
            return original_parse(
                repository,
                parse_request,
                settings=settings,
                parser_registry=parser_registry,
                filesystem=filesystem,
                profiling=profiling,
                parent_stage_id=parent_stage_id,
            )

        with (
            patch(
                "inference_lab.serving.ingest_documents.execute_baseline_parse",
                fail_one_parse,
            ),
            patch.object(
                ServingPublicationCoordinator,
                "build_and_publish",
                side_effect=KeyboardInterrupt,
            ),
        ):
            with pytest.raises(KeyboardInterrupt):
                facade.ingest(request)
        assert repository.get_active_corpus_version(corpus_id) is None
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
    )
    try:
        facade = LocalServingFacade(reopened, settings)
        summary = facade.ingest(request)
        assert summary.publication_status.value == "committed"
        assert summary.execution_status.value == "completed_with_failures"
        assert len(summary.document_outcomes) == 2
        assert (
            summary.document_outcomes[0].kind is not BaselineDocumentOutcomeKind.FAILED
        )
        assert (
            summary.document_outcomes[0].parse_disposition
            is BaselineStageDisposition.REUSED
        )
        assert summary.document_outcomes[1].kind is BaselineDocumentOutcomeKind.FAILED
        assert (
            summary.document_outcomes[1].parse_disposition
            is BaselineStageDisposition.FAILED
        )
        assert len(_membership_hashes(reopened, request)) == 1
        assert facade.search(
            ServingSearchRequest(corpus_id=corpus_id, text="surviving", limit=5)
        ).hits
        assert not facade.search(
            ServingSearchRequest(corpus_id=corpus_id, text="failed", limit=5)
        ).hits
    finally:
        reopened.close()


def test_parallel_contention_exhaustion_leaves_proposal_resumable(
    tmp_path: Path,
) -> None:
    """Exhausted lock retries do not terminally fail the document work key."""
    settings = _settings(tmp_path, max_concurrent_documents=2)
    source = tmp_path / "source.txt"
    source.write_text("retry after contention", encoding="utf-8")
    corpus_id = CorpusId.generate()
    request = _request(
        corpus_id,
        (source,),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Contention corpus", corpus_id=corpus_id)
        worker_repositories: list[MetadataRepository] = []
        assert isinstance(repository, WorkerRepositoryOpener)
        original_open = repository.open_worker_repository

        def record_worker_open() -> MetadataRepository:
            worker = original_open()
            worker_repositories.append(worker)
            return worker

        with (
            patch.object(
                repository,
                "open_worker_repository",
                side_effect=record_worker_open,
            ),
            patch(
                "inference_lab.serving.ingest_documents.execute_baseline_parse",
                side_effect=StorageConcurrencyError,
            ) as parse,
        ):
            summary = facade.ingest(request)
        assert parse.call_count == 4
        assert len(worker_repositories) == 1
        assert worker_repositories[0].is_closed
        assert summary.execution_status.value == "failed"
        assert summary.publication_status.value == "not_published"
        assert len(summary.failures) == 1
        assert summary.failures[0].retryable
        assert summary.failures[0].stage is BaselineStage.PARSE_PERSISTENCE
        assert repository.get_active_corpus_version(corpus_id) is None
        assert facade.ingest(request).publication_status.value == "committed"
    finally:
        repository.close()
