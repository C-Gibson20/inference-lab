"""Tests for persistent immutable lexical generations."""

import hashlib
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    BaselineChunkGenerationEntry,
    BaselineChunkGenerationManifest,
    ChunkingPolicy,
    begin_baseline_lexical_corpus_update,
    build_and_publish_baseline_lexical_generation,
    publish_baseline_lexical_generations,
)
from inference_lab.profiling import FailureEvent, ProfilingContext, StageTiming
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
)
from inference_lab.retrieval import (
    CandidateKind,
    EvidenceResult,
    LexicalError,
    LexicalFailureCategory,
    LexicalGenerationParticipant,
    LexicalIndexConfiguration,
    LexicalMatchMode,
    LexicalQuery,
    RetrievalLimit,
    RetrievalQuery,
    build_lexical_generation,
    build_lexical_generation_for_corpus_update,
    derive_lexical_generation_id,
    execute_lexical_retrieval,
    open_lexical_generation,
    select_lexical_parent_reference,
    serialise_lexical_manifest,
)
from inference_lab.storage import (
    ChunkSetReader,
    ChunkSetWriteRequest,
    CommitCorpusUpdateOutcome,
    ComponentGenerationState,
    ComponentKind,
    CorpusVersionManifest,
    MetadataRepository,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.retrieval import lexical_fixtures  # noqa: TID252
from tests.inference_lab.retrieval.lexical_fixtures import (  # noqa: TID252
    CORPUS_ID,
    CORPUS_VERSION_ID,
    SECOND_CORPUS_ID,
    manifest,
    privacy_source_manifest,
    publish_query_fixture,
    published_lexical_generation_id,
    seed_metadata_repository,
    selection,
)


class _FakeSnapshot:
    """Pin one supplied committed manifest for parent-selection tests."""

    def __init__(self, manifest: CorpusVersionManifest) -> None:
        self._manifest = manifest

    def __enter__(self) -> "_FakeSnapshot":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    @property
    def version(self) -> object:
        return type("Version", (), {"corpus_version": self._manifest.corpus_version})()

    @property
    def manifest(self) -> CorpusVersionManifest:
        return self._manifest


class _FakeRepository:
    """Expose only the active-snapshot boundary used by lexical building."""

    def __init__(self, snapshot: _FakeSnapshot) -> None:
        self._snapshot = snapshot
        self.open_count = 0

    def open_active_corpus_snapshot(self, corpus_id: CorpusId) -> _FakeSnapshot:
        assert corpus_id == self._snapshot.manifest.corpus_id
        self.open_count += 1
        return self._snapshot

    def open_corpus_snapshot(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> _FakeSnapshot:
        assert corpus_id == self._snapshot.manifest.corpus_id
        assert corpus_version == self._snapshot.manifest.corpus_version
        self.open_count += 1
        return self._snapshot


class _CollectingSink:
    """Collect shared profiling records without rendering them."""

    def __init__(self) -> None:
        self.records: list[object] = []

    def emit(self, record: object) -> None:
        self.records.append(record)


def test_lexical_generation_preserves_chunk_source_range(tmp_path: Path) -> None:
    """Persisted candidates retain the complete source interval after reopen."""
    chunk_set, selected = selection("A source passage.")
    expected = chunk_set.chunks[0]
    source_manifest = manifest((selected,))
    built = build_lexical_generation(tmp_path, source_manifest)

    with open_lexical_generation(tmp_path, built.manifest.generation_id) as generation:
        actual = generation.get_chunk(expected.record_id)
        assert actual is not None
        assert actual.source_start == expected.source_start
        assert actual.source_end == expected.source_end
        assert (
            actual.source_links[0].source_start
            == expected.source_references[0].source_start
        )
        assert (
            actual.source_links[0].source_end
            == expected.source_references[0].source_end
        )


def _prepare_changed_manifest(
    repository: SQLiteMetadataRepository,
    source_manifest: BaselineChunkGenerationManifest,
    *,
    corpus_version: CorpusVersionId,
) -> BaselineChunkGenerationManifest:
    """Persist a changed first member while removing the second member."""
    first_entry = next(
        entry
        for entry in source_manifest.entries
        if b"alpha cites" in entry.selection.canonical_payload
    )
    changed_version = DocumentVersionId.generate()
    _, changed_selection = selection(
        "changed lexical content",
        document_id=first_entry.membership.document_id,
        document_version_id=changed_version,
    )
    changed_membership = CorpusDocumentVersionMembership(
        corpus_id=source_manifest.corpus_id,
        document_id=first_entry.membership.document_id,
        document_version_id=changed_version,
    )
    with repository.transaction() as transaction:
        transaction.add_document_version(
            DocumentVersion(
                changed_version,
                changed_membership.document_id,
                ContentHash(HashAlgorithm.SHA256, "c" * 64),
            )
        )
        transaction.add_corpus_document_version_membership(changed_membership)
        transaction.write_chunk_set(
            ChunkSetWriteRequest(
                corpus_id=source_manifest.corpus_id,
                selection=changed_selection,
            )
        )
    return BaselineChunkGenerationManifest(
        corpus_id=source_manifest.corpus_id,
        corpus_version=corpus_version,
        configuration_identity=source_manifest.configuration_identity,
        generation_id="baseline-query-update",
        entries=(
            BaselineChunkGenerationEntry(
                membership=changed_membership,
                selection=changed_selection,
            ),
        ),
    )


def _rebind_source_manifest(
    source_manifest: BaselineChunkGenerationManifest,
    *,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    generation_id: str,
) -> BaselineChunkGenerationManifest:
    """Reuse detached selections under a different corpus/version identity."""
    entries = tuple(
        sorted(
            (
                BaselineChunkGenerationEntry(
                    membership=CorpusDocumentVersionMembership(
                        corpus_id=corpus_id,
                        document_id=entry.membership.document_id,
                        document_version_id=entry.membership.document_version_id,
                    ),
                    selection=entry.selection,
                )
                for entry in source_manifest.entries
            ),
            key=lambda entry: entry.key(),
        )
    )
    return BaselineChunkGenerationManifest(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        configuration_identity=source_manifest.configuration_identity,
        generation_id=generation_id,
        entries=entries,
    )


def _install_publication_failure(database_path: Path) -> None:
    """Install a private SQLite failure at component registration."""
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """CREATE TRIGGER reject_lexical_publication
            BEFORE INSERT ON corpus_update_component_generations
            BEGIN
                SELECT RAISE(ABORT, 'private-lexical-publication-failure');
            END"""
        )
        connection.commit()
    finally:
        connection.close()


def _remove_publication_failure(database_path: Path) -> None:
    """Remove the private SQLite publication failure trigger."""
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP TRIGGER reject_lexical_publication")
        connection.commit()
    finally:
        connection.close()


def test_configured_content_weight_changes_bm25_score(tmp_path: Path) -> None:
    """The sole indexed content weight must affect the executed FTS5 rank."""
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    baseline = build_lexical_generation(tmp_path, source_manifest)
    weighted = build_lexical_generation(
        tmp_path,
        source_manifest,
        configuration=LexicalIndexConfiguration(column_weights=(2.0,)),
    )
    request = LexicalQuery(
        RetrievalQuery("alpha", source_manifest.corpus_version, RetrievalLimit(5))
    )
    with open_lexical_generation(
        tmp_path, baseline.manifest.generation_id
    ) as baseline_generation:
        baseline_matches = baseline_generation.search(request)
    with open_lexical_generation(
        tmp_path, weighted.manifest.generation_id
    ) as weighted_generation:
        weighted_matches = weighted_generation.search(request)

    assert len(baseline_matches) == len(weighted_matches) == 1
    assert baseline_matches[0][0].record_id == weighted_matches[0][0].record_id
    assert weighted_matches[0][1] > baseline_matches[0][1]

    # Simulate a previously published v1 generation: its advertised knobs
    # were ignored, so opening it must retain the original unweighted score.
    legacy = LexicalIndexConfiguration(
        engine_revision="sqlite-fts5-baseline",
        bm25_k1=1.5,
        bm25_b=0.5,
        column_weights=(1.0, 2.0),
        score_label="sqlite_fts5_negated_bm25_v1",
    )
    generation_dir = tmp_path / "lexical" / weighted.manifest.generation_id
    database = generation_dir / "index.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE generation_metadata SET value = ? WHERE key = ?",
            (legacy.configuration_identity.digest, "configuration_digest"),
        )
    legacy_manifest = replace(
        weighted.manifest,
        configuration=legacy,
        configuration_digest=legacy.configuration_identity.digest,
        database_checksum=hashlib.sha256(database.read_bytes()).hexdigest(),
        contract_version="1.0.0",
    )
    (generation_dir / "manifest.json").write_bytes(
        serialise_lexical_manifest(legacy_manifest)
    )
    with open_lexical_generation(
        tmp_path, legacy_manifest.generation_id
    ) as legacy_generation:
        legacy_matches = legacy_generation.search(request)
    assert legacy_matches[0][1] == baseline_matches[0][1]
    with pytest.raises(LexicalError) as error:
        build_lexical_generation(tmp_path, source_manifest, configuration=legacy)
    assert error.value.category is LexicalFailureCategory.INVALID_REQUEST


def test_pr4_query_executes_terms_phrase_and_identifier_modes_from_pinned_snapshot(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    try:
        seed_metadata_repository(repository, source_manifest)
        published = build_and_publish_baseline_lexical_generation(
            repository,
            tmp_path / "indexes",
            source_manifest,
            expected_corpus_version=None,
        )
        assert isinstance(published, CommitCorpusUpdateOutcome)

        terms = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "alpha",
                    source_manifest.corpus_version,
                    RetrievalLimit(5),
                )
            ),
        )
        phrase = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "alpha cites",
                    source_manifest.corpus_version,
                    RetrievalLimit(5),
                ),
                LexicalMatchMode.EXACT_PHRASE,
            ),
        )
        identifier = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "10.1234/ABC-7",
                    source_manifest.corpus_version,
                    RetrievalLimit(5),
                ),
                LexicalMatchMode.IDENTIFIER,
            ),
        )

        assert all(
            isinstance(result, EvidenceResult)
            and result.status.value == "completed"
            and not result.failures
            for result in (terms, phrase, identifier)
        )
        assert len(terms.hits) == 1
        assert terms.hits[0].candidate.kind is CandidateKind.ORIGINAL_PASSAGE
        assert terms.hits[0].candidate.content.startswith("alpha cites")
        assert terms.hits[0].candidate.sources
        assert len(phrase.hits) == 1
        assert phrase.hits[0].candidate.record_id == terms.hits[0].candidate.record_id
        assert len(identifier.hits) == 1
        assert (
            identifier.hits[0].candidate.record_id == terms.hits[0].candidate.record_id
        )
        assert identifier.hits[0].contributions[0].score_label == (
            "sqlite_exact_identifier_occurrences_v1"
        )
    finally:
        repository.close()


def test_pr4_query_rejects_foreign_generation_reference_before_opening_files(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    outcome = build_lexical_generation(tmp_path, source_manifest)
    reference = replace(
        LexicalGenerationParticipant(
            tmp_path,
            outcome,
            source_manifest,
        ).component_generation(),
        corpus_id=SECOND_CORPUS_ID,
        state=ComponentGenerationState.READY,
    )
    fake_manifest = type(
        "QueryManifest",
        (),
        {
            "corpus_id": CORPUS_ID,
            "corpus_version": source_manifest.corpus_version,
            "component_generations": (reference,),
        },
    )()
    repository = _FakeRepository(
        _FakeSnapshot(cast(CorpusVersionManifest, fake_manifest))
    )

    result = execute_lexical_retrieval(
        cast(MetadataRepository, repository),
        tmp_path,
        CORPUS_ID,
        LexicalQuery(
            RetrievalQuery(
                "alpha",
                source_manifest.corpus_version,
                RetrievalLimit(5),
            )
        ),
    )

    assert result.status.value == "failed"
    assert result.hits == ()
    assert result.failures[0].category.value == "invalid_channel_response"
    assert repository.open_count == 1


def test_pr4_query_keeps_old_reader_queryable_during_new_publication(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    old_reader: EvidenceResult | None = None
    new_reader: EvidenceResult | None = None
    try:
        index_dir = tmp_path / "indexes"
        publish_query_fixture(repository, index_dir, source_manifest)
        with repository.open_corpus_snapshot(
            CORPUS_ID,
            source_manifest.corpus_version,
        ) as entered_old_snapshot:
            old_snapshot = _FakeSnapshot(entered_old_snapshot.manifest)
            changed_version = CorpusVersionId.generate()
            changed_manifest = _prepare_changed_manifest(
                repository,
                source_manifest,
                corpus_version=changed_version,
            )
            updated = build_and_publish_baseline_lexical_generation(
                repository,
                index_dir,
                changed_manifest,
                expected_corpus_version=source_manifest.corpus_version,
            )
            assert isinstance(updated, CommitCorpusUpdateOutcome)

            old_reader = execute_lexical_retrieval(
                cast(MetadataRepository, _FakeRepository(old_snapshot)),
                index_dir,
                CORPUS_ID,
                LexicalQuery(
                    RetrievalQuery(
                        "alpha",
                        source_manifest.corpus_version,
                        RetrievalLimit(5),
                    )
                ),
            )
            new_reader = execute_lexical_retrieval(
                repository,
                index_dir,
                CORPUS_ID,
                LexicalQuery(
                    RetrievalQuery(
                        "changed",
                        changed_version,
                        RetrievalLimit(5),
                    )
                ),
            )

        assert old_reader is not None
        assert new_reader is not None
        assert old_reader.status.value == "completed"
        assert len(old_reader.hits) == 1
        assert old_reader.hits[0].candidate.content.startswith("alpha cites")
        assert new_reader.status.value == "completed"
        assert len(new_reader.hits) == 1
        assert new_reader.hits[0].candidate.content == "changed lexical content"
    finally:
        repository.close()


def test_pr4_query_isolates_two_corpora_with_overlapping_document_versions(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    shared_entry = next(
        entry
        for entry in source_manifest.entries
        if b"alpha cites" in entry.selection.canonical_payload
    )
    unique_document_id = DocumentId.generate()
    unique_version_id = DocumentVersionId.generate()
    _, unique_selection = selection(
        "second corpus only",
        document_id=unique_document_id,
        document_version_id=unique_version_id,
    )
    second_manifest = BaselineChunkGenerationManifest(
        corpus_id=SECOND_CORPUS_ID,
        corpus_version=CorpusVersionId.generate(),
        configuration_identity=source_manifest.configuration_identity,
        generation_id="baseline-second-query-corpus",
        entries=tuple(
            sorted(
                (
                    BaselineChunkGenerationEntry(
                        membership=CorpusDocumentVersionMembership(
                            corpus_id=SECOND_CORPUS_ID,
                            document_id=shared_entry.membership.document_id,
                            document_version_id=shared_entry.membership.document_version_id,
                        ),
                        selection=shared_entry.selection,
                    ),
                    BaselineChunkGenerationEntry(
                        membership=CorpusDocumentVersionMembership(
                            corpus_id=SECOND_CORPUS_ID,
                            document_id=unique_document_id,
                            document_version_id=unique_version_id,
                        ),
                        selection=unique_selection,
                    ),
                ),
                key=lambda entry: entry.key(),
            )
        ),
    )
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    try:
        publish_query_fixture(repository, tmp_path / "indexes", source_manifest)
        with repository.transaction() as transaction:
            transaction.add_document(DocumentIdentity(unique_document_id))
            transaction.add_document_version(
                DocumentVersion(
                    unique_version_id,
                    unique_document_id,
                    ContentHash(HashAlgorithm.SHA256, "2" * 64),
                )
            )
        seed_metadata_repository(repository, second_manifest, seed_catalogue=False)
        second = build_and_publish_baseline_lexical_generation(
            repository,
            tmp_path / "indexes",
            second_manifest,
            expected_corpus_version=None,
        )
        assert isinstance(second, CommitCorpusUpdateOutcome)

        first_only = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "second corpus only",
                    source_manifest.corpus_version,
                    RetrievalLimit(5),
                )
            ),
        )
        second_only = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            SECOND_CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "second corpus only",
                    second_manifest.corpus_version,
                    RetrievalLimit(5),
                )
            ),
        )
        second_beta = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            SECOND_CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "beta",
                    second_manifest.corpus_version,
                    RetrievalLimit(5),
                )
            ),
        )

        assert first_only.status.value == "completed"
        assert first_only.hits == ()
        assert second_only.status.value == "completed"
        assert len(second_only.hits) == 1
        assert second_only.hits[0].candidate.content == "second corpus only"
        assert second_beta.status.value == "completed"
        assert second_beta.hits == ()
    finally:
        repository.close()


def test_pr4_query_keeps_payloads_out_of_observability_surfaces(
    tmp_path: Path,
) -> None:
    source_manifest = privacy_source_manifest()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    sink = _CollectingSink()
    profiling = ProfilingContext.create(
        run_id="run-lexical-privacy",
        sink=sink,
        clock=iter(range(0, 200_000_000, 10_000_000)).__next__,
        observed_at_factory=lambda: datetime(2026, 9, 18, tzinfo=UTC),
    )
    query_marker = "private lexical query marker"
    chunk_marker = "private lexical chunk marker"
    identifier_marker = "10.1234/PRIVATE-7"
    raw_error_marker = "private lexical sqlite error marker"
    try:
        outcome = publish_query_fixture(
            repository,
            tmp_path / "indexes",
            source_manifest,
        )
        with capture_logs() as logs:
            result = execute_lexical_retrieval(
                repository,
                tmp_path / "indexes",
                CORPUS_ID,
                LexicalQuery(
                    RetrievalQuery(
                        query_marker,
                        source_manifest.corpus_version,
                        RetrievalLimit(5),
                    )
                ),
                profiling=profiling,
                profiling_identity="privacy-success",
            )

        assert result.status.value == "completed"
        assert result.hits == ()
        assert result.failures == ()
        safe_surfaces = (repr(result), repr(sink.records), repr(logs))
        for marker in (
            query_marker,
            chunk_marker,
            identifier_marker,
            str(tmp_path),
        ):
            assert all(marker not in surface for surface in safe_surfaces)

        database = (
            tmp_path
            / "indexes"
            / "lexical"
            / published_lexical_generation_id(outcome)
            / "index.sqlite3"
        )
        with database.open("ab") as stream:
            stream.write(raw_error_marker.encode())

        with capture_logs() as failure_logs:
            failed = execute_lexical_retrieval(
                repository,
                tmp_path / "indexes",
                CORPUS_ID,
                LexicalQuery(
                    RetrievalQuery(
                        query_marker,
                        source_manifest.corpus_version,
                        RetrievalLimit(5),
                    )
                ),
                profiling=profiling,
                profiling_identity="privacy-failure",
            )

        assert failed.status.value == "failed"
        assert failed.hits == ()
        assert failed.failures[0].category.value == "channel_unavailable"
        failure_surfaces = (
            repr(failed),
            repr(sink.records),
            repr(failure_logs),
        )
        for marker in (
            query_marker,
            chunk_marker,
            identifier_marker,
            raw_error_marker,
            str(tmp_path),
        ):
            assert all(marker not in surface for surface in failure_surfaces)
    finally:
        repository.close()


def test_lexical_build_profiles_success_and_reuse_with_shared_context(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    sink = _CollectingSink()
    clock_values = iter(range(0, 100_000_000, 10_000_000))
    profiling = ProfilingContext.create(
        run_id="run-lexical",
        sink=sink,
        clock=lambda: next(clock_values),
        observed_at_factory=lambda: datetime(2026, 9, 18, tzinfo=UTC),
    )

    first = build_lexical_generation(
        tmp_path,
        source_manifest,
        profiling=profiling,
        parent_stage_id="run-lexical:publication:request",
        profiling_identity="first",
    )
    replay = build_lexical_generation(
        tmp_path,
        source_manifest,
        profiling=profiling,
        profiling_identity="replay",
    )

    assert replay.statistics.database_reused is True
    completions = [
        record
        for record in sink.records
        if isinstance(record, StageTiming) and record.event == "stage_completed"
    ]
    assert [
        (
            record.module,
            record.pipeline_stage,
            record.status,
            record.items_processed,
            record.records_emitted,
            record.parent_stage_id,
        )
        for record in completions
    ] == [
        (
            "inference_lab.retrieval",
            "lexical_index",
            "completed",
            first.manifest.chunk_count,
            first.manifest.chunk_count,
            "run-lexical:publication:request",
        ),
        (
            "inference_lab.retrieval",
            "lexical_index",
            "completed",
            replay.manifest.chunk_count,
            replay.manifest.chunk_count,
            None,
        ),
    ]
    assert all(record.run_id == "run-lexical" for record in completions)
    assert not [record for record in sink.records if isinstance(record, FailureEvent)]


def test_lexical_build_profiles_generation_failure_without_payloads(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    sink = _CollectingSink()
    profiling = ProfilingContext.create(
        run_id="run-lexical-failure",
        sink=sink,
        clock=iter((0, 10_000_000)).__next__,
        observed_at_factory=lambda: datetime(2026, 9, 18, tzinfo=UTC),
    )

    with pytest.raises(LexicalError) as raised:
        build_lexical_generation(
            tmp_path,
            source_manifest,
            parent_generation_id="missing-parent",
            profiling=profiling,
        )

    assert raised.value.category is LexicalFailureCategory.GENERATION_UNAVAILABLE
    failures = [record for record in sink.records if isinstance(record, FailureEvent)]
    assert len(failures) == 1
    assert failures[0].failure_category == "dependency"
    assert failures[0].failure_code == "generation_unavailable"
    assert "missing-parent" not in repr(sink.records)
    assert str(tmp_path) not in repr(sink.records)


def test_build_and_publish_uses_one_atomic_generation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    index_dir = tmp_path / "indexes"
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    repository = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
    )
    try:
        seed_metadata_repository(repository, source_manifest)
        result = build_and_publish_baseline_lexical_generation(
            repository,
            index_dir,
            source_manifest,
            expected_corpus_version=None,
        )

        assert isinstance(result, CommitCorpusUpdateOutcome)
        with repository.open_active_corpus_snapshot(CORPUS_ID) as snapshot:
            assert snapshot.version.corpus_version == CORPUS_VERSION_ID
            assert snapshot.manifest.document_memberships == tuple(
                entry.membership for entry in source_manifest.entries
            )
            assert tuple(
                generation.component_kind.as_str()
                for generation in snapshot.manifest.component_generations
            ) == ("chunks", "lexical-index")
            assert all(
                generation.state is ComponentGenerationState.READY
                for generation in snapshot.manifest.component_generations
            )
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
    )
    try:
        with reopened.open_active_corpus_snapshot(CORPUS_ID) as snapshot:
            assert snapshot.version.corpus_version == CORPUS_VERSION_ID
            assert len(snapshot.manifest.component_generations) == 2
    finally:
        reopened.close()


def test_pr3_failed_publication_keeps_active_version_and_leaves_orphan_unreachable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    index_dir = tmp_path / "indexes"
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    repository = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
    )
    try:
        seed_metadata_repository(repository, source_manifest)
        first = build_and_publish_baseline_lexical_generation(
            repository,
            index_dir,
            source_manifest,
            expected_corpus_version=None,
        )
        assert isinstance(first, CommitCorpusUpdateOutcome)

        child = _rebind_source_manifest(
            source_manifest,
            corpus_id=CORPUS_ID,
            corpus_version=CorpusVersionId.generate(),
            generation_id="baseline-failed-generation",
        )
        child_generation_id = derive_lexical_generation_id(
            child.corpus_id,
            child.corpus_version,
            LexicalIndexConfiguration(),
            child.source_manifest_digest,
        )
        child_outcome = build_lexical_generation_for_corpus_update(
            index_dir,
            child,
            repository,
            expected_corpus_version=source_manifest.corpus_version,
        )
        child_participant = LexicalGenerationParticipant(
            index_dir,
            child_outcome,
            child,
        )
        _install_publication_failure(database_path)

        with pytest.raises(StorageConflictError) as raised:
            with repository.transaction() as transaction:
                begin_baseline_lexical_corpus_update(
                    transaction,
                    child.corpus_id,
                    child.corpus_version,
                    expected_corpus_version=source_manifest.corpus_version,
                )
                for entry in child.entries:
                    transaction.add_corpus_update_document_membership(
                        child.corpus_id,
                        child.corpus_version,
                        entry.membership,
                    )
                publish_baseline_lexical_generations(
                    transaction,
                    repository,
                    child,
                    child_participant,
                )

        assert "private-lexical-publication-failure" not in str(raised.value)
        _remove_publication_failure(database_path)
        with repository.open_active_corpus_snapshot(CORPUS_ID) as snapshot:
            assert snapshot.version.corpus_version == source_manifest.corpus_version
            assert len(snapshot.manifest.component_generations) == 2

        orphan = index_dir / "lexical" / child_generation_id
        assert orphan.is_dir()
        with open_lexical_generation(index_dir, child_generation_id) as generation:
            assert generation.manifest.corpus_version == child.corpus_version
    finally:
        repository.close()


def test_stor03_reference_and_validator_require_exact_reopenable_generation(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    outcome = build_lexical_generation(tmp_path, source_manifest)
    participant = LexicalGenerationParticipant(
        tmp_path,
        outcome,
        source_manifest,
    )
    reference = participant.component_generation()
    validator = participant.build_validator(cast(ChunkSetReader, object()))
    memberships = tuple(entry.membership for entry in source_manifest.entries)

    assert reference.component_kind.as_str() == "lexical-index"
    assert reference.state is ComponentGenerationState.PREPARING
    assert validator.validate_component_generation(reference, memberships) is True
    assert validator.validate_component_generation(reference, memberships[:-1]) is False
    assert (
        validator.validate_component_generation(
            replace(reference, corpus_id=CorpusId.generate()),
            memberships,
        )
        is False
    )
    assert (
        validator.validate_component_generation(
            replace(reference, source_manifest_digest="0" * 64),
            memberships,
        )
        is False
    )


def test_stor03_validator_rejects_checksum_corruption_without_disclosure(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    outcome = build_lexical_generation(tmp_path, source_manifest)
    participant = LexicalGenerationParticipant(tmp_path, outcome, source_manifest)
    validator = participant.build_validator(cast(ChunkSetReader, object()))
    database = tmp_path / "lexical" / outcome.manifest.generation_id / "index.sqlite3"
    with database.open("ab") as stream:
        stream.write(b"corruption")

    memberships = tuple(entry.membership for entry in source_manifest.entries)
    assert (
        validator.validate_component_generation(
            participant.component_generation(),
            memberships,
        )
        is False
    )


def test_pr3_parent_selection_uses_expected_active_lexical_reference(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    parent = build_lexical_generation(tmp_path, source_manifest)
    parent_reference = replace(
        LexicalGenerationParticipant(
            tmp_path,
            parent,
            source_manifest,
        ).component_generation(),
        state=ComponentGenerationState.READY,
    )
    active_manifest = CorpusVersionManifest(
        corpus_id=source_manifest.corpus_id,
        corpus_version=source_manifest.corpus_version,
        document_memberships=tuple(
            entry.membership for entry in source_manifest.entries
        ),
        component_generations=(parent_reference,),
    )
    repository = _FakeRepository(_FakeSnapshot(active_manifest))

    selected = select_lexical_parent_reference(
        cast(MetadataRepository, repository),
        source_manifest.corpus_id,
        source_manifest.corpus_version,
    )

    assert selected == parent_reference
    assert selected is not None
    assert selected.component_kind == ComponentKind("lexical-index")
    assert repository.open_count == 1

    child_manifest = BaselineChunkGenerationManifest(
        corpus_id=source_manifest.corpus_id,
        corpus_version=CorpusVersionId.generate(),
        configuration_identity=source_manifest.configuration_identity,
        generation_id=source_manifest.generation_id,
        entries=source_manifest.entries,
    )
    child = build_lexical_generation_for_corpus_update(
        tmp_path,
        child_manifest,
        cast(MetadataRepository, repository),
        expected_corpus_version=source_manifest.corpus_version,
    )

    assert child.statistics.database_reused is True
    assert child.statistics.reused_chunks == parent.manifest.chunk_count
    assert repository.open_count == 2


def test_pr3_parent_selection_rejects_missing_or_unexpected_active_reference(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    missing_manifest = CorpusVersionManifest(
        corpus_id=source_manifest.corpus_id,
        corpus_version=source_manifest.corpus_version,
        document_memberships=tuple(
            entry.membership for entry in source_manifest.entries
        ),
        component_generations=(),
    )
    repository = _FakeRepository(_FakeSnapshot(missing_manifest))

    with pytest.raises(LexicalError) as missing_error:
        select_lexical_parent_reference(
            cast(MetadataRepository, repository),
            source_manifest.corpus_id,
            source_manifest.corpus_version,
        )
    assert (
        missing_error.value.category is LexicalFailureCategory.INCOMPATIBLE_GENERATION
    )

    with pytest.raises(LexicalError) as mismatch_error:
        select_lexical_parent_reference(
            cast(MetadataRepository, repository),
            source_manifest.corpus_id,
            CorpusVersionId.generate(),
        )
    assert (
        mismatch_error.value.category is LexicalFailureCategory.INCOMPATIBLE_GENERATION
    )


def test_zero_chunk_member_is_retained_and_exact_replay_reuses_database(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    empty_set, empty_selection = selection("", empty=True)
    entries = tuple(
        sorted(
            source_manifest.entries
            + (
                BaselineChunkGenerationEntry(
                    membership=CorpusDocumentVersionMembership(
                        corpus_id=CORPUS_ID,
                        document_id=empty_selection.document_id,
                        document_version_id=empty_selection.document_version_id,
                    ),
                    selection=empty_selection,
                ),
            ),
            key=lambda entry: entry.key(),
        )
    )
    manifest = BaselineChunkGenerationManifest(
        corpus_id=CORPUS_ID,
        corpus_version=CORPUS_VERSION_ID,
        configuration_identity=source_manifest.configuration_identity,
        generation_id="baseline-test-generation",
        entries=entries,
    )

    first = build_lexical_generation(tmp_path, manifest)
    database = tmp_path / "lexical" / first.manifest.generation_id / "index.sqlite3"
    before = hashlib.sha256(database.read_bytes()).digest()
    second = build_lexical_generation(tmp_path, manifest)
    after = hashlib.sha256(database.read_bytes()).digest()

    assert empty_set.chunks == ()
    assert first.manifest.member_count == 3
    assert second.statistics.database_reused is True
    assert second.statistics.reused_chunks == first.manifest.chunk_count
    assert before == after


def test_corrupt_database_is_rejected_without_exposing_sqlite_details(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    outcome = build_lexical_generation(tmp_path, source_manifest)
    database = tmp_path / "lexical" / outcome.manifest.generation_id / "index.sqlite3"
    with database.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(LexicalError) as error:
        open_lexical_generation(tmp_path, outcome.manifest.generation_id)

    assert error.value.category is LexicalFailureCategory.CORRUPT_GENERATION
    assert str(error.value) == LexicalFailureCategory.CORRUPT_GENERATION.value
    assert str(tmp_path) not in str(error.value)


def test_published_database_is_read_only(tmp_path: Path) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    outcome = build_lexical_generation(tmp_path, source_manifest)
    database = tmp_path / "lexical" / outcome.manifest.generation_id / "index.sqlite3"
    connection = sqlite3.connect(f"file:{database.absolute()}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO members VALUES ('x', 'y', 0)")
    finally:
        connection.close()


def test_generation_id_cannot_escape_the_lexical_directory(tmp_path: Path) -> None:
    with pytest.raises(LexicalError) as error:
        open_lexical_generation(tmp_path, "../outside")

    assert error.value.category is LexicalFailureCategory.INVALID_REQUEST


def test_compatible_parent_diff_reuses_adds_replaces_and_removes_chunks(
    tmp_path: Path,
) -> None:
    document_a = DocumentId.from_str("00000000-0000-4000-8000-000000000011")
    document_b = DocumentId.from_str("00000000-0000-4000-8000-000000000012")
    document_c = DocumentId.from_str("00000000-0000-4000-8000-000000000013")
    document_d = DocumentId.from_str("00000000-0000-4000-8000-000000000014")
    version_a = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000021")
    version_b_old = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000022")
    version_b_new = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000023")
    version_c = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000024")
    version_d = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000025")

    unchanged, unchanged_selection = selection(
        "unchanged", document_id=document_a, document_version_id=version_a
    )
    old, old_selection = selection(
        "old version", document_id=document_b, document_version_id=version_b_old
    )
    removed, removed_selection = selection(
        "removed", document_id=document_c, document_version_id=version_c
    )
    replacement, replacement_selection = selection(
        "new version", document_id=document_b, document_version_id=version_b_new
    )
    inserted, inserted_selection = selection(
        "inserted", document_id=document_d, document_version_id=version_d
    )
    del unchanged, old, removed, replacement, inserted

    parent_manifest = manifest(
        (unchanged_selection, old_selection, removed_selection),
    )
    child_manifest = manifest(
        (unchanged_selection, replacement_selection, inserted_selection),
        corpus_version=CorpusVersionId.from_str("00000000-0000-4000-8000-000000000099"),
    )
    parent = build_lexical_generation(tmp_path, parent_manifest)
    child = build_lexical_generation(
        tmp_path,
        child_manifest,
        parent_generation_id=parent.manifest.generation_id,
    )

    assert child.statistics.database_reused is True
    assert child.statistics.members_processed == 3
    assert child.statistics.chunks_processed == 2
    assert child.statistics.reused_chunks == 1
    assert child.statistics.inserted_chunks == 1
    assert child.statistics.replaced_chunks == 1
    assert child.statistics.removed_chunks == 2
    assert child.manifest.generation_id != parent.manifest.generation_id

    with open_lexical_generation(tmp_path, child.manifest.generation_id) as generation:
        assert {chunk.content for chunk in generation.iter_chunks()} == {
            "unchanged",
            "new version",
            "inserted",
        }
        assert tuple(
            member.document_id for member in generation.member_selections()
        ) == tuple(
            sorted(
                (document_a, document_b, document_d), key=lambda value: value.as_str()
            )
        )


def test_corrupt_expected_parent_is_not_silently_rebuilt(tmp_path: Path) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    parent = build_lexical_generation(tmp_path, source_manifest)
    database = tmp_path / "lexical" / parent.manifest.generation_id / "index.sqlite3"
    with database.open("ab") as stream:
        stream.write(b"corruption")
    child_manifest = manifest(
        tuple(entry.selection for entry in source_manifest.entries),
        corpus_version=CorpusVersionId.from_str("00000000-0000-4000-8000-000000000099"),
    )

    with pytest.raises(LexicalError) as error:
        build_lexical_generation(
            tmp_path,
            child_manifest,
            parent_generation_id=parent.manifest.generation_id,
        )

    assert error.value.category is LexicalFailureCategory.CORRUPT_GENERATION


def test_parent_diff_reuses_unchanged_chunk_rows_inside_changed_member(
    tmp_path: Path,
) -> None:
    document_id = DocumentId.from_str("00000000-0000-4000-8000-000000000031")
    document_version_id = DocumentVersionId.from_str(
        "00000000-0000-4000-8000-000000000032"
    )
    policy = ChunkingPolicy(min_tokens=1, max_tokens=2)
    _, parent_selection = selection(
        "one two three four",
        document_id=document_id,
        document_version_id=document_version_id,
        policy=policy,
    )
    _, child_selection = selection(
        "one two three five",
        document_id=document_id,
        document_version_id=document_version_id,
        policy=policy,
    )
    parent = build_lexical_generation(tmp_path, manifest((parent_selection,)))
    child = build_lexical_generation(
        tmp_path,
        manifest(
            (child_selection,),
            corpus_version=CorpusVersionId.from_str(
                "00000000-0000-4000-8000-000000000099"
            ),
        ),
        parent_generation_id=parent.manifest.generation_id,
    )

    assert child.statistics.chunks_processed == 2
    assert child.statistics.reused_chunks == 1
    assert child.statistics.replaced_chunks == 1
    assert child.statistics.removed_chunks == 1
    assert child.statistics.inserted_chunks == 0
    with open_lexical_generation(tmp_path, child.manifest.generation_id) as generation:
        assert {chunk.content for chunk in generation.iter_chunks()} == {
            "one two ",
            "three five",
        }
