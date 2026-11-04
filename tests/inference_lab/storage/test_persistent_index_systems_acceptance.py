"""Cold restart, reuse, and corpus isolation tests for persistent indexes."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from structlog.testing import capture_logs

from inference_lab.backends.runtime import (
    BackendMetadata,
    BackendOperation,
)
from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    BaselineChunkGenerationEntry,
    BaselineChunkGenerationManifest,
    ChunkingPolicy,
    ChunkingRequest,
    FakeChunkTokeniser,
    MappedParseRecords,
    ParseOutcome,
    ParsePersistenceRequest,
    ParserInput,
    build_chunk_set,
    map_chunk_set,
    map_parse_persistence_request,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    RecordId,
)
from inference_lab.retrieval import (
    DENSE_COMPONENT_KIND,
    DenseGenerationParticipant,
    DenseIndexCompatibility,
    EvidenceResult,
    FailureCategory,
    LexicalGenerationParticipant,
    LexicalQuery,
    ResultStatus,
    RetrievalLimit,
    RetrievalQuery,
    build_dense_generation,
    build_dense_generation_for_corpus_update,
    build_lexical_generation,
    build_lexical_generation_for_corpus_update,
    execute_dense_retrieval,
    execute_lexical_retrieval,
)
from inference_lab.retrieval.dense_store import (
    DENSE_DATABASE_NAME,
    DENSE_DIRECTORY_NAME,
)
from inference_lab.retrieval.lexical_store_models import (
    LEXICAL_DATABASE_NAME,
    LEXICAL_DIRECTORY_NAME,
)
from inference_lab.storage import (
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    CommitCorpusUpdateOutcome,
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    MetadataRepository,
    MetadataTransaction,
    SourceArtefact,
    StorageConflictError,
    WorkerRepositoryOpener,
    build_source_blob_storage_key,
    open_metadata_repository,
)
from tests.inference_lab.retrieval.lexical_fixtures import parsed_document
from tests.inference_lab.storage.persistent_index_fixtures import (
    CORPUS_A,
    CORPUS_B,
    DOC_A,
    DOC_B,
    DOC_SHARED,
    DOC_STABLE,
    DOC_VERSION_A,
    DOC_VERSION_A2,
    DOC_VERSION_A3,
    DOC_VERSION_B,
    DOC_VERSION_SHARED,
    DOC_VERSION_STABLE,
    DOCUMENTS,
    UNPUBLISHED_VERSION,
    VERSION_A,
    VERSION_A2,
    VERSION_A3,
    VERSION_B,
    EmbeddingFixture,
)


class _AcceptingValidator:
    """Provide a valid validator when testing ownership rejection first."""

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        return True


class _RejectingValidator:
    """Reject a candidate deterministically at component registration."""

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        return False


class _InjectedPublicationFailure(RuntimeError):
    """Represent a deterministic interruption after sealing but before commit."""


def _build_source_manifest(
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    selections: tuple[
        tuple[CorpusDocumentVersionMembership, ChunkSetSelectionRecord], ...
    ],
) -> BaselineChunkGenerationManifest:
    """Bind canonical selections to one corpus and proposed version."""
    entries = tuple(
        sorted(
            (
                BaselineChunkGenerationEntry(membership, selection)
                for membership, selection in selections
            ),
            key=lambda entry: entry.key(),
        )
    )
    return BaselineChunkGenerationManifest(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        configuration_identity=entries[0].selection.configuration_identity,
        generation_id=f"source-{corpus_id.as_str()[-12:]}",
        entries=entries,
    )


def _make_selection(
    corpus_id: CorpusId,
    document_id: DocumentId,
    document_version_id: DocumentVersionId,
    text: str,
) -> ChunkSetSelectionRecord:
    """Create one deterministic chunk selection for a corpus member."""
    parsed = parsed_document(
        text,
        document_id=document_id,
        document_version_id=document_version_id,
    )
    chunk_set = build_chunk_set(
        ChunkingRequest(
            corpus_id,
            parsed,
            FakeChunkTokeniser(),
            ChunkingPolicy(min_tokens=1, max_tokens=128),
        )
    )
    return map_chunk_set(chunk_set)


def _map_source_records(
    corpus_id: CorpusId,
    document_id: DocumentId,
    document_version_id: DocumentVersionId,
    text: str,
) -> MappedParseRecords:
    """Build deterministic parser persistence values for one source version."""
    source_bytes = text.encode("utf-8")
    parsed = parsed_document(
        text,
        document_id=document_id,
        document_version_id=document_version_id,
    )
    return map_parse_persistence_request(
        ParsePersistenceRequest(
            corpus_id=corpus_id,
            parser_input=ParserInput(
                document_id,
                document_version_id,
                source_bytes,
                "text/plain",
                "utf-8",
            ),
            parser_version=CURRENT_PARSER_VERSION,
            outcome=ParseOutcome("succeeded", parsed, ()),
            parse_id=RecordId.generate(),
            parsed_record_id=RecordId.generate(),
        )
    )


def _persist_a_catalogue_member(
    repository: MetadataRepository,
    *,
    document_id: DocumentId,
    document_version_id: DocumentVersionId,
    text: str,
    parse_records: MappedParseRecords,
    is_new_version: bool,
) -> tuple[CorpusDocumentVersionMembership, ChunkSetSelectionRecord]:
    """Add a source version to corpus A's catalogue and persist its provenance."""
    source_bytes = text.encode("utf-8")
    content_hash = ContentHash(
        HashAlgorithm.SHA256,
        hashlib.sha256(source_bytes).hexdigest(),
    )
    membership = CorpusDocumentVersionMembership(
        CORPUS_A,
        document_id,
        document_version_id,
    )
    selection = _make_selection(
        CORPUS_A,
        document_id,
        document_version_id,
        text,
    )
    with repository.transaction() as transaction:
        if is_new_version:
            transaction.add_document_version(
                DocumentVersion(document_version_id, document_id, content_hash)
            )
        transaction.add_corpus_document_version_membership(membership)
        transaction.write_chunk_set(
            ChunkSetWriteRequest(corpus_id=CORPUS_A, selection=selection)
        )
    repository.persist_source_artefact(
        CORPUS_A,
        SourceArtefact(
            document_id=document_id,
            document_version_id=document_version_id,
            content_hash=content_hash,
            byte_count=len(source_bytes),
            media_type="text/plain",
            storage_key=build_source_blob_storage_key(content_hash),
        ),
        (source_bytes,),
        max_source_bytes=len(source_bytes),
        parse_outcome=parse_records.outcome,
        parsed_representation=parse_records.representation,
    )
    return membership, selection


def _stage_a_update(
    transaction: MetadataTransaction,
    *,
    old_membership: CorpusDocumentVersionMembership,
    replacement: CorpusDocumentVersionMembership,
    removed: CorpusDocumentVersionMembership,
    added: CorpusDocumentVersionMembership,
    lexical: LexicalGenerationParticipant,
    dense: DenseGenerationParticipant,
    repository: MetadataRepository,
    reject_dense: bool,
) -> None:
    """Stage the same membership delta and both indexes for a publication attempt."""
    transaction.begin_corpus_update(
        CORPUS_A,
        VERSION_A2,
        expected_corpus_version=VERSION_A,
        required_component_kinds=(
            ComponentKind("lexical-index"),
            DENSE_COMPONENT_KIND,
        ),
    )
    transaction.replace_corpus_update_document_membership(
        CORPUS_A,
        VERSION_A2,
        old_membership,
        replacement,
    )
    transaction.remove_corpus_update_document_membership(
        CORPUS_A,
        VERSION_A2,
        removed,
    )
    transaction.add_corpus_update_document_membership(
        CORPUS_A,
        VERSION_A2,
        added,
    )
    transaction.register_corpus_update_component_generation(
        CORPUS_A,
        VERSION_A2,
        lexical.component_generation(),
        validator=lexical.build_validator(repository),
    )
    transaction.register_corpus_update_component_generation(
        CORPUS_A,
        VERSION_A2,
        dense.component_generation(),
        validator=(
            _RejectingValidator() if reject_dense else dense.build_validator(repository)
        ),
    )
    transaction.seal_corpus_update(CORPUS_A, VERSION_A2)


def _publish_fixture(
    database_path: Path,
    source_dir: Path,
    index_dir: Path,
) -> tuple[
    dict[CorpusId, CorpusVersionId],
    EmbeddingFixture,
    dict[str, str],
    dict[DocumentVersionId, MappedParseRecords],
]:
    """Persist two overlapping corpora and publish both index components."""
    repository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    corpus_members = {
        CORPUS_A: (DOCUMENTS[0], DOCUMENTS[1], DOCUMENTS[3]),
        CORPUS_B: (DOCUMENTS[0], DOCUMENTS[2]),
    }
    selected: dict[tuple[CorpusId, DocumentId], ChunkSetSelectionRecord] = {}
    memberships: dict[CorpusId, tuple[CorpusDocumentVersionMembership, ...]] = {}
    for corpus_id, documents in corpus_members.items():
        built_members = []
        for document_id, version_id, text in documents:
            parsed = parsed_document(
                text,
                document_id=document_id,
                document_version_id=version_id,
            )
            chunk_set = build_chunk_set(
                ChunkingRequest(
                    corpus_id,
                    parsed,
                    FakeChunkTokeniser(),
                    ChunkingPolicy(min_tokens=1, max_tokens=128),
                )
            )
            selection = map_chunk_set(chunk_set)
            membership = CorpusDocumentVersionMembership(
                corpus_id,
                document_id,
                version_id,
            )
            selected[(corpus_id, document_id)] = selection
            built_members.append(membership)
        memberships[corpus_id] = tuple(built_members)

    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(CORPUS_A, "Shared corpus A"))
        transaction.add_corpus(CorpusMetadata(CORPUS_B, "Shared corpus B"))
        for document_id, version_id, text in DOCUMENTS:
            source_bytes = text.encode("utf-8")
            content_hash = ContentHash(
                HashAlgorithm.SHA256,
                hashlib.sha256(source_bytes).hexdigest(),
            )
            transaction.add_document(DocumentIdentity(document_id))
            transaction.add_document_version(
                DocumentVersion(version_id, document_id, content_hash)
            )
        for corpus_id, corpus_memberships in memberships.items():
            for membership in corpus_memberships:
                transaction.add_corpus_document_version_membership(membership)
                selection = selected[(corpus_id, membership.document_id)]
                transaction.write_chunk_set(
                    ChunkSetWriteRequest(corpus_id=corpus_id, selection=selection)
                )

    parse_records_by_version: dict[DocumentVersionId, MappedParseRecords] = {}
    for document_id, version_id, text in DOCUMENTS:
        source_bytes = text.encode("utf-8")
        content_hash = ContentHash(
            HashAlgorithm.SHA256,
            hashlib.sha256(source_bytes).hexdigest(),
        )
        parsed = parsed_document(
            text,
            document_id=document_id,
            document_version_id=version_id,
        )
        parse_records = map_parse_persistence_request(
            ParsePersistenceRequest(
                corpus_id=CORPUS_A if document_id != DOC_B else CORPUS_B,
                parser_input=ParserInput(
                    document_id,
                    version_id,
                    source_bytes,
                    "text/plain",
                    "utf-8",
                ),
                parser_version=CURRENT_PARSER_VERSION,
                outcome=ParseOutcome("succeeded", parsed, ()),
                parse_id=RecordId.generate(),
                parsed_record_id=RecordId.generate(),
            )
        )
        parse_records_by_version[version_id] = parse_records
        for corpus_id in corpus_members:
            if any(m.document_version_id == version_id for m in memberships[corpus_id]):
                repository.persist_source_artefact(
                    corpus_id,
                    SourceArtefact(
                        document_id=document_id,
                        document_version_id=version_id,
                        content_hash=content_hash,
                        byte_count=len(source_bytes),
                        media_type="text/plain",
                        storage_key=build_source_blob_storage_key(content_hash),
                    ),
                    (source_bytes,),
                    max_source_bytes=len(source_bytes),
                    parse_outcome=parse_records.outcome,
                    parsed_representation=parse_records.representation,
                )

    versions = {CORPUS_A: VERSION_A, CORPUS_B: VERSION_B}
    manifests = {
        corpus_id: _build_source_manifest(
            corpus_id,
            version,
            tuple(
                (membership, selected[(corpus_id, membership.document_id)])
                for membership in memberships[corpus_id]
            ),
        )
        for corpus_id, version in versions.items()
    }
    embedder = EmbeddingFixture()
    compatibility = DenseIndexCompatibility(
        model_metadata=embedder.metadata,
        dimensions=3,
        embedding_profile=embedder.profile,
    )
    generation_ids: dict[str, str] = {}
    for corpus_id, manifest in manifests.items():
        lexical = build_lexical_generation(index_dir, manifest)
        dense = build_dense_generation(
            index_dir,
            manifest,
            embedder,
            metadata_db_path=database_path,
            source_dir=source_dir,
            compatibility=compatibility,
        )
        participants = (
            LexicalGenerationParticipant(index_dir, lexical, manifest),
            DenseGenerationParticipant(index_dir, dense, manifest),
        )
        committed: CommitCorpusUpdateOutcome | None = None
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                corpus_id,
                versions[corpus_id],
                expected_corpus_version=None,
                required_component_kinds=(
                    ComponentKind("lexical-index"),
                    DENSE_COMPONENT_KIND,
                ),
            )
            for membership in memberships[corpus_id]:
                transaction.add_corpus_update_document_membership(
                    corpus_id,
                    versions[corpus_id],
                    membership,
                )
            for participant in participants:
                transaction.register_corpus_update_component_generation(
                    corpus_id,
                    versions[corpus_id],
                    participant.component_generation(),
                    validator=participant.build_validator(repository),
                )
            transaction.seal_corpus_update(corpus_id, versions[corpus_id])
            committed = transaction.commit_corpus_update(corpus_id, versions[corpus_id])
        assert committed is not None
        for reference in committed.manifest.component_generations:
            generation_ids[
                f"{corpus_id.as_str()}:{reference.component_kind.as_str()}"
            ] = reference.generation_id

    # Same complete build requests must reopen the registered immutable data.
    previous_embedding_work = tuple(embedder.embedded_texts)
    for manifest in manifests.values():
        lexical_replay = build_lexical_generation(index_dir, manifest)
        dense_replay = build_dense_generation(
            index_dir,
            manifest,
            embedder,
            metadata_db_path=database_path,
            source_dir=source_dir,
            compatibility=compatibility,
        )
        assert lexical_replay.statistics.database_reused
        assert dense_replay.statistics.generation_reused
    assert tuple(embedder.embedded_texts) == previous_embedding_work
    repository.close()
    return versions, embedder, generation_ids, parse_records_by_version


def test_persistent_indexes_reopen_reuse_and_isolate_corpora(
    tmp_path: Path,
) -> None:
    """Cold queries use both persisted channels without crossing membership."""
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    versions, _, generation_ids, _ = _publish_fixture(
        database_path,
        source_dir,
        index_dir,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.inference_lab.storage.cold_query",
            str(database_path),
            str(source_dir),
            str(index_dir),
        ],
        check=False,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    # The committed manifest, not a mutable index path, names both durable
    # generations for each corpus.
    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:
        for corpus_id, corpus_version in versions.items():
            with repository.open_corpus_snapshot(corpus_id, corpus_version) as snapshot:
                refs = {
                    ref.component_kind.as_str(): ref
                    for ref in snapshot.manifest.component_generations
                }
                assert (
                    refs["lexical-index"].generation_id
                    == generation_ids[f"{corpus_id.as_str()}:lexical-index"]
                )
                assert (
                    refs["dense-index"].generation_id
                    == generation_ids[f"{corpus_id.as_str()}:dense-index"]
                )
                assert all(
                    ref.state is ComponentGenerationState.READY for ref in refs.values()
                )
        foreign_lexical: ComponentGenerationReference | None = None
        with repository.open_corpus_snapshot(CORPUS_B, VERSION_B) as snapshot_b:
            foreign_lexical = next(
                ref
                for ref in snapshot_b.manifest.component_generations
                if ref.component_kind.as_str() == "lexical-index"
            )
        assert foreign_lexical is not None
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    CORPUS_A,
                    UNPUBLISHED_VERSION,
                    expected_corpus_version=VERSION_A,
                    required_component_kinds=(ComponentKind("lexical-index"),),
                )
                transaction.register_corpus_update_component_generation(
                    CORPUS_A,
                    UNPUBLISHED_VERSION,
                    foreign_lexical,
                    validator=_AcceptingValidator(),
                )
        active = repository.get_active_corpus_version(CORPUS_A)
        assert active is not None
        assert active.corpus_version == VERSION_A
        query = RetrievalQuery("orchid", VERSION_A, RetrievalLimit(1))
        lexical = execute_lexical_retrieval(
            repository,
            index_dir,
            CORPUS_A,
            LexicalQuery(query),
        )
        dense = execute_dense_retrieval(
            repository,
            index_dir,
            CORPUS_A,
            query,
            EmbeddingFixture(),
        )
        assert tuple(hit.candidate.content for hit in lexical.hits) == (
            "orchid receptor evidence",
        )
        assert tuple(hit.candidate.content for hit in dense.hits) == (
            "orchid receptor evidence",
        )
    finally:
        repository.close()


def test_incremental_update_reuses_unchanged_work_and_keeps_active_version_on_failure(
    tmp_path: Path,
) -> None:
    """Add, replacement, removal, and failed publication preserve one valid view."""
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    versions, embedder, _, parse_records_by_version = _publish_fixture(
        database_path,
        source_dir,
        index_dir,
    )
    del versions
    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:
        replacement_text = "orchid reformulated evidence"
        added_text = "tulip catalyst evidence"
        replacement, replacement_selection = _persist_a_catalogue_member(
            repository,
            document_id=DOC_A,
            document_version_id=DOC_VERSION_A2,
            text=replacement_text,
            parse_records=_map_source_records(
                CORPUS_A,
                DOC_A,
                DOC_VERSION_A2,
                replacement_text,
            ),
            is_new_version=True,
        )
        added, added_selection = _persist_a_catalogue_member(
            repository,
            document_id=DOC_B,
            document_version_id=DOC_VERSION_B,
            text=added_text,
            parse_records=parse_records_by_version[DOC_VERSION_B],
            is_new_version=False,
        )
        stable = CorpusDocumentVersionMembership(
            CORPUS_A,
            DOC_STABLE,
            DOC_VERSION_STABLE,
        )
        stable_selection = _make_selection(
            CORPUS_A,
            DOC_STABLE,
            DOC_VERSION_STABLE,
            "stable control evidence",
        )
        source_manifest = _build_source_manifest(
            CORPUS_A,
            VERSION_A2,
            (
                (replacement, replacement_selection),
                (added, added_selection),
                (stable, stable_selection),
            ),
        )
        compatibility = DenseIndexCompatibility(
            model_metadata=embedder.metadata,
            dimensions=3,
            embedding_profile=embedder.profile,
        )
        embedded_before = len(embedder.embedded_texts)
        lexical_outcome = build_lexical_generation_for_corpus_update(
            index_dir,
            source_manifest,
            repository,
            expected_corpus_version=VERSION_A,
        )
        dense_outcome = build_dense_generation_for_corpus_update(
            index_dir,
            source_manifest,
            embedder,
            repository,
            expected_corpus_version=VERSION_A,
            metadata_db_path=database_path,
            source_dir=source_dir,
            compatibility=compatibility,
        )
        assert lexical_outcome.statistics.database_reused
        assert lexical_outcome.statistics.reused_chunks == 1
        assert dense_outcome.statistics.reused_vectors == 1
        assert dense_outcome.statistics.embedded_vectors == 2
        assert len(embedder.embedded_texts) - embedded_before == 2

        lexical = LexicalGenerationParticipant(
            index_dir,
            lexical_outcome,
            source_manifest,
        )
        dense = DenseGenerationParticipant(index_dir, dense_outcome, source_manifest)
        old_membership = CorpusDocumentVersionMembership(
            CORPUS_A,
            DOC_A,
            DOC_VERSION_A,
        )
        removed = CorpusDocumentVersionMembership(
            CORPUS_A,
            DOC_SHARED,
            DOC_VERSION_SHARED,
        )

        # Reject the second component after the first one has registered. The
        # failed transaction must leave the old active pair intact.
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                _stage_a_update(
                    transaction,
                    old_membership=old_membership,
                    replacement=replacement,
                    removed=removed,
                    added=added,
                    lexical=lexical,
                    dense=dense,
                    repository=repository,
                    reject_dense=True,
                )
        active = repository.get_active_corpus_version(CORPUS_A)
        assert active is not None
        assert active.corpus_version == VERSION_A
        previous_query = RetrievalQuery("orchid", VERSION_A, RetrievalLimit(1))
        previous_lexical = execute_lexical_retrieval(
            repository,
            index_dir,
            CORPUS_A,
            LexicalQuery(previous_query),
        )
        previous_dense = execute_dense_retrieval(
            repository,
            index_dir,
            CORPUS_A,
            previous_query,
            embedder,
        )
        assert tuple(hit.candidate.content for hit in previous_lexical.hits) == (
            "orchid receptor evidence",
        )
        assert tuple(hit.candidate.content for hit in previous_dense.hits) == (
            "orchid receptor evidence",
        )

        committed: CommitCorpusUpdateOutcome | None = None
        with repository.transaction() as transaction:
            _stage_a_update(
                transaction,
                old_membership=old_membership,
                replacement=replacement,
                removed=removed,
                added=added,
                lexical=lexical,
                dense=dense,
                repository=repository,
                reject_dense=False,
            )
            committed = transaction.commit_corpus_update(CORPUS_A, VERSION_A2)
        assert committed is not None
        assert committed.version.corpus_version == VERSION_A2
        assert {
            membership.document_version_id
            for membership in committed.manifest.document_memberships
        } == {DOC_VERSION_A2, DOC_VERSION_B, DOC_VERSION_STABLE}

        expected_results = (
            ("orchid", "orchid reformulated evidence", DOC_VERSION_A2),
            ("tulip", "tulip catalyst evidence", DOC_VERSION_B),
            ("stable", "stable control evidence", DOC_VERSION_STABLE),
        )
        for term, expected_text, expected_version in expected_results:
            query = RetrievalQuery(term, VERSION_A2, RetrievalLimit(1))
            lexical_result = execute_lexical_retrieval(
                repository,
                index_dir,
                CORPUS_A,
                LexicalQuery(query),
            )
            dense_result = execute_dense_retrieval(
                repository,
                index_dir,
                CORPUS_A,
                query,
                embedder,
            )
            for result in (lexical_result, dense_result):
                assert result.status.value == "completed"
                assert tuple(hit.candidate.content for hit in result.hits) == (
                    expected_text,
                )
                assert (
                    result.hits[0].candidate.sources[0].document_version_id
                    == expected_version
                )

        # A deterministic failure after sealing but before commit must not
        # advance the active pointer or replace its usable component pair.
        with pytest.raises(_InjectedPublicationFailure):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    CORPUS_A,
                    VERSION_A3,
                    expected_corpus_version=VERSION_A2,
                )
                transaction.seal_corpus_update(CORPUS_A, VERSION_A3)
                raise _InjectedPublicationFailure
        active = repository.get_active_corpus_version(CORPUS_A)
        assert active is not None
        assert active.corpus_version == VERSION_A2
        query = RetrievalQuery("orchid", VERSION_A2, RetrievalLimit(1))
        assert (
            execute_lexical_retrieval(
                repository,
                index_dir,
                CORPUS_A,
                LexicalQuery(query),
            )
            .hits[0]
            .candidate.content
            == "orchid reformulated evidence"
        )
        assert (
            execute_dense_retrieval(
                repository,
                index_dir,
                CORPUS_A,
                query,
                embedder,
            )
            .hits[0]
            .candidate.content
            == "orchid reformulated evidence"
        )
    finally:
        repository.close()


def test_pinned_readers_remain_coherent_during_publication(
    tmp_path: Path,
) -> None:
    """Pinned readers stay on one version while active readers switch together."""
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    _, embedder, _, _ = _publish_fixture(
        database_path,
        source_dir,
        index_dir,
    )
    repository = open_metadata_repository(database_path, source_dir=source_dir)
    assert isinstance(repository, WorkerRepositoryOpener)
    canary_text = "orchid confidential evidence token aurora9"
    ready_events = (Event(), Event(), Event())
    published = Event()
    reader_errors: list[BaseException] = []

    def read_channels(
        corpus_id: CorpusId,
        term: str,
        channels: tuple[str, ...],
        ready_event: Event,
    ) -> tuple[
        CorpusVersionId,
        tuple[CorpusDocumentVersionMembership, ...],
        tuple[EvidenceResult, ...],
        tuple[EvidenceResult, ...],
    ]:
        reader_repository = repository.open_worker_repository()
        try:
            with reader_repository.open_active_corpus_snapshot(corpus_id) as snapshot:
                pinned_version = snapshot.version.corpus_version

                def read(channel: str) -> EvidenceResult:
                    query = RetrievalQuery(term, pinned_version, RetrievalLimit(1))
                    if channel == "lexical":
                        return execute_lexical_retrieval(
                            reader_repository,
                            index_dir,
                            corpus_id,
                            LexicalQuery(query),
                            snapshot=snapshot,
                        )
                    return execute_dense_retrieval(
                        reader_repository,
                        index_dir,
                        corpus_id,
                        query,
                        embedder,
                        snapshot=snapshot,
                    )

                try:
                    before = tuple(read(channel) for channel in channels)
                finally:
                    ready_event.set()
                if not published.wait(timeout=30):
                    raise TimeoutError("publication did not release pinned readers")
                after = tuple(read(channel) for channel in channels)
                return (
                    pinned_version,
                    snapshot.manifest.document_memberships,
                    before,
                    after,
                )
        finally:
            reader_repository.close()
        raise AssertionError("snapshot context exited without a reader result")

    def run_reader(
        corpus_id: CorpusId,
        term: str,
        channels: tuple[str, ...],
        ready_event: Event,
    ) -> tuple[
        CorpusVersionId,
        tuple[CorpusDocumentVersionMembership, ...],
        tuple[EvidenceResult, ...],
        tuple[EvidenceResult, ...],
    ]:
        try:
            return read_channels(corpus_id, term, channels, ready_event)
        except BaseException as error:
            reader_errors.append(error)
            ready_event.set()
            raise

    try:
        with capture_logs() as logs:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = (
                    executor.submit(
                        run_reader,
                        CORPUS_A,
                        "orchid",
                        ("lexical",),
                        ready_events[0],
                    ),
                    executor.submit(
                        run_reader,
                        CORPUS_A,
                        "orchid",
                        ("dense",),
                        ready_events[1],
                    ),
                    executor.submit(
                        run_reader,
                        CORPUS_B,
                        "tulip",
                        ("lexical", "dense"),
                        ready_events[2],
                    ),
                )
                for ready_event in ready_events:
                    if not ready_event.wait(timeout=30):
                        raise TimeoutError("pinned readers did not start")
                if reader_errors:
                    raise reader_errors[0]
                for future in futures:
                    if future.done():
                        future.result()
                try:
                    replacement_selection = _make_selection(
                        CORPUS_A,
                        DOC_A,
                        DOC_VERSION_A3,
                        canary_text,
                    )
                    _persist_a_catalogue_member(
                        repository,
                        document_id=DOC_A,
                        document_version_id=DOC_VERSION_A3,
                        text=canary_text,
                        parse_records=_map_source_records(
                            CORPUS_A,
                            DOC_A,
                            DOC_VERSION_A3,
                            canary_text,
                        ),
                        is_new_version=True,
                    )
                    shared_a = CorpusDocumentVersionMembership(
                        CORPUS_A,
                        DOC_SHARED,
                        DOC_VERSION_SHARED,
                    )
                    stable_a = CorpusDocumentVersionMembership(
                        CORPUS_A,
                        DOC_STABLE,
                        DOC_VERSION_STABLE,
                    )
                    source_manifest = _build_source_manifest(
                        CORPUS_A,
                        VERSION_A3,
                        (
                            (
                                CorpusDocumentVersionMembership(
                                    CORPUS_A,
                                    DOC_A,
                                    DOC_VERSION_A3,
                                ),
                                replacement_selection,
                            ),
                            (
                                shared_a,
                                _make_selection(
                                    CORPUS_A,
                                    DOC_SHARED,
                                    DOC_VERSION_SHARED,
                                    "common assay terminology",
                                ),
                            ),
                            (
                                stable_a,
                                _make_selection(
                                    CORPUS_A,
                                    DOC_STABLE,
                                    DOC_VERSION_STABLE,
                                    "stable control evidence",
                                ),
                            ),
                        ),
                    )
                    compatibility = DenseIndexCompatibility(
                        model_metadata=embedder.metadata,
                        dimensions=3,
                        embedding_profile=embedder.profile,
                    )
                    lexical_outcome = build_lexical_generation_for_corpus_update(
                        index_dir,
                        source_manifest,
                        repository,
                        expected_corpus_version=VERSION_A,
                    )
                    dense_outcome = build_dense_generation_for_corpus_update(
                        index_dir,
                        source_manifest,
                        embedder,
                        repository,
                        expected_corpus_version=VERSION_A,
                        metadata_db_path=database_path,
                        source_dir=source_dir,
                        compatibility=compatibility,
                    )
                    lexical = LexicalGenerationParticipant(
                        index_dir,
                        lexical_outcome,
                        source_manifest,
                    )
                    dense = DenseGenerationParticipant(
                        index_dir,
                        dense_outcome,
                        source_manifest,
                    )
                    committed: CommitCorpusUpdateOutcome | None = None
                    with repository.transaction() as transaction:
                        transaction.begin_corpus_update(
                            CORPUS_A,
                            VERSION_A3,
                            expected_corpus_version=VERSION_A,
                            required_component_kinds=(
                                ComponentKind("lexical-index"),
                                DENSE_COMPONENT_KIND,
                            ),
                        )
                        transaction.replace_corpus_update_document_membership(
                            CORPUS_A,
                            VERSION_A3,
                            CorpusDocumentVersionMembership(
                                CORPUS_A,
                                DOC_A,
                                DOC_VERSION_A,
                            ),
                            CorpusDocumentVersionMembership(
                                CORPUS_A,
                                DOC_A,
                                DOC_VERSION_A3,
                            ),
                        )
                        transaction.register_corpus_update_component_generation(
                            CORPUS_A,
                            VERSION_A3,
                            lexical.component_generation(),
                            validator=lexical.build_validator(repository),
                        )
                        transaction.register_corpus_update_component_generation(
                            CORPUS_A,
                            VERSION_A3,
                            dense.component_generation(),
                            validator=dense.build_validator(repository),
                        )
                        transaction.seal_corpus_update(CORPUS_A, VERSION_A3)
                        committed = transaction.commit_corpus_update(
                            CORPUS_A,
                            VERSION_A3,
                        )
                    assert committed is not None
                    assert committed.version.corpus_version == VERSION_A3

                    with repository.open_active_corpus_snapshot(
                        CORPUS_A
                    ) as active_snapshot:
                        assert active_snapshot.version.corpus_version == VERSION_A3
                        active_query = RetrievalQuery(
                            "orchid",
                            active_snapshot.version.corpus_version,
                            RetrievalLimit(1),
                        )
                        active_lexical = execute_lexical_retrieval(
                            repository,
                            index_dir,
                            CORPUS_A,
                            LexicalQuery(active_query),
                            snapshot=active_snapshot,
                        )
                        active_dense = execute_dense_retrieval(
                            repository,
                            index_dir,
                            CORPUS_A,
                            active_query,
                            embedder,
                            snapshot=active_snapshot,
                        )
                        for result in (active_lexical, active_dense):
                            assert result.status.value == "completed"
                            assert result.corpus_version == VERSION_A3
                            assert result.hits[0].candidate.content == canary_text
                            assert (
                                result.hits[0].candidate.sources[0].document_version_id
                                == DOC_VERSION_A3
                            )
                finally:
                    published.set()

                reader_results = tuple(future.result(timeout=30) for future in futures)
            assert canary_text not in repr(logs)

        expected_by_reader = (
            (VERSION_A, "orchid receptor evidence", DOC_VERSION_A),
            (VERSION_A, "orchid receptor evidence", DOC_VERSION_A),
            (VERSION_B, "tulip catalyst evidence", DOC_VERSION_B),
        )
        for (pinned_version, memberships, before, after), expected in zip(
            reader_results,
            expected_by_reader,
            strict=True,
        ):
            expected_version, expected_text, expected_document_version = expected
            assert pinned_version == expected_version
            expected_membership_versions = (
                {DOC_VERSION_SHARED, DOC_VERSION_A, DOC_VERSION_STABLE}
                if expected_version == VERSION_A
                else {DOC_VERSION_SHARED, DOC_VERSION_B}
            )
            assert {
                membership.document_version_id for membership in memberships
            } == expected_membership_versions
            for result in (*before, *after):
                assert result.status.value == "completed"
                assert result.corpus_version == expected_version
                assert result.hits[0].candidate.content == expected_text
                assert (
                    result.hits[0].candidate.sources[0].document_version_id
                    == expected_document_version
                )
    finally:
        published.set()
        repository.close()


class _IncompatibleEmbeddingFixture(EmbeddingFixture):
    """Return valid vectors under a different pinned model revision."""

    metadata = BackendMetadata(
        backend_name="persistent-index-fixture",
        model_name="persistent-index-fixture",
        model_revision="2",
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(),
        embedding_dimensions=3,
    )


def _truncate(path: Path) -> None:
    """Damage a published component file without removing it."""
    assert path.is_file()
    path.write_bytes(b"")


def test_corrupt_or_incompatible_components_fail_closed_without_losing_the_view(
    tmp_path: Path,
) -> None:
    """One damaged component fails closed while the rest of the view still serves."""
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    versions, _, generation_ids, _ = _publish_fixture(
        database_path,
        source_dir,
        index_dir,
    )

    def lexical(corpus_id: CorpusId, term: str) -> EvidenceResult:
        query = RetrievalQuery(term, versions[corpus_id], RetrievalLimit(5))
        return execute_lexical_retrieval(
            repository, index_dir, corpus_id, LexicalQuery(query)
        )

    def dense(
        corpus_id: CorpusId,
        term: str,
        embedder: EmbeddingFixture | None = None,
    ) -> EvidenceResult:
        query = RetrievalQuery(term, versions[corpus_id], RetrievalLimit(5))
        return execute_dense_retrieval(
            repository,
            index_dir,
            corpus_id,
            query,
            embedder or EmbeddingFixture(),
        )

    def categories(result: EvidenceResult) -> set[FailureCategory]:
        return {failure.category for failure in result.failures}

    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:
        # Both channels serve both corpora before anything is damaged.
        for corpus_id in (CORPUS_A, CORPUS_B):
            assert lexical(corpus_id, "common").status is ResultStatus.COMPLETED
            assert dense(corpus_id, "common").hits

        # An embedder that no longer matches the published compatibility must
        # fail closed rather than return vectors from a different model.
        incompatible = dense(CORPUS_A, "common", _IncompatibleEmbeddingFixture())
        assert incompatible.status is ResultStatus.FAILED
        assert not incompatible.hits
        assert categories(incompatible) == {FailureCategory.INVALID_CHANNEL_RESPONSE}
        assert lexical(CORPUS_A, "common").hits, (
            "an incompatible embedder must not disturb the lexical channel"
        )

        # Corrupting one corpus's dense database contains the failure: that
        # channel fails closed, its lexical channel still serves, and the other
        # corpus is untouched.
        _truncate(
            index_dir
            / DENSE_DIRECTORY_NAME
            / generation_ids[f"{CORPUS_A.as_str()}:{DENSE_COMPONENT_KIND.as_str()}"]
            / DENSE_DATABASE_NAME
        )
        corrupt_dense = dense(CORPUS_A, "common")
        assert corrupt_dense.status is ResultStatus.FAILED
        assert not corrupt_dense.hits
        assert categories(corrupt_dense) == {FailureCategory.CHANNEL_UNAVAILABLE}
        assert lexical(CORPUS_A, "common").hits
        assert dense(CORPUS_B, "common").hits

        # Corrupting the lexical database of the same corpus leaves the other
        # corpus serving both channels and the metadata view intact.
        _truncate(
            index_dir
            / LEXICAL_DIRECTORY_NAME
            / generation_ids[f"{CORPUS_A.as_str()}:lexical-index"]
            / LEXICAL_DATABASE_NAME
        )
        corrupt_lexical = lexical(CORPUS_A, "common")
        assert corrupt_lexical.status is ResultStatus.FAILED
        assert not corrupt_lexical.hits
        assert categories(corrupt_lexical) == {FailureCategory.CHANNEL_UNAVAILABLE}
        assert lexical(CORPUS_B, "common").hits
        assert dense(CORPUS_B, "common").hits

        # Damaged index files never invalidate the published corpus version.
        with repository.open_corpus_snapshot(CORPUS_A, versions[CORPUS_A]) as snapshot:
            assert len(snapshot.manifest.document_memberships) == 3
            assert snapshot.version.corpus_version == versions[CORPUS_A]
    finally:
        repository.close()


def _staged_generations(
    repository: MetadataRepository,
) -> dict[str, ComponentGenerationReference]:
    """Read the staged component generations of corpus A's open proposal."""
    return {
        reference.component_kind.as_str(): reference
        for reference in repository.list_corpus_update_component_generations(
            CORPUS_A, VERSION_A2
        )
    }


def test_superseded_preparing_generation_is_replaced_before_sealing(
    tmp_path: Path,
) -> None:
    """A rebuilt component replaces its staged predecessor under compare-and-set."""
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    versions, _, generation_ids, _ = _publish_fixture(
        database_path,
        source_dir,
        index_dir,
    )

    # The proposal removes one inherited member, so the sealed manifest is a
    # genuine change rather than a no-op republication of the active version.
    members = tuple(
        (
            CorpusDocumentVersionMembership(CORPUS_A, document_id, version_id),
            _make_selection(CORPUS_A, document_id, version_id, text),
        )
        for document_id, version_id, text in (DOCUMENTS[0], DOCUMENTS[1])
    )
    removed = CorpusDocumentVersionMembership(
        CORPUS_A, DOCUMENTS[3][0], DOCUMENTS[3][1]
    )
    # The superseded candidate was built before the last member was staged, so
    # it carries a different source-manifest digest and generation identity.
    superseded_manifest = _build_source_manifest(CORPUS_A, VERSION_A2, members[:1])
    complete_manifest = _build_source_manifest(CORPUS_A, VERSION_A2, members)
    assert (
        superseded_manifest.source_manifest_digest
        != complete_manifest.source_manifest_digest
    )

    embedder = EmbeddingFixture()
    compatibility = DenseIndexCompatibility(
        model_metadata=embedder.metadata,
        dimensions=3,
        embedding_profile=embedder.profile,
    )
    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:

        def dense_participant(
            manifest: BaselineChunkGenerationManifest,
        ) -> DenseGenerationParticipant:
            return DenseGenerationParticipant(
                index_dir,
                build_dense_generation(
                    index_dir,
                    manifest,
                    embedder,
                    metadata_db_path=database_path,
                    source_dir=source_dir,
                    compatibility=compatibility,
                ),
                manifest,
            )

        superseded = dense_participant(superseded_manifest)
        complete = dense_participant(complete_manifest)
        lexical = LexicalGenerationParticipant(
            index_dir,
            build_lexical_generation(index_dir, complete_manifest),
            complete_manifest,
        )
        assert (
            superseded.component_generation().generation_id
            != complete.component_generation().generation_id
        )

        # Staging is its own unit of work. A failed operation makes a
        # transaction rollback-only, so each rejected replacement below runs in
        # a separate transaction and must leave the staged update usable.
        staged: ComponentGenerationReference | None = None
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                CORPUS_A,
                VERSION_A2,
                expected_corpus_version=versions[CORPUS_A],
                required_component_kinds=(
                    ComponentKind("lexical-index"),
                    DENSE_COMPONENT_KIND,
                ),
            )
            transaction.remove_corpus_update_document_membership(
                CORPUS_A, VERSION_A2, removed
            )
            transaction.register_corpus_update_component_generation(
                CORPUS_A,
                VERSION_A2,
                lexical.component_generation(),
                validator=lexical.build_validator(repository),
            )
            staged = transaction.register_corpus_update_component_generation(
                CORPUS_A,
                VERSION_A2,
                superseded.component_generation(),
                validator=_AcceptingValidator(),
            )
        assert staged is not None
        baseline = _staged_generations(repository)
        assert baseline[DENSE_COMPONENT_KIND.as_str()] == staged

        # Replacing against an expectation that is not the staged reference is a
        # conflict, so one writer cannot overwrite another writer's candidate.
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.replace_corpus_update_component_generation(
                    CORPUS_A,
                    VERSION_A2,
                    complete.component_generation(),
                    complete.component_generation(),
                    validator=complete.build_validator(repository),
                )

        # A rejected replacement must leave the staged candidates untouched.
        assert _staged_generations(repository) == baseline, (
            "a stale expected reference must not partially mutate staged component state"
        )

        # A replacement must keep the component kind it supersedes.
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.replace_corpus_update_component_generation(
                    CORPUS_A,
                    VERSION_A2,
                    staged,
                    lexical.component_generation(),
                    validator=lexical.build_validator(repository),
                )

        # A rejected replacement must leave the staged candidates untouched.
        assert _staged_generations(repository) == baseline, (
            "a mismatched component kind must not partially mutate staged component state"
        )

        committed: CommitCorpusUpdateOutcome | None = None
        with repository.transaction() as transaction:
            replaced = transaction.replace_corpus_update_component_generation(
                CORPUS_A,
                VERSION_A2,
                staged,
                complete.component_generation(),
                validator=complete.build_validator(repository),
            )
            assert replaced.state is ComponentGenerationState.READY
            assert (
                replaced.generation_id == complete.component_generation().generation_id
            )
            # Exact replay of the same replacement is a no-op.
            assert (
                transaction.replace_corpus_update_component_generation(
                    CORPUS_A,
                    VERSION_A2,
                    staged,
                    complete.component_generation(),
                    validator=complete.build_validator(repository),
                )
                == replaced
            )
            transaction.seal_corpus_update(CORPUS_A, VERSION_A2)
            committed = transaction.commit_corpus_update(CORPUS_A, VERSION_A2)

        assert committed is not None
        published = {
            reference.component_kind.as_str(): reference.generation_id
            for reference in committed.manifest.component_generations
        }
        assert (
            published[DENSE_COMPONENT_KIND.as_str()]
            == complete.component_generation().generation_id
        ), "the sealed version must name the replacement, not its predecessor"
        assert (
            published[DENSE_COMPONENT_KIND.as_str()]
            != superseded.component_generation().generation_id
        )
        assert (
            published[DENSE_COMPONENT_KIND.as_str()]
            != generation_ids[f"{CORPUS_A.as_str()}:{DENSE_COMPONENT_KIND.as_str()}"]
        )

        # The replacement is the generation that actually serves queries.
        query = RetrievalQuery("orchid", VERSION_A2, RetrievalLimit(5))
        result = execute_dense_retrieval(
            repository, index_dir, CORPUS_A, query, EmbeddingFixture()
        )
        assert result.status is ResultStatus.COMPLETED
        assert result.hits
    finally:
        repository.close()
