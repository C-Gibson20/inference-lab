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
    EmbeddingProfile,
    EmbeddingRequest,
    EmbeddingResult,
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
    LexicalGenerationParticipant,
    LexicalQuery,
    RetrievalLimit,
    RetrievalQuery,
    build_dense_generation,
    build_dense_generation_for_corpus_update,
    build_lexical_generation,
    build_lexical_generation_for_corpus_update,
    execute_dense_retrieval,
    execute_lexical_retrieval,
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

_CORPUS_A = CorpusId.from_str("00000000-0000-4000-8000-000000000011")
_CORPUS_B = CorpusId.from_str("00000000-0000-4000-8000-000000000012")
_VERSION_A = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000021")
_VERSION_B = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000022")
_UNPUBLISHED_VERSION = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000023")
_VERSION_A2 = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000024")
_VERSION_A3 = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000025")
_DOC_SHARED = DocumentId.from_str("00000000-0000-4000-8000-000000000031")
_DOC_A = DocumentId.from_str("00000000-0000-4000-8000-000000000032")
_DOC_B = DocumentId.from_str("00000000-0000-4000-8000-000000000033")
_DOC_STABLE = DocumentId.from_str("00000000-0000-4000-8000-000000000034")
_DOC_VERSION_SHARED = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000041")
_DOC_VERSION_A = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000042")
_DOC_VERSION_B = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000043")
_DOC_VERSION_STABLE = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000044")
_DOC_VERSION_A2 = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000045")
_DOC_VERSION_A3 = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000046")
_DOCUMENTS = (
    (_DOC_SHARED, _DOC_VERSION_SHARED, "common assay terminology"),
    (_DOC_A, _DOC_VERSION_A, "orchid receptor evidence"),
    (_DOC_B, _DOC_VERSION_B, "tulip catalyst evidence"),
    (_DOC_STABLE, _DOC_VERSION_STABLE, "stable control evidence"),
)


class _EmbeddingFixture:
    """Deterministic local vectors with an observable embed-work count."""

    metadata = BackendMetadata(
        backend_name="persistent-index-fixture",
        model_name="persistent-index-fixture",
        model_revision="1",
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(),
        embedding_dimensions=3,
    )
    profile = EmbeddingProfile(
        profile_version="1.0.0",
        tokenizer_id="persistent-index-tokenizer",
        tokenizer_revision="1",
        query_prefix="query: ",
        document_prefix="",
        pooling_strategy="mean",
        input_normalisation="identity",
        max_input_tokens=128,
        over_limit_policy="truncate_tail",
        l2_normalised=False,
        inference_dtype="float32",
        device_type="cpu",
    )

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    def load(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_closed(self) -> bool:
        return False

    @property
    def output_metadata(self) -> BackendMetadata:
        return self.metadata

    @property
    def output_profile(self) -> EmbeddingProfile:
        return self.profile

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.embedded_texts.extend(request.input_texts)

        def vector(text: str) -> tuple[float, float, float]:
            if "orchid" in text:
                return (1.0, 0.0, 0.0)
            if "tulip" in text:
                return (0.0, 1.0, 0.0)
            return (0.0, 0.0, 1.0)

        return EmbeddingResult(
            tuple(vector(text) for text in request.input_texts),
            self.metadata,
            self.profile,
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
        _CORPUS_A,
        document_id,
        document_version_id,
    )
    selection = _make_selection(
        _CORPUS_A,
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
            ChunkSetWriteRequest(corpus_id=_CORPUS_A, selection=selection)
        )
    repository.persist_source_artefact(
        _CORPUS_A,
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
        _CORPUS_A,
        _VERSION_A2,
        expected_corpus_version=_VERSION_A,
        required_component_kinds=(
            ComponentKind("lexical-index"),
            DENSE_COMPONENT_KIND,
        ),
    )
    transaction.replace_corpus_update_document_membership(
        _CORPUS_A,
        _VERSION_A2,
        old_membership,
        replacement,
    )
    transaction.remove_corpus_update_document_membership(
        _CORPUS_A,
        _VERSION_A2,
        removed,
    )
    transaction.add_corpus_update_document_membership(
        _CORPUS_A,
        _VERSION_A2,
        added,
    )
    transaction.register_corpus_update_component_generation(
        _CORPUS_A,
        _VERSION_A2,
        lexical.component_generation(),
        validator=lexical.build_validator(repository),
    )
    transaction.register_corpus_update_component_generation(
        _CORPUS_A,
        _VERSION_A2,
        dense.component_generation(),
        validator=(
            _RejectingValidator() if reject_dense else dense.build_validator(repository)
        ),
    )
    transaction.seal_corpus_update(_CORPUS_A, _VERSION_A2)


def _publish_fixture(
    database_path: Path,
    source_dir: Path,
    index_dir: Path,
) -> tuple[
    dict[CorpusId, CorpusVersionId],
    _EmbeddingFixture,
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
        _CORPUS_A: (_DOCUMENTS[0], _DOCUMENTS[1], _DOCUMENTS[3]),
        _CORPUS_B: (_DOCUMENTS[0], _DOCUMENTS[2]),
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
        transaction.add_corpus(CorpusMetadata(_CORPUS_A, "Shared corpus A"))
        transaction.add_corpus(CorpusMetadata(_CORPUS_B, "Shared corpus B"))
        for document_id, version_id, text in _DOCUMENTS:
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
    for document_id, version_id, text in _DOCUMENTS:
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
                corpus_id=_CORPUS_A if document_id != _DOC_B else _CORPUS_B,
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

    versions = {_CORPUS_A: _VERSION_A, _CORPUS_B: _VERSION_B}
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
    embedder = _EmbeddingFixture()
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

    cold_query = """
from pathlib import Path
import sys

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import execute_dense_retrieval, execute_lexical_retrieval
from inference_lab.retrieval.lexical import LexicalQuery
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.values import RetrievalLimit
from inference_lab.storage import open_metadata_repository
from tests.inference_lab.retrieval.test_persistent_index_systems_acceptance import _EmbeddingFixture

database_path, source_dir, index_dir = map(Path, sys.argv[1:])
repository = open_metadata_repository(database_path, source_dir=source_dir)
expected = (
    ("00000000-0000-4000-8000-000000000011", "00000000-0000-4000-8000-000000000021", "orchid", "orchid receptor evidence", "00000000-0000-4000-8000-000000000042"),
    ("00000000-0000-4000-8000-000000000011", "00000000-0000-4000-8000-000000000021", "common", "common assay terminology", "00000000-0000-4000-8000-000000000041"),
    ("00000000-0000-4000-8000-000000000012", "00000000-0000-4000-8000-000000000022", "tulip", "tulip catalyst evidence", "00000000-0000-4000-8000-000000000043"),
    ("00000000-0000-4000-8000-000000000012", "00000000-0000-4000-8000-000000000022", "common", "common assay terminology", "00000000-0000-4000-8000-000000000041"),
)
for corpus, version, term, expected_text, expected_document_version in expected:
    corpus_id = CorpusId.from_str(corpus)
    query = RetrievalQuery(term, CorpusVersionId.from_str(version), RetrievalLimit(1))
    lexical = execute_lexical_retrieval(repository, index_dir, corpus_id, LexicalQuery(query))
    dense = execute_dense_retrieval(repository, index_dir, corpus_id, query, _EmbeddingFixture())
    for result in (lexical, dense):
        assert result.status.value == "completed"
        assert tuple(hit.candidate.content for hit in result.hits) == (expected_text,)
        assert result.hits[0].candidate.sources[0].document_version_id.as_str() == expected_document_version
repository.close()
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            cold_query,
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
        with repository.open_corpus_snapshot(_CORPUS_B, _VERSION_B) as snapshot_b:
            foreign_lexical = next(
                ref
                for ref in snapshot_b.manifest.component_generations
                if ref.component_kind.as_str() == "lexical-index"
            )
        assert foreign_lexical is not None
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    _CORPUS_A,
                    _UNPUBLISHED_VERSION,
                    expected_corpus_version=_VERSION_A,
                    required_component_kinds=(ComponentKind("lexical-index"),),
                )
                transaction.register_corpus_update_component_generation(
                    _CORPUS_A,
                    _UNPUBLISHED_VERSION,
                    foreign_lexical,
                    validator=_AcceptingValidator(),
                )
        active = repository.get_active_corpus_version(_CORPUS_A)
        assert active is not None
        assert active.corpus_version == _VERSION_A
        query = RetrievalQuery("orchid", _VERSION_A, RetrievalLimit(1))
        lexical = execute_lexical_retrieval(
            repository,
            index_dir,
            _CORPUS_A,
            LexicalQuery(query),
        )
        dense = execute_dense_retrieval(
            repository,
            index_dir,
            _CORPUS_A,
            query,
            _EmbeddingFixture(),
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
            document_id=_DOC_A,
            document_version_id=_DOC_VERSION_A2,
            text=replacement_text,
            parse_records=_map_source_records(
                _CORPUS_A,
                _DOC_A,
                _DOC_VERSION_A2,
                replacement_text,
            ),
            is_new_version=True,
        )
        added, added_selection = _persist_a_catalogue_member(
            repository,
            document_id=_DOC_B,
            document_version_id=_DOC_VERSION_B,
            text=added_text,
            parse_records=parse_records_by_version[_DOC_VERSION_B],
            is_new_version=False,
        )
        stable = CorpusDocumentVersionMembership(
            _CORPUS_A,
            _DOC_STABLE,
            _DOC_VERSION_STABLE,
        )
        stable_selection = _make_selection(
            _CORPUS_A,
            _DOC_STABLE,
            _DOC_VERSION_STABLE,
            "stable control evidence",
        )
        source_manifest = _build_source_manifest(
            _CORPUS_A,
            _VERSION_A2,
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
            expected_corpus_version=_VERSION_A,
        )
        dense_outcome = build_dense_generation_for_corpus_update(
            index_dir,
            source_manifest,
            embedder,
            repository,
            expected_corpus_version=_VERSION_A,
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
            _CORPUS_A,
            _DOC_A,
            _DOC_VERSION_A,
        )
        removed = CorpusDocumentVersionMembership(
            _CORPUS_A,
            _DOC_SHARED,
            _DOC_VERSION_SHARED,
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
        active = repository.get_active_corpus_version(_CORPUS_A)
        assert active is not None
        assert active.corpus_version == _VERSION_A
        previous_query = RetrievalQuery("orchid", _VERSION_A, RetrievalLimit(1))
        previous_lexical = execute_lexical_retrieval(
            repository,
            index_dir,
            _CORPUS_A,
            LexicalQuery(previous_query),
        )
        previous_dense = execute_dense_retrieval(
            repository,
            index_dir,
            _CORPUS_A,
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
            committed = transaction.commit_corpus_update(_CORPUS_A, _VERSION_A2)
        assert committed is not None
        assert committed.version.corpus_version == _VERSION_A2
        assert {
            membership.document_version_id
            for membership in committed.manifest.document_memberships
        } == {_DOC_VERSION_A2, _DOC_VERSION_B, _DOC_VERSION_STABLE}

        expected_results = (
            ("orchid", "orchid reformulated evidence", _DOC_VERSION_A2),
            ("tulip", "tulip catalyst evidence", _DOC_VERSION_B),
            ("stable", "stable control evidence", _DOC_VERSION_STABLE),
        )
        for term, expected_text, expected_version in expected_results:
            query = RetrievalQuery(term, _VERSION_A2, RetrievalLimit(1))
            lexical_result = execute_lexical_retrieval(
                repository,
                index_dir,
                _CORPUS_A,
                LexicalQuery(query),
            )
            dense_result = execute_dense_retrieval(
                repository,
                index_dir,
                _CORPUS_A,
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
                    _CORPUS_A,
                    _VERSION_A3,
                    expected_corpus_version=_VERSION_A2,
                )
                transaction.seal_corpus_update(_CORPUS_A, _VERSION_A3)
                raise _InjectedPublicationFailure
        active = repository.get_active_corpus_version(_CORPUS_A)
        assert active is not None
        assert active.corpus_version == _VERSION_A2
        query = RetrievalQuery("orchid", _VERSION_A2, RetrievalLimit(1))
        assert (
            execute_lexical_retrieval(
                repository,
                index_dir,
                _CORPUS_A,
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
                _CORPUS_A,
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
                        _CORPUS_A,
                        "orchid",
                        ("lexical",),
                        ready_events[0],
                    ),
                    executor.submit(
                        run_reader,
                        _CORPUS_A,
                        "orchid",
                        ("dense",),
                        ready_events[1],
                    ),
                    executor.submit(
                        run_reader,
                        _CORPUS_B,
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
                        _CORPUS_A,
                        _DOC_A,
                        _DOC_VERSION_A3,
                        canary_text,
                    )
                    _persist_a_catalogue_member(
                        repository,
                        document_id=_DOC_A,
                        document_version_id=_DOC_VERSION_A3,
                        text=canary_text,
                        parse_records=_map_source_records(
                            _CORPUS_A,
                            _DOC_A,
                            _DOC_VERSION_A3,
                            canary_text,
                        ),
                        is_new_version=True,
                    )
                    shared_a = CorpusDocumentVersionMembership(
                        _CORPUS_A,
                        _DOC_SHARED,
                        _DOC_VERSION_SHARED,
                    )
                    stable_a = CorpusDocumentVersionMembership(
                        _CORPUS_A,
                        _DOC_STABLE,
                        _DOC_VERSION_STABLE,
                    )
                    source_manifest = _build_source_manifest(
                        _CORPUS_A,
                        _VERSION_A3,
                        (
                            (
                                CorpusDocumentVersionMembership(
                                    _CORPUS_A,
                                    _DOC_A,
                                    _DOC_VERSION_A3,
                                ),
                                replacement_selection,
                            ),
                            (
                                shared_a,
                                _make_selection(
                                    _CORPUS_A,
                                    _DOC_SHARED,
                                    _DOC_VERSION_SHARED,
                                    "common assay terminology",
                                ),
                            ),
                            (
                                stable_a,
                                _make_selection(
                                    _CORPUS_A,
                                    _DOC_STABLE,
                                    _DOC_VERSION_STABLE,
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
                        expected_corpus_version=_VERSION_A,
                    )
                    dense_outcome = build_dense_generation_for_corpus_update(
                        index_dir,
                        source_manifest,
                        embedder,
                        repository,
                        expected_corpus_version=_VERSION_A,
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
                            _CORPUS_A,
                            _VERSION_A3,
                            expected_corpus_version=_VERSION_A,
                            required_component_kinds=(
                                ComponentKind("lexical-index"),
                                DENSE_COMPONENT_KIND,
                            ),
                        )
                        transaction.replace_corpus_update_document_membership(
                            _CORPUS_A,
                            _VERSION_A3,
                            CorpusDocumentVersionMembership(
                                _CORPUS_A,
                                _DOC_A,
                                _DOC_VERSION_A,
                            ),
                            CorpusDocumentVersionMembership(
                                _CORPUS_A,
                                _DOC_A,
                                _DOC_VERSION_A3,
                            ),
                        )
                        transaction.register_corpus_update_component_generation(
                            _CORPUS_A,
                            _VERSION_A3,
                            lexical.component_generation(),
                            validator=lexical.build_validator(repository),
                        )
                        transaction.register_corpus_update_component_generation(
                            _CORPUS_A,
                            _VERSION_A3,
                            dense.component_generation(),
                            validator=dense.build_validator(repository),
                        )
                        transaction.seal_corpus_update(_CORPUS_A, _VERSION_A3)
                        committed = transaction.commit_corpus_update(
                            _CORPUS_A,
                            _VERSION_A3,
                        )
                    assert committed is not None
                    assert committed.version.corpus_version == _VERSION_A3

                    with repository.open_active_corpus_snapshot(
                        _CORPUS_A
                    ) as active_snapshot:
                        assert active_snapshot.version.corpus_version == _VERSION_A3
                        active_query = RetrievalQuery(
                            "orchid",
                            active_snapshot.version.corpus_version,
                            RetrievalLimit(1),
                        )
                        active_lexical = execute_lexical_retrieval(
                            repository,
                            index_dir,
                            _CORPUS_A,
                            LexicalQuery(active_query),
                            snapshot=active_snapshot,
                        )
                        active_dense = execute_dense_retrieval(
                            repository,
                            index_dir,
                            _CORPUS_A,
                            active_query,
                            embedder,
                            snapshot=active_snapshot,
                        )
                        for result in (active_lexical, active_dense):
                            assert result.status.value == "completed"
                            assert result.corpus_version == _VERSION_A3
                            assert result.hits[0].candidate.content == canary_text
                            assert (
                                result.hits[0].candidate.sources[0].document_version_id
                                == _DOC_VERSION_A3
                            )
                finally:
                    published.set()

                reader_results = tuple(future.result(timeout=30) for future in futures)
            assert canary_text not in repr(logs)

        expected_by_reader = (
            (_VERSION_A, "orchid receptor evidence", _DOC_VERSION_A),
            (_VERSION_A, "orchid receptor evidence", _DOC_VERSION_A),
            (_VERSION_B, "tulip catalyst evidence", _DOC_VERSION_B),
        )
        for (pinned_version, memberships, before, after), expected in zip(
            reader_results,
            expected_by_reader,
            strict=True,
        ):
            expected_version, expected_text, expected_document_version = expected
            assert pinned_version == expected_version
            expected_membership_versions = (
                {_DOC_VERSION_SHARED, _DOC_VERSION_A, _DOC_VERSION_STABLE}
                if expected_version == _VERSION_A
                else {_DOC_VERSION_SHARED, _DOC_VERSION_B}
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
