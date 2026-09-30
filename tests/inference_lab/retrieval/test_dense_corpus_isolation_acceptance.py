"""RET-04 two-corpus publication, restart, and generation-isolation acceptance."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

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
    ParsePersistenceRequest,
    ParserInput,
    build_chunk_set,
    map_chunk_set,
    map_parse_persistence_request,
)
from inference_lab.ingest import (
    ParseOutcome as IngestParseOutcome,
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
    execute_dense_retrieval,
)
from inference_lab.retrieval.dense_store import open_dense_generation
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.results import FailureCategory, ResultStatus
from inference_lab.retrieval.values import RetrievalLimit
from inference_lab.storage import (
    ChunkSetWriteRequest,
    CommitCorpusUpdateOutcome,
    SourceArtefact,
    build_source_blob_storage_key,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.retrieval.lexical_fixtures import parsed_document

_CORPORA = (
    (
        CorpusId.from_str("00000000-0000-4000-8000-000000000001"),
        CorpusVersionId.from_str("00000000-0000-4000-8000-000000000201"),
        DocumentId.from_str("00000000-0000-4000-8000-000000000211"),
        DocumentVersionId.from_str("00000000-0000-4000-8000-000000000221"),
        "alpha passage",
    ),
    (
        CorpusId.from_str("00000000-0000-4000-8000-000000000003"),
        CorpusVersionId.from_str("00000000-0000-4000-8000-000000000203"),
        DocumentId.from_str("00000000-0000-4000-8000-000000000241"),
        DocumentVersionId.from_str("00000000-0000-4000-8000-000000000251"),
        "gamma passage",
    ),
)


class _EmbeddingFixture:
    """Small deterministic runtime fixture shared by both published corpora."""

    metadata = BackendMetadata(
        backend_name="isolation-fixture",
        model_name="isolation-model",
        model_revision="r1",
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(),
        embedding_dimensions=3,
    )
    profile = EmbeddingProfile(
        profile_version="1.0.0",
        tokenizer_id="isolation-tokenizer",
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
        def vector(text: str) -> tuple[float, float, float]:
            if text.endswith("alpha passage") or text == "nearest alpha":
                return (1.0, 0.0, 0.0)
            if text.endswith("gamma passage"):
                return (0.0, 1.0, 0.0)
            return (0.0, 0.0, 1.0)

        return EmbeddingResult(
            tuple(vector(text) for text in request.input_texts),
            self.metadata,
            self.profile,
        )


def _publish_corpus(
    repository: SQLiteMetadataRepository,
    index_dir: Path,
    metadata_db_path: Path,
    source_dir: Path,
    embedder: _EmbeddingFixture,
    identity: tuple[CorpusId, CorpusVersionId, DocumentId, DocumentVersionId, str],
) -> str:
    corpus_id, corpus_version, document_id, document_version_id, text = identity
    source_bytes = text.encode("utf-8")
    content_hash = ContentHash(
        HashAlgorithm.SHA256,
        hashlib.sha256(source_bytes).hexdigest(),
    )
    parsed = parsed_document(
        text,
        document_id=document_id,
        document_version_id=document_version_id,
    )
    chunk_set = build_chunk_set(
        ChunkingRequest(
            corpus_id=corpus_id,
            parsed_document=parsed,
            tokeniser=FakeChunkTokeniser(),
            policy=ChunkingPolicy(min_tokens=1, max_tokens=512),
        )
    )
    selection = map_chunk_set(chunk_set)
    membership = CorpusDocumentVersionMembership(
        corpus_id,
        document_id,
        document_version_id,
    )
    source_manifest = BaselineChunkGenerationManifest(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        configuration_identity=selection.configuration_identity,
        generation_id=f"source-{corpus_id.as_str()[-12:]}",
        entries=(BaselineChunkGenerationEntry(membership, selection),),
    )

    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(corpus_id, "dense isolation corpus"))
        transaction.add_document(DocumentIdentity(document_id))
        transaction.add_document_version(
            DocumentVersion(document_version_id, document_id, content_hash)
        )
        transaction.add_corpus_document_version_membership(membership)
        transaction.write_chunk_set(
            ChunkSetWriteRequest(corpus_id=corpus_id, selection=selection)
        )

    parse_request = ParsePersistenceRequest(
        corpus_id=corpus_id,
        parser_input=ParserInput(
            document_id,
            document_version_id,
            source_bytes,
            "text/plain",
            "utf-8",
        ),
        parser_version=CURRENT_PARSER_VERSION,
        outcome=IngestParseOutcome("succeeded", parsed, ()),
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
    )
    parse_records = map_parse_persistence_request(parse_request)
    repository.persist_source_artefact(
        corpus_id,
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

    compatibility = DenseIndexCompatibility(
        model_metadata=embedder.metadata,
        dimensions=3,
        embedding_profile=embedder.profile,
    )
    from inference_lab.retrieval import build_dense_generation

    build_outcome = build_dense_generation(
        index_dir,
        source_manifest,
        embedder,
        metadata_db_path=metadata_db_path,
        source_dir=source_dir,
        compatibility=compatibility,
    )
    participant = DenseGenerationParticipant(index_dir, build_outcome, source_manifest)
    committed: CommitCorpusUpdateOutcome | None = None
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            corpus_id,
            corpus_version,
            expected_corpus_version=None,
            required_component_kinds=(DENSE_COMPONENT_KIND,),
        )
        transaction.add_corpus_update_document_membership(
            corpus_id,
            corpus_version,
            membership,
        )
        transaction.register_corpus_update_component_generation(
            corpus_id,
            corpus_version,
            participant.component_generation(),
            validator=participant.build_validator(repository),
        )
        transaction.seal_corpus_update(corpus_id, corpus_version)
        committed = transaction.commit_corpus_update(corpus_id, corpus_version)
    assert committed is not None
    references = committed.manifest.component_generations
    assert len(references) == 1
    return references[0].generation_id


def test_RET_04_two_corpora_restart_and_generation_mismatch_isolation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    repository = SQLiteMetadataRepository.open(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    embedder = _EmbeddingFixture()
    generation_ids = tuple(
        _publish_corpus(
            repository,
            index_dir,
            database_path,
            source_dir,
            embedder,
            identity,
        )
        for identity in _CORPORA
    )
    repository.close()

    cold_query = """
from pathlib import Path
import sys

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import execute_dense_retrieval
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.values import RetrievalLimit
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.retrieval.test_dense_corpus_isolation_acceptance import (
    _EmbeddingFixture,
)

database_path, source_dir, index_dir = map(Path, sys.argv[1:])
repository = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
embedder = _EmbeddingFixture()
expected = (
    ("00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000201", "alpha passage"),
    ("00000000-0000-4000-8000-000000000003", "00000000-0000-4000-8000-000000000203", "gamma passage"),
)
for corpus_id, version, content in expected:
    result = execute_dense_retrieval(
        repository,
        index_dir,
        CorpusId.from_str(corpus_id),
        RetrievalQuery("nearest alpha", CorpusVersionId.from_str(version), RetrievalLimit(2)),
        embedder,
    )
    assert result.status.value == "completed"
    assert tuple(hit.candidate.content for hit in result.hits) == (content,)
repository.close()
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            cold_query,
            str(database_path),
            str(source_dir),
            str(index_dir),
        ],
        check=True,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    repository = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    corpus_id, corpus_version, _, _, _ = _CORPORA[0]
    with open_dense_generation(index_dir, generation_ids[1]) as foreign_generation:
        mismatched = execute_dense_retrieval(
            repository,
            index_dir,
            corpus_id,
            RetrievalQuery("nearest alpha", corpus_version, RetrievalLimit(2)),
            embedder,
            generation=foreign_generation,
        )
    assert mismatched.status is ResultStatus.FAILED
    assert mismatched.hits == ()
    assert mismatched.failures[0].category is FailureCategory.INVALID_CHANNEL_RESPONSE
    repository.close()
