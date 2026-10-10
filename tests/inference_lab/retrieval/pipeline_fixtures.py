"""Published two-corpus fixtures with both retrieval index components."""

import hashlib
from pathlib import Path

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingProfile,
    EmbeddingRequest,
    EmbeddingResult,
    FakeBackend,
    FakeBackendConfig,
    FakeFailureMode,
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
    EvidenceTokenisation,
    EvidenceTokeniserIdentity,
    EvidenceTokenOffset,
    LexicalGenerationParticipant,
    build_dense_generation,
    build_lexical_generation,
)
from inference_lab.storage import (
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    CommitCorpusUpdateOutcome,
    ComponentKind,
    SourceArtefact,
    build_source_blob_storage_key,
    open_metadata_repository,
)
from tests.inference_lab.retrieval.lexical_fixtures import parsed_document

SELECTED_CORPUS = CorpusId.from_str("00000000-0000-4000-8000-0000000000a1")
OTHER_CORPUS = CorpusId.from_str("00000000-0000-4000-8000-0000000000a2")
SELECTED_VERSION = CorpusVersionId.from_str("00000000-0000-4000-8000-0000000000b1")
OTHER_VERSION = CorpusVersionId.from_str("00000000-0000-4000-8000-0000000000b2")

# Texts are chosen so the lexical channel matches every selected member on one
# exact term while the embedding fixture orders them differently. That gives
# fusion two disagreeing channels to combine.
ORCHID = (
    DocumentId.from_str("00000000-0000-4000-8000-0000000000c1"),
    DocumentVersionId.from_str("00000000-0000-4000-8000-0000000000d1"),
    "orchid receptor evidence describes a selective allosteric assay result",
)
TULIP = (
    DocumentId.from_str("00000000-0000-4000-8000-0000000000c2"),
    DocumentVersionId.from_str("00000000-0000-4000-8000-0000000000d2"),
    "tulip catalyst evidence",
)
COMMON = (
    DocumentId.from_str("00000000-0000-4000-8000-0000000000c3"),
    DocumentVersionId.from_str("00000000-0000-4000-8000-0000000000d3"),
    "common assay evidence",
)
EXCLUDED = (
    DocumentId.from_str("00000000-0000-4000-8000-0000000000c4"),
    DocumentVersionId.from_str("00000000-0000-4000-8000-0000000000d4"),
    "withheld evidence that must never reach the selected corpus",
)
SELECTED_MEMBERS = (ORCHID, TULIP, COMMON)
OTHER_MEMBERS = (EXCLUDED,)
ALL_MEMBERS = SELECTED_MEMBERS + OTHER_MEMBERS


class PipelineEmbedder:
    """Deterministic one-hot vectors with an observable embed-work count."""

    metadata = BackendMetadata(
        backend_name="pipeline-fixture",
        model_name="pipeline-fixture",
        model_revision="1",
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(),
        embedding_dimensions=3,
    )
    profile = EmbeddingProfile(
        profile_version="1.0.0",
        tokenizer_id="pipeline-tokeniser",
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


class WhitespaceTokeniser:
    """Offset-bearing tokeniser so evidence reduction can shorten passages."""

    identity = EvidenceTokeniserIdentity(
        implementation="pipeline-whitespace",
        revision="fixture-v1",
        contract_version="1.0.0",
    )

    def tokenise(self, text: str) -> EvidenceTokenisation:
        offsets: list[EvidenceTokenOffset] = []
        start: int | None = None
        for index, character in enumerate(text):
            if character.isspace():
                if start is not None:
                    offsets.append(EvidenceTokenOffset(start, index))
                    start = None
            elif start is None:
                start = index
        if start is not None:
            offsets.append(EvidenceTokenOffset(start, len(text)))
        return EvidenceTokenisation(tuple(offsets))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenise(text).offsets)


def reranker(
    scores: tuple[tuple[str, float], ...] = (),
    *,
    failure_mode: FakeFailureMode | None = None,
) -> FakeBackend:
    """Return a loaded deterministic reranker declaring a preprocessing profile."""
    backend = FakeBackend(
        FakeBackendConfig(
            metadata=BackendMetadata(
                backend_name="pipeline-reranker",
                model_name="pipeline-reranker",
                model_revision="1",
                preprocessing_profile_version="1.0.0",
                supported_operations=(BackendOperation.RERANK,),
                capabilities=(BackendCapability.BATCHING,),
            ),
            rerank_scores=scores,
            failure_mode=failure_mode,
            fail_operation=(
                BackendOperation.RERANK if failure_mode is not None else None
            ),
        )
    )
    backend.load()
    return backend


def compatibility() -> DenseIndexCompatibility:
    """Return the dense compatibility shared by every build in this journey."""
    return DenseIndexCompatibility(
        model_metadata=PipelineEmbedder.metadata,
        dimensions=3,
        embedding_profile=PipelineEmbedder.profile,
    )


def source_manifest(
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    members: tuple[
        tuple[CorpusDocumentVersionMembership, ChunkSetSelectionRecord], ...
    ],
) -> BaselineChunkGenerationManifest:
    """Describe one complete canonical chunk generation for selected members."""
    entries = tuple(
        sorted(
            (
                BaselineChunkGenerationEntry(membership=membership, selection=selection)
                for membership, selection in members
            ),
            key=lambda entry: entry.key(),
        )
    )
    return BaselineChunkGenerationManifest(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        configuration_identity=members[0][1].configuration_identity,
        generation_id="hybrid-pipeline-generation",
        entries=entries,
    )


def parse_records(
    corpus_id: CorpusId,
    document_id: DocumentId,
    version_id: DocumentVersionId,
    text: str,
) -> MappedParseRecords:
    """Map one parsed document into its durable storage records."""
    return map_parse_persistence_request(
        ParsePersistenceRequest(
            corpus_id=corpus_id,
            parser_input=ParserInput(
                document_id,
                version_id,
                text.encode("utf-8"),
                "text/plain",
                "utf-8",
            ),
            parser_version=CURRENT_PARSER_VERSION,
            outcome=ParseOutcome(
                "succeeded",
                parsed_document(
                    text,
                    document_id=document_id,
                    document_version_id=version_id,
                ),
                (),
            ),
            parse_id=RecordId.generate(),
            parsed_record_id=RecordId.generate(),
        )
    )


def publish_corpora(
    database_path: Path,
    source_dir: Path,
    index_dir: Path,
    embedder: PipelineEmbedder,
) -> dict[CorpusId, BaselineChunkGenerationManifest]:
    """Publish both corpora with a lexical and a dense generation each."""
    repository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    corpus_members = {
        SELECTED_CORPUS: SELECTED_MEMBERS,
        OTHER_CORPUS: OTHER_MEMBERS,
    }
    versions = {SELECTED_CORPUS: SELECTED_VERSION, OTHER_CORPUS: OTHER_VERSION}
    staged: dict[
        CorpusId,
        tuple[tuple[CorpusDocumentVersionMembership, ChunkSetSelectionRecord], ...],
    ] = {}
    for corpus_id, documents in corpus_members.items():
        built = []
        for document_id, version_id, text in documents:
            chunk_set = build_chunk_set(
                ChunkingRequest(
                    corpus_id,
                    parsed_document(
                        text,
                        document_id=document_id,
                        document_version_id=version_id,
                    ),
                    FakeChunkTokeniser(),
                    ChunkingPolicy(min_tokens=1, max_tokens=128),
                )
            )
            built.append(
                (
                    CorpusDocumentVersionMembership(corpus_id, document_id, version_id),
                    map_chunk_set(chunk_set),
                )
            )
        staged[corpus_id] = tuple(built)

    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(SELECTED_CORPUS, "selected corpus"))
        transaction.add_corpus(CorpusMetadata(OTHER_CORPUS, "other corpus"))
        for document_id, version_id, text in ALL_MEMBERS:
            transaction.add_document(DocumentIdentity(document_id))
            transaction.add_document_version(
                DocumentVersion(
                    version_id,
                    document_id,
                    ContentHash(
                        HashAlgorithm.SHA256,
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    ),
                )
            )
        for corpus_id, members in staged.items():
            for membership, selection in members:
                transaction.add_corpus_document_version_membership(membership)
                transaction.write_chunk_set(
                    ChunkSetWriteRequest(corpus_id=corpus_id, selection=selection)
                )

    for corpus_id, documents in corpus_members.items():
        for document_id, version_id, text in documents:
            source_bytes = text.encode("utf-8")
            content_hash = ContentHash(
                HashAlgorithm.SHA256, hashlib.sha256(source_bytes).hexdigest()
            )
            records = parse_records(corpus_id, document_id, version_id, text)
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
                parse_outcome=records.outcome,
                parsed_representation=records.representation,
            )

    manifests = {
        corpus_id: source_manifest(corpus_id, versions[corpus_id], members)
        for corpus_id, members in staged.items()
    }
    for corpus_id, manifest in manifests.items():
        lexical = build_lexical_generation(index_dir, manifest)
        dense = build_dense_generation(
            index_dir,
            manifest,
            embedder,
            metadata_db_path=database_path,
            source_dir=source_dir,
            compatibility=compatibility(),
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
            for membership, _selection in staged[corpus_id]:
                transaction.add_corpus_update_document_membership(
                    corpus_id, versions[corpus_id], membership
                )
            for participant in (
                LexicalGenerationParticipant(index_dir, lexical, manifest),
                DenseGenerationParticipant(index_dir, dense, manifest),
            ):
                transaction.register_corpus_update_component_generation(
                    corpus_id,
                    versions[corpus_id],
                    participant.component_generation(),
                    validator=participant.build_validator(repository),
                )
            transaction.seal_corpus_update(corpus_id, versions[corpus_id])
            committed = transaction.commit_corpus_update(corpus_id, versions[corpus_id])
        assert committed is not None
    repository.close()
    return manifests
