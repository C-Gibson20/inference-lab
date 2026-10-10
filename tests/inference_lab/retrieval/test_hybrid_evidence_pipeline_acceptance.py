"""Composed acceptance journey for the hybrid evidence pipeline.

One scenario takes two corpora through dense and lexical index publication,
a repository restart, fused multi-channel retrieval, reranked evidence
selection, and token-budgeted evidence reduction, then re-runs the build to
prove unchanged indexes are reused. It protects the composition of those
stages, which their individual suites cannot observe.
"""

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
    EvidenceHit,
    EvidenceReductionAction,
    EvidenceReductionOutcome,
    EvidenceTokenisation,
    EvidenceTokeniserIdentity,
    EvidenceTokenOffset,
    FusionPolicy,
    LexicalGenerationParticipant,
    ResultStatus,
    RetrievalChannel,
    RetrievalLimit,
    RetrievalQuery,
    build_dense_generation,
    build_lexical_generation,
    execute_hybrid_retrieval,
    open_lexical_generation,
    reduce_retrieval_evidence,
)
from inference_lab.retrieval.reranking import (
    EvidenceCount,
    RerankCandidateLimit,
    execute_reranking,
)
from inference_lab.settings import RetrievalConfig
from inference_lab.storage import (
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    CommitCorpusUpdateOutcome,
    ComponentKind,
    MetadataRepository,
    SourceArtefact,
    build_source_blob_storage_key,
    open_metadata_repository,
)
from tests.inference_lab.retrieval.lexical_fixtures import parsed_document

_SELECTED_CORPUS = CorpusId.from_str("00000000-0000-4000-8000-0000000000a1")
_OTHER_CORPUS = CorpusId.from_str("00000000-0000-4000-8000-0000000000a2")
_SELECTED_VERSION = CorpusVersionId.from_str("00000000-0000-4000-8000-0000000000b1")
_OTHER_VERSION = CorpusVersionId.from_str("00000000-0000-4000-8000-0000000000b2")

# Texts are chosen so the lexical channel matches every selected member on one
# exact term while the embedding fixture orders them differently. That gives
# fusion two genuinely disagreeing channels to combine.
_ORCHID = (
    DocumentId.from_str("00000000-0000-4000-8000-0000000000c1"),
    DocumentVersionId.from_str("00000000-0000-4000-8000-0000000000d1"),
    "orchid receptor evidence describes a selective allosteric assay result",
)
_TULIP = (
    DocumentId.from_str("00000000-0000-4000-8000-0000000000c2"),
    DocumentVersionId.from_str("00000000-0000-4000-8000-0000000000d2"),
    "tulip catalyst evidence",
)
_COMMON = (
    DocumentId.from_str("00000000-0000-4000-8000-0000000000c3"),
    DocumentVersionId.from_str("00000000-0000-4000-8000-0000000000d3"),
    "common assay evidence",
)
_EXCLUDED = (
    DocumentId.from_str("00000000-0000-4000-8000-0000000000c4"),
    DocumentVersionId.from_str("00000000-0000-4000-8000-0000000000d4"),
    "withheld evidence that must never reach the selected corpus",
)
_SELECTED_MEMBERS = (_ORCHID, _TULIP, _COMMON)
_OTHER_MEMBERS = (_EXCLUDED,)
_ALL_MEMBERS = _SELECTED_MEMBERS + _OTHER_MEMBERS


class _PipelineEmbedder:
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


class _WhitespaceTokeniser:
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


def _reranker(
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


def _compatibility() -> DenseIndexCompatibility:
    """Return the dense compatibility shared by every build in this journey."""
    return DenseIndexCompatibility(
        model_metadata=_PipelineEmbedder.metadata,
        dimensions=3,
        embedding_profile=_PipelineEmbedder.profile,
    )


def _source_manifest(
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


def _parse_records(
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


def _publish_corpora(
    database_path: Path,
    source_dir: Path,
    index_dir: Path,
    embedder: _PipelineEmbedder,
) -> dict[CorpusId, BaselineChunkGenerationManifest]:
    """Publish both corpora with a lexical and a dense generation each."""
    repository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    corpus_members = {
        _SELECTED_CORPUS: _SELECTED_MEMBERS,
        _OTHER_CORPUS: _OTHER_MEMBERS,
    }
    versions = {_SELECTED_CORPUS: _SELECTED_VERSION, _OTHER_CORPUS: _OTHER_VERSION}
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
        transaction.add_corpus(CorpusMetadata(_SELECTED_CORPUS, "selected corpus"))
        transaction.add_corpus(CorpusMetadata(_OTHER_CORPUS, "other corpus"))
        for document_id, version_id, text in _ALL_MEMBERS:
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
            records = _parse_records(corpus_id, document_id, version_id, text)
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
        corpus_id: _source_manifest(corpus_id, versions[corpus_id], members)
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
            compatibility=_compatibility(),
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


def _generation_id(
    repository: MetadataRepository, corpus_id: CorpusId, kind: str
) -> str:
    """Return the committed generation identity for one component kind."""
    selected: str | None = None
    with repository.open_active_corpus_snapshot(corpus_id) as snapshot:
        references = tuple(
            reference
            for reference in snapshot.manifest.component_generations
            if reference.component_kind.as_str() == kind
        )
        assert len(references) == 1
        selected = references[0].generation_id
    assert selected is not None
    return selected


def _fused_order_from_contributions(
    hits: tuple[EvidenceHit, ...], policy: FusionPolicy
) -> tuple[str, ...]:
    """Recompute the fused order from each hit's retained channel contributions.

    This verifies the fusion arithmetic, weights, rank constant, and record-id
    tie break against the ranks fusion actually retained. It deliberately does
    not re-derive each channel's own ranks, which belong to that channel's
    contract.
    """
    scored = []
    for hit in hits:
        score = sum(
            policy.weight_for(contribution.channel)
            / (policy.rank_constant + contribution.rank)
            for contribution in hit.contributions
        )
        scored.append((score, hit.candidate.record_id.as_str()))
    return tuple(
        record_id for _score, record_id in sorted(scored, key=lambda i: (-i[0], i[1]))
    )


def test_hybrid_evidence_pipeline_survives_restart_and_reuses_indexes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    embedder = _PipelineEmbedder()
    manifests = _publish_corpora(database_path, source_dir, index_dir, embedder)
    build_work = tuple(embedder.embedded_texts)
    assert build_work, "publication must embed the staged chunks"

    # Restart: every handle from publication is closed before serving reads.
    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:
        query = RetrievalQuery("evidence", _SELECTED_VERSION, RetrievalLimit(10))
        policy = FusionPolicy()
        fused = execute_hybrid_retrieval(
            repository,
            index_dir,
            _SELECTED_CORPUS,
            query,
            embedder,
            policy=policy,
        )

        assert fused.status is ResultStatus.COMPLETED
        assert fused.corpus_version == _SELECTED_VERSION
        assert fused.hits, "the fused result must retain evidence after restart"

        contributing = {
            contribution.channel
            for hit in fused.hits
            for contribution in hit.contributions
        }
        assert contributing == {RetrievalChannel.LEXICAL, RetrievalChannel.DENSE}, (
            "both ranked channels must contribute to the fused evidence set"
        )
        assert any(len(hit.contributions) == 2 for hit in fused.hits), (
            "at least one record must be found by both channels"
        )

        # Fusion order is exactly weighted reciprocal rank over the retained
        # per-channel ranks, with canonical record identity breaking ties.
        leaders = tuple(
            hit
            for hit in fused.hits
            if len(hit.contributions) == 2
            and all(contribution.rank == 1 for contribution in hit.contributions)
        )
        assert len(leaders) == 1 and leaders[0].rank == 1, (
            "a record ranked first by both channels must lead the fused order"
        )

        actual_order = tuple(hit.candidate.record_id.as_str() for hit in fused.hits)
        assert actual_order == _fused_order_from_contributions(fused.hits, policy)
        assert tuple(hit.rank for hit in fused.hits) == tuple(
            range(1, len(fused.hits) + 1)
        )

        # Corpus isolation: the other corpus shares physical storage only.
        excluded_versions = {version.as_str() for _d, version, _t in _OTHER_MEMBERS}
        assert (
            not {
                source.document_version_id.as_str()
                for hit in fused.hits
                for source in hit.candidate.sources
            }
            & excluded_versions
        )

        # Reranking selects a bounded evidence set and preserves provenance.
        fused_sources = {
            hit.candidate.record_id.as_str(): hit.candidate.sources
            for hit in fused.hits
        }
        # Score the fused candidates in reverse so the final order must come
        # from the backend rather than from the incoming fused order.
        inverted = tuple(
            (hit.candidate.record_id.as_str(), float(index))
            for index, hit in enumerate(fused.hits)
        )
        reranked = execute_reranking(
            _reranker(inverted),
            query,
            fused,
            corpus_id=_SELECTED_CORPUS,
            evidence_count=EvidenceCount(5),
            candidate_limit=RerankCandidateLimit(10),
        )
        assert reranked.degraded is False
        assert reranked.result.status is ResultStatus.COMPLETED
        assert 0 < len(reranked.result.hits) <= 5
        assert tuple(hit.rank for hit in reranked.result.hits) == tuple(
            range(1, len(reranked.result.hits) + 1)
        )
        assert tuple(
            hit.candidate.record_id.as_str() for hit in reranked.result.hits
        ) == tuple(
            record_id for record_id, _score in sorted(inverted, key=lambda i: -i[1])
        ), "the final order must follow the reranker scores, not the fused order"
        for hit in reranked.result.hits:
            assert (
                hit.candidate.sources == fused_sources[hit.candidate.record_id.as_str()]
            ), "reranking must not alter candidate provenance"

        # Evidence reduction fits the selected passages into an exact budget.
        lexical_generation_id = _generation_id(
            repository, _SELECTED_CORPUS, "lexical-index"
        )
        tokeniser = _WhitespaceTokeniser()
        # One token below the leading passage forces exactly one truncation.
        budget = max(
            2,
            tokeniser.count_tokens(reranked.result.hits[0].candidate.content) - 1,
        )
        reduction: EvidenceReductionOutcome | None = None
        with repository.open_corpus_snapshot(
            _SELECTED_CORPUS, _SELECTED_VERSION
        ) as snapshot:
            with open_lexical_generation(
                index_dir, lexical_generation_id
            ) as generation:
                reduction = reduce_retrieval_evidence(
                    _SELECTED_CORPUS,
                    snapshot,
                    generation,
                    reranked.result,
                    tokeniser,
                    RetrievalConfig(),
                    token_budget_override=budget,
                )

        assert reduction is not None
        assert reduction.corpus_version == _SELECTED_VERSION
        assert reduction.token_count <= budget
        assert reduction.items, "reduction must retain at least one passage"
        assert any(
            item.action is EvidenceReductionAction.TRUNCATED for item in reduction.items
        ), "the chosen budget must force exactly the truncation path under test"
        assert reduction.identity.tokeniser == tokeniser.identity
        assert reduction.input_count == len(reranked.result.hits)
        for item in reduction.items:
            if item.action is EvidenceReductionAction.TRUNCATED:
                assert item.source_ranges, (
                    "a shortened passage must retain exact source ranges"
                )
                assert item.text != item.hit.candidate.content
                assert len(item.text) < len(item.hit.candidate.content)
            else:
                assert item.text == item.hit.candidate.content
    finally:
        repository.close()

    # Unchanged input must reuse both durable generations and re-embed nothing.
    replay_embedder = _PipelineEmbedder()
    for corpus_id, manifest in manifests.items():
        lexical_replay = build_lexical_generation(index_dir, manifest)
        dense_replay = build_dense_generation(
            index_dir,
            manifest,
            replay_embedder,
            metadata_db_path=database_path,
            source_dir=source_dir,
            compatibility=_compatibility(),
        )
        assert lexical_replay.statistics.database_reused, corpus_id.as_str()
        assert dense_replay.statistics.generation_reused, corpus_id.as_str()
    assert replay_embedder.embedded_texts == []


def test_retryable_reranker_failure_degrades_to_fused_order(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    embedder = _PipelineEmbedder()
    _publish_corpora(database_path, source_dir, index_dir, embedder)

    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:
        query = RetrievalQuery("evidence", _SELECTED_VERSION, RetrievalLimit(10))
        fused = execute_hybrid_retrieval(
            repository, index_dir, _SELECTED_CORPUS, query, embedder
        )
        assert fused.hits

        outcome = execute_reranking(
            _reranker(failure_mode=FakeFailureMode.RESOURCE),
            query,
            fused,
            corpus_id=_SELECTED_CORPUS,
            evidence_count=EvidenceCount(5),
            candidate_limit=RerankCandidateLimit(10),
        )
    finally:
        repository.close()

    # A retryable backend failure keeps the deterministic fused order, marks the
    # outcome degraded, and reports a typed failure without losing provenance.
    assert outcome.degraded is True
    assert outcome.result.status is ResultStatus.PARTIAL
    assert outcome.result.failures
    assert all(failure.retryable for failure in outcome.result.failures)
    expected = tuple(hit.candidate.record_id.as_str() for hit in fused.hits)[
        : len(outcome.result.hits)
    ]
    assert (
        tuple(hit.candidate.record_id.as_str() for hit in outcome.result.hits)
        == expected
    )
