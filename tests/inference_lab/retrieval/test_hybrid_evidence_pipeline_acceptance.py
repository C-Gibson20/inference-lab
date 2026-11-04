"""Composed journey: publish, restart, fuse, rerank, reduce, and reuse."""

from pathlib import Path

from inference_lab.backends.runtime import FakeFailureMode
from inference_lab.provenance import CorpusId
from inference_lab.retrieval import (
    EvidenceHit,
    EvidenceReductionAction,
    EvidenceReductionOutcome,
    FusionPolicy,
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
from inference_lab.storage import MetadataRepository, open_metadata_repository
from tests.inference_lab.retrieval.pipeline_fixtures import (
    OTHER_MEMBERS,
    SELECTED_CORPUS,
    SELECTED_VERSION,
    PipelineEmbedder,
    WhitespaceTokeniser,
    compatibility,
    publish_corpora,
    reranker,
)


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
    """Recompute the fused order from each hit's retained contributions."""
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
    embedder = PipelineEmbedder()
    manifests = publish_corpora(database_path, source_dir, index_dir, embedder)
    build_work = tuple(embedder.embedded_texts)
    assert build_work, "publication must embed the staged chunks"

    # Restart: every handle from publication is closed before serving reads.
    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:
        query = RetrievalQuery("evidence", SELECTED_VERSION, RetrievalLimit(10))
        policy = FusionPolicy()
        fused = execute_hybrid_retrieval(
            repository,
            index_dir,
            SELECTED_CORPUS,
            query,
            embedder,
            policy=policy,
        )

        assert fused.status is ResultStatus.COMPLETED
        assert fused.corpus_version == SELECTED_VERSION
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
        excluded_versions = {version.as_str() for _d, version, _t in OTHER_MEMBERS}
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
            reranker(inverted),
            query,
            fused,
            corpus_id=SELECTED_CORPUS,
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
            repository, SELECTED_CORPUS, "lexical-index"
        )
        tokeniser = WhitespaceTokeniser()
        # One token below the leading passage forces exactly one truncation.
        budget = max(
            2,
            tokeniser.count_tokens(reranked.result.hits[0].candidate.content) - 1,
        )
        reduction: EvidenceReductionOutcome | None = None
        with repository.open_corpus_snapshot(
            SELECTED_CORPUS, SELECTED_VERSION
        ) as snapshot:
            with open_lexical_generation(
                index_dir, lexical_generation_id
            ) as generation:
                reduction = reduce_retrieval_evidence(
                    SELECTED_CORPUS,
                    snapshot,
                    generation,
                    reranked.result,
                    tokeniser,
                    RetrievalConfig(),
                    token_budget_override=budget,
                )

        assert reduction is not None
        assert reduction.corpus_version == SELECTED_VERSION
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
    replay_embedder = PipelineEmbedder()
    for corpus_id, manifest in manifests.items():
        lexical_replay = build_lexical_generation(index_dir, manifest)
        dense_replay = build_dense_generation(
            index_dir,
            manifest,
            replay_embedder,
            metadata_db_path=database_path,
            source_dir=source_dir,
            compatibility=compatibility(),
        )
        assert lexical_replay.statistics.database_reused, corpus_id.as_str()
        assert dense_replay.statistics.generation_reused, corpus_id.as_str()
    assert replay_embedder.embedded_texts == []


def test_retryable_reranker_failure_degrades_to_fused_order(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    embedder = PipelineEmbedder()
    publish_corpora(database_path, source_dir, index_dir, embedder)

    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:
        query = RetrievalQuery("evidence", SELECTED_VERSION, RetrievalLimit(10))
        fused = execute_hybrid_retrieval(
            repository, index_dir, SELECTED_CORPUS, query, embedder
        )
        assert fused.hits

        outcome = execute_reranking(
            reranker(failure_mode=FakeFailureMode.RESOURCE),
            query,
            fused,
            corpus_id=SELECTED_CORPUS,
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
