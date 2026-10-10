"""Build local development retrieval requests from pinned private assets."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

from inference_lab.backends.runtime import LocalRerankerBackend, QwenEmbeddingBackend
from inference_lab.evaluation import (
    LocalDenseEvaluationAdapter,
    LocalFusedEvaluationAdapter,
    LocalLexicalEvaluationAdapter,
    LocalRerankedEvaluationAdapter,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRequest,
    lexical_relationship_snapshot,
    load_dataset,
)
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval.reranking import (
    EvidenceCount,
    RerankCandidateLimit,
)
from inference_lab.settings import get_settings
from inference_lab.storage import open_metadata_repository

_ROOT = Path(__file__).resolve().parents[3]
_DATASET_MANIFEST = _ROOT / ".dev/benchmark-authoring/bound-manifest.json"
_SETUP_RECORD = _ROOT / ".dev/local-evaluation/setup.json"
_OPEN_RESOURCES: list[object] = []


def _close_resources() -> None:
    for resource in reversed(_OPEN_RESOURCES):
        close = getattr(resource, "close", None)
        if callable(close):
            close()


atexit.register(_close_resources)


def make_development_request() -> RetrievalEvaluationRequest:
    """Build one development-only request for a pinned local channel."""
    channel = os.environ.get("EVALUATION_CHANNEL")
    if channel not in {"lexical", "dense", "fused", "reranked"}:
        raise ValueError("EVALUATION_CHANNEL must select a supported local channel")

    settings = get_settings()
    dataset = load_dataset(_DATASET_MANIFEST)
    bound_manifest = dataset.manifest
    setup = json.loads(_SETUP_RECORD.read_text(encoding="utf-8"))
    corpus_id = CorpusId.from_str(bound_manifest.corpus_id)
    corpus_version = CorpusVersionId.from_str(bound_manifest.corpus_version)
    if (
        setup.get("status") != "ready"
        or setup.get("corpus_id") != corpus_id.as_str()
        or setup.get("proposed_corpus_version") != corpus_version.as_str()
    ):
        raise ValueError("local setup and bound benchmark corpus identities differ")

    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=False,
    )
    _OPEN_RESOURCES.append(repository)
    relationships = lexical_relationship_snapshot(repository, corpus_id, corpus_version)

    embedder = None
    if channel in {"dense", "fused", "reranked"}:
        embedder = QwenEmbeddingBackend(settings.models.embedder_path)
        _OPEN_RESOURCES.append(embedder)
        embedder.load()

    if channel == "lexical":
        adapter = LocalLexicalEvaluationAdapter(
            repository,
            settings.storage.index_dir,
            corpus_id,
            corpus_version,
        )
    elif channel == "dense":
        if embedder is None:
            raise RuntimeError("dense channel requires its configured embedder")
        adapter = LocalDenseEvaluationAdapter(
            repository,
            settings.storage.index_dir,
            corpus_id,
            corpus_version,
            embedder,
        )
    else:
        if embedder is None:
            raise RuntimeError("fused channels require their configured embedder")
        fused = LocalFusedEvaluationAdapter(
            repository,
            settings.storage.index_dir,
            corpus_id,
            corpus_version,
            embedder,
        )
        if channel == "fused":
            adapter = fused
        else:
            reranker = LocalRerankerBackend(settings.models.reranker_path)
            _OPEN_RESOURCES.append(reranker)
            reranker.load()
            adapter = LocalRerankedEvaluationAdapter(
                fused,
                reranker,
                evidence_count=EvidenceCount.from_config(settings.retrieval),
                candidate_limit=RerankCandidateLimit.from_config(settings.retrieval),
            )

    return RetrievalEvaluationRequest(
        dataset=dataset,
        selected_case_ids=tuple(bound_manifest.splits["development"]),
        selected_split="development",
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        policy=RetrievalEvaluationPolicy(
            passage_cutoffs=(8, 10),
            candidate_depth=settings.retrieval.candidate_limit,
        ),
        relationship_snapshot=relationships,
        adapter=adapter,
    )
