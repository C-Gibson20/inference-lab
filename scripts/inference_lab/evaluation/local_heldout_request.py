"""Build the frozen held-out baseline request from local private assets."""

from __future__ import annotations

import atexit
import json
from pathlib import Path

from inference_lab.backends.runtime import LocalRerankerBackend, QwenEmbeddingBackend
from inference_lab.evaluation import (
    HeldOutBaselineRequest,
    HeldOutChannel,
    HeldOutChannelIdentity,
    HeldOutGatePolicy,
    HeldOutRunContract,
    LocalDenseEvaluationAdapter,
    LocalFusedEvaluationAdapter,
    LocalHeldOutChannelRunner,
    LocalLexicalEvaluationAdapter,
    LocalRerankedEvaluationAdapter,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRequest,
    RetrievalEvaluationRunIdentity,
    UnavailableMetadataEvaluationAdapter,
    lexical_relationship_snapshot,
    load_dataset,
    validate_final_benchmark,
)
from inference_lab.evaluation.heldout_gate_codec import (
    HeldOutGateBinding,
    deserialise_approved_held_out_gate,
)
from inference_lab.evaluation.loader import LoadedDataset
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
_GATE_POLICY = _ROOT / "docs/operator/retrieval-baseline-gate.json"
_OPEN_RESOURCES: list[object] = []


def _close_resources() -> None:
    for resource in reversed(_OPEN_RESOURCES):
        close = getattr(resource, "close", None)
        if callable(close):
            close()


atexit.register(_close_resources)


def _load_gate_policy(
    dataset: LoadedDataset,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> HeldOutGatePolicy:
    """Load the approved policy only when it matches these frozen inputs."""
    approved = deserialise_approved_held_out_gate(_GATE_POLICY.read_bytes())
    manifest = dataset.manifest
    checksum = manifest.checksums.get(manifest.case_file)
    if checksum is None:
        raise ValueError("local benchmark manifest has no case-file checksum")
    expected = HeldOutGateBinding(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        dataset_schema_version=manifest.schema_version,
        case_file_checksum=checksum,
        corpus_id=corpus_id.as_str(),
        corpus_version=corpus_version.as_str(),
    )
    if approved.binding != expected:
        raise ValueError("approved gate policy does not match the local benchmark")
    return approved.policy


def make_heldout_request() -> HeldOutBaselineRequest:
    """Assemble all pinned channel identities before executing held-out cases."""
    settings = get_settings()
    dataset = load_dataset(_DATASET_MANIFEST)
    validate_final_benchmark(dataset.cases)
    manifest = dataset.manifest
    corpus_id = CorpusId.from_str(manifest.corpus_id)
    corpus_version = CorpusVersionId.from_str(manifest.corpus_version)
    setup = json.loads(_SETUP_RECORD.read_text(encoding="utf-8"))
    if (
        setup.get("status") != "ready"
        or setup.get("corpus_id") != corpus_id.as_str()
        or setup.get("proposed_corpus_version") != corpus_version.as_str()
    ):
        raise ValueError("local setup and bound benchmark corpus identities differ")
    gate_policy = _load_gate_policy(dataset, corpus_id, corpus_version)

    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=False,
    )
    _OPEN_RESOURCES.append(repository)
    relationships = lexical_relationship_snapshot(repository, corpus_id, corpus_version)
    embedder = QwenEmbeddingBackend(settings.models.embedder_path)
    _OPEN_RESOURCES.append(embedder)
    embedder.load()
    reranker = LocalRerankerBackend(settings.models.reranker_path)
    _OPEN_RESOURCES.append(reranker)
    reranker.load()

    lexical = LocalLexicalEvaluationAdapter(
        repository,
        settings.storage.index_dir,
        corpus_id,
        corpus_version,
    )
    dense = LocalDenseEvaluationAdapter(
        repository,
        settings.storage.index_dir,
        corpus_id,
        corpus_version,
        embedder,
    )
    fused = LocalFusedEvaluationAdapter(
        repository,
        settings.storage.index_dir,
        corpus_id,
        corpus_version,
        embedder,
    )
    reranked = LocalRerankedEvaluationAdapter(
        fused,
        reranker,
        evidence_count=EvidenceCount.from_config(settings.retrieval),
        candidate_limit=RerankCandidateLimit.from_config(settings.retrieval),
    )
    metadata = UnavailableMetadataEvaluationAdapter()
    adapters = {
        HeldOutChannel.LEXICAL_ONLY: lexical,
        HeldOutChannel.DENSE_ONLY: dense,
        HeldOutChannel.METADATA: metadata,
        HeldOutChannel.FUSED_PRE_RERANK: fused,
        HeldOutChannel.FUSED_RERANKED: reranked,
    }
    identities = tuple(
        HeldOutChannelIdentity(
            channel=channel,
            system_identity=adapters[channel].system_identity,
            fusion_policy=(
                fused.fusion_policy
                if channel
                in {
                    HeldOutChannel.FUSED_PRE_RERANK,
                    HeldOutChannel.FUSED_RERANKED,
                }
                else None
            ),
            reranking_identity=(
                reranked.reranking_identity
                if channel is HeldOutChannel.FUSED_RERANKED
                else None
            ),
        )
        for channel in HeldOutChannel
    )
    selected_case_ids = tuple(manifest.splits["held_out"])
    policy = RetrievalEvaluationPolicy(
        passage_cutoffs=(8, 10),
        candidate_depth=settings.retrieval.candidate_limit,
    )
    if relationships.content_digest is None:
        raise ValueError("corpus relationship snapshot has no content digest")
    evaluation_identity = RetrievalEvaluationRunIdentity(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        dataset_schema_version=manifest.schema_version,
        case_file_checksum=manifest.checksums[manifest.case_file],
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        selected_split="held_out",
        selected_case_ids=selected_case_ids,
        policy=policy,
        relationship_snapshot_digest=relationships.content_digest,
        system_identity=fused.system_identity,
    )
    contract = HeldOutRunContract(
        evaluation_identity=evaluation_identity,
        gate_policy=gate_policy,
        channel_identities=identities,
        final_evidence_count=settings.retrieval.evidence_count,
    )
    request = RetrievalEvaluationRequest(
        dataset=dataset,
        selected_case_ids=selected_case_ids,
        selected_split="held_out",
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        policy=policy,
        relationship_snapshot=relationships,
        adapter=fused,
    )
    channel_runner = LocalHeldOutChannelRunner(contract, request, adapters)
    cases_by_id = {case.case_id: case for case in dataset.cases}
    cases = tuple(cases_by_id[case_id] for case_id in selected_case_ids)
    return HeldOutBaselineRequest(
        contract=contract,
        cases=cases,
        run_channel=channel_runner,
    )
