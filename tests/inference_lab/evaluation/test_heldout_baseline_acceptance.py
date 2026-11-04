"""Held-out baseline acceptance over a published fixture corpus.

One scenario drives every shipped channel adapter against a real two-corpus
deployment, aggregates the frozen run, applies the gate policy, publishes the
report bundle, and repeats the run to prove the fingerprint and report digest
are byte-identical. It is the only path that exercises the adapters, runner,
reporter, gate, and publisher together; the operator suite substitutes fakes.
"""

import hashlib
import json
from pathlib import Path

from inference_lab.evaluation import (
    HeldOutChannel,
    HeldOutChannelIdentity,
    HeldOutGateOutcome,
    HeldOutGatePolicy,
    HeldOutRunContract,
    LocalDenseEvaluationAdapter,
    LocalFusedEvaluationAdapter,
    LocalHeldOutChannelRunner,
    LocalLexicalEvaluationAdapter,
    LocalRerankedEvaluationAdapter,
    MetricFraction,
    RetrievalEvaluationAdapter,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRequest,
    RetrievalEvaluationRunIdentity,
    UnavailableMetadataEvaluationAdapter,
    lexical_relationship_snapshot,
    load_dataset,
    publish_heldout_baseline_report,
    render_held_out_baseline_report,
    run_held_out_baseline,
    serialise_held_out_baseline_report,
)
from inference_lab.evaluation.loader import LoadedDataset
from inference_lab.provenance import DocumentId, DocumentVersionId
from inference_lab.retrieval import FusionPolicy
from inference_lab.retrieval.reranking import EvidenceCount, RerankCandidateLimit
from inference_lab.storage import open_metadata_repository
from tests.inference_lab.evaluation.fixtures import write_dataset_fixture
from tests.inference_lab.retrieval.pipeline_fixtures import (
    COMMON,
    ORCHID,
    SELECTED_CORPUS,
    SELECTED_VERSION,
    TULIP,
    PipelineEmbedder,
    parse_records,
    publish_corpora,
    reranker,
)

_Adapters = dict[HeldOutChannel, RetrievalEvaluationAdapter]

_CANDIDATE_DEPTH = 10
_EVIDENCE_COUNT = 5


def _source_span_id(member: tuple[DocumentId, DocumentVersionId, str]) -> str:
    """Return the deterministic parser source-span identity of one member.

    Span identities derive from the document version, parser version, block
    ordinal, kind, and range, so re-parsing the fixture text reproduces exactly
    the spans the published corpus holds.
    """
    document_id, version_id, text = member
    representation = parse_records(
        SELECTED_CORPUS, document_id, version_id, text
    ).representation
    assert representation is not None
    assert representation.source_locations
    return representation.source_locations[0].source_span_id.as_str()


def _dataset(tmp_path: Path) -> LoadedDataset:
    """Write a held-out fixture dataset whose evidence resolves in the corpus."""
    members = (
        ("case-lookup", "lookup", "orchid", ORCHID),
        ("case-comparison", "comparison", "tulip", TULIP),
        ("case-thematic", "thematic", "common", COMMON),
    )
    cases = []
    for case_id, question_type, term, member in members:
        document_id, version_id, _text = member
        span_id = _source_span_id((document_id, version_id, _text))
        cases.append(
            {
                "schema_version": "3.0.0",
                "case_id": case_id,
                "question": term,
                "question_type": question_type,
                "expected_answer": {
                    "claims": (
                        {
                            "key": "evidence",
                            "description": f"Evidence mentioning {term}",
                            "evidence_ids": ("evidence-primary",),
                        },
                    )
                },
                "evidence": [
                    {
                        "evidence_id": "evidence-primary",
                        "corpus_version": SELECTED_VERSION.as_str(),
                        "document_id": document_id.as_str(),
                        "document_version_id": version_id.as_str(),
                        "source_span_id": span_id,
                        "kind": "source_span",
                        "role": "authoritative",
                    }
                ],
                "abstention": {"expected": False, "reason": "not_applicable"},
                "split": "held_out",
            }
        )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_dataset_fixture(
        dataset_dir,
        cases=cases,
        manifest_overrides={
            "corpus_id": SELECTED_CORPUS.as_str(),
            "corpus_version": SELECTED_VERSION.as_str(),
            "splits": {
                "development": [],
                "held_out": [case["case_id"] for case in cases],
            },
        },
    )
    return load_dataset(manifest_path)


def _contract(
    dataset: LoadedDataset,
    relationship_digest: str,
    adapters: _Adapters,
) -> HeldOutRunContract:
    """Bind the frozen gate policy and every channel identity to this dataset."""
    manifest = dataset.manifest
    held_out = set(manifest.splits["held_out"])
    case_ids = tuple(case.case_id for case in dataset.cases if case.case_id in held_out)
    fused = adapters[HeldOutChannel.FUSED_PRE_RERANK]
    identity = RetrievalEvaluationRunIdentity(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        dataset_schema_version=manifest.schema_version,
        case_file_checksum=manifest.checksums[manifest.case_file],
        corpus_id=SELECTED_CORPUS,
        corpus_version=SELECTED_VERSION,
        selected_split="held_out",
        selected_case_ids=case_ids,
        policy=RetrievalEvaluationPolicy(
            passage_cutoffs=(_EVIDENCE_COUNT, _CANDIDATE_DEPTH),
            candidate_depth=_CANDIDATE_DEPTH,
        ),
        relationship_snapshot_digest=relationship_digest,
        system_identity=fused.system_identity,
    )
    return HeldOutRunContract(
        evaluation_identity=identity,
        gate_policy=HeldOutGatePolicy(
            minimum_document_recall_at_10=MetricFraction(1, 1),
            maximum_failure_rate=MetricFraction(0, 1),
            maximum_degraded_rate=MetricFraction(0, 1),
            minimum_category_cases=2,
            minimum_category_leakage_groups=2,
            confidence_level_percent=95,
            bootstrap_replicates=64,
            rationale="Synthetic fixture policy; not a production threshold.",
        ),
        channel_identities=tuple(
            HeldOutChannelIdentity(
                channel,
                adapters[channel].system_identity,
                fusion_policy=getattr(adapters[channel], "fusion_policy", None),
                reranking_identity=getattr(
                    adapters[channel], "reranking_identity", None
                ),
            )
            for channel in HeldOutChannel
        ),
        candidate_recall_cutoff=_CANDIDATE_DEPTH,
        final_evidence_count=_EVIDENCE_COUNT,
    )


def _run(tmp_path: Path, database_path: Path, source_dir: Path, index_dir: Path):
    """Execute every frozen channel against the published corpus."""
    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:
        dataset = _dataset(tmp_path)
        relationships = lexical_relationship_snapshot(
            repository, SELECTED_CORPUS, SELECTED_VERSION
        )
        embedder = PipelineEmbedder()
        fused = LocalFusedEvaluationAdapter(
            repository,
            index_dir,
            SELECTED_CORPUS,
            SELECTED_VERSION,
            embedder,
            policy=FusionPolicy(),
        )
        adapters: _Adapters = {
            HeldOutChannel.LEXICAL_ONLY: LocalLexicalEvaluationAdapter(
                repository, index_dir, SELECTED_CORPUS, SELECTED_VERSION
            ),
            HeldOutChannel.DENSE_ONLY: LocalDenseEvaluationAdapter(
                repository, index_dir, SELECTED_CORPUS, SELECTED_VERSION, embedder
            ),
            HeldOutChannel.METADATA: UnavailableMetadataEvaluationAdapter(),
            HeldOutChannel.FUSED_PRE_RERANK: fused,
            HeldOutChannel.FUSED_RERANKED: LocalRerankedEvaluationAdapter(
                fused,
                reranker(),
                evidence_count=EvidenceCount(_EVIDENCE_COUNT),
                candidate_limit=RerankCandidateLimit(_CANDIDATE_DEPTH),
            ),
        }
        contract = _contract(
            dataset,
            str(relationships.content_digest),
            adapters,
        )
        request = RetrievalEvaluationRequest(
            dataset=dataset,
            selected_case_ids=contract.evaluation_identity.selected_case_ids,
            selected_split="held_out",
            corpus_id=SELECTED_CORPUS,
            corpus_version=SELECTED_VERSION,
            policy=contract.evaluation_identity.policy,
            relationship_snapshot=relationships,
            adapter=fused,
        )
        by_id = {case.case_id: case for case in dataset.cases}
        selected = tuple(
            by_id[case_id] for case_id in contract.evaluation_identity.selected_case_ids
        )
        by_id = {case.case_id: case for case in dataset.cases}
        selected = tuple(
            by_id[case_id] for case_id in contract.evaluation_identity.selected_case_ids
        )
        return run_held_out_baseline(
            contract,
            selected,
            LocalHeldOutChannelRunner(contract, request, adapters),
            limitations=("Synthetic three-case fixture; not corpus evidence.",),
        )
    finally:
        repository.close()


def test_held_out_baseline_runs_every_channel_and_republishes_identically(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "sources"
    index_dir = tmp_path / "indexes"
    publish_corpora(database_path, source_dir, index_dir, PipelineEmbedder())

    report = _run(tmp_path, database_path, source_dir, index_dir)

    # Every frozen channel is summarised, including the unavailable one.
    summaries = {summary.channel: summary for summary in report.channel_summaries}
    assert set(summaries) == set(HeldOutChannel)
    assert report.case_count == 3

    # An unavailable channel keeps its cases in the denominator and reports
    # them as failures, so its zero recall cannot be read as a measurement.
    metadata = summaries[HeldOutChannel.METADATA]
    assert metadata.failed_case_count == 3
    assert metadata.document_recall is not None
    assert metadata.document_recall.failed_case_count == 3
    assert metadata.document_recall.matched_target_count == 0
    assert metadata.document_recall.relevant_target_count == 3

    for channel in (
        HeldOutChannel.LEXICAL_ONLY,
        HeldOutChannel.DENSE_ONLY,
        HeldOutChannel.FUSED_PRE_RERANK,
        HeldOutChannel.FUSED_RERANKED,
    ):
        summary = summaries[channel]
        assert summary.failed_case_count == 0, channel.value
        assert summary.document_recall is not None, channel.value
        assert summary.document_recall.relevant_target_count == 3, channel.value

    # The dense and fused channels retrieve the intended member for every case;
    # the gate's primary channel must not be weaker than dense alone.
    dense_recall = summaries[HeldOutChannel.DENSE_ONLY].document_recall
    fused_recall = summaries[HeldOutChannel.FUSED_PRE_RERANK].document_recall
    assert dense_recall is not None and fused_recall is not None
    assert dense_recall.matched_target_count == 3, (
        "the embedding fixture must retrieve the intended member for every case"
    )
    assert fused_recall.matched_target_count >= dense_recall.matched_target_count, (
        "fusing a working channel must not lose documents that channel found"
    )

    # Channel comparison is retained rather than collapsed into one number.
    assert report.paired_comparisons

    # The gate is applied and its reasons recorded rather than inferred.
    assert report.outcome in set(HeldOutGateOutcome)
    assert isinstance(report.outcome_reasons, tuple)
    assert report.contract.fingerprint

    # Publication writes exactly the canonical bundle.
    destination = tmp_path / "published"
    manifest = publish_heldout_baseline_report(destination, report)
    assert sorted(path.name for path in destination.iterdir()) == [
        "manifest.json",
        "report.json",
        "report.md",
    ]
    assert manifest.run_fingerprint == report.contract.fingerprint
    for artefact in manifest.artifacts:
        body = (destination / artefact.name).read_bytes()
        assert artefact.size_bytes == len(body)
        assert artefact.checksum == f"sha256:{hashlib.sha256(body).hexdigest()}"
    assert json.loads((destination / "report.json").read_text(encoding="utf-8"))

    # An exact repeat must be byte-identical in both artefacts.
    repeat = _run(tmp_path, database_path, source_dir, index_dir)
    assert repeat.contract.fingerprint == report.contract.fingerprint
    assert serialise_held_out_baseline_report(
        repeat
    ) == serialise_held_out_baseline_report(report)
    assert render_held_out_baseline_report(repeat) == render_held_out_baseline_report(
        report
    )
