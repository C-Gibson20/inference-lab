"""Contract tests for the pinned local M3 reranker backend."""

import hashlib
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import structlog.testing
import torch

from inference_lab.backends.runtime import (
    BackendClosedError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendOperation,
    BackendRequestError,
    BackendResourceError,
    LocalRerankerBackend,
    MemoryAccountingAdapter,
    MemoryComponentRole,
    RerankCandidate,
    RerankerBackend,
    RerankRequest,
    reranker,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalLimit,
    RetrievalQuery,
)
from inference_lab.retrieval.reranking import (
    EvidenceCount,
    RerankCandidateLimit,
    execute_reranking,
)
from inference_lab.settings import LoggingConfig


class _TestTokenizer:
    """Small pair tokeniser with the BGE-M3 special-token structure."""

    def __init__(self) -> None:
        self.padding_side = "right"

    def __len__(self) -> int:
        return 250002

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 10 for character in text]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is True
        return 4

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        if token_ids_1 is None:
            return [0, *token_ids_0, 2]
        return [0, *token_ids_0, 2, 2, *token_ids_1, 2]

    def pad(
        self,
        encoded_inputs: list[dict[str, list[int]]],
        *,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding is True
        assert return_tensors == "pt"
        longest = max(len(item["input_ids"]) for item in encoded_inputs)
        input_ids = []
        attention_mask = []
        for item in encoded_inputs:
            values = item["input_ids"]
            input_ids.append(values + [1] * (longest - len(values)))
            attention_mask.append([1] * len(values) + [0] * (longest - len(values)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class _TestModel(torch.nn.Module):
    """Return fixed scores based on the first candidate token."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            model_type="xlm-roberta", hidden_size=1024, num_labels=1
        )
        self.score_by_first_candidate_token = {
            ord("a") + 10: 1.0,
            ord("b") + 10: 3.0,
            ord("c") + 10: 2.0,
            ord("d") + 10: 3.0,
            ord("e") + 10: 4.0,
        }
        self.batch_sizes: list[int] = []
        self.input_ids: list[tuple[int, ...]] = []
        self.non_finite = False
        self.out_of_memory = False
        self.failure_message: str | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert return_dict is True
        if self.out_of_memory:
            raise torch.OutOfMemoryError("private allocation detail")
        if self.failure_message is not None:
            raise RuntimeError(self.failure_message)
        self.batch_sizes.append(input_ids.shape[0])
        scores = []
        for row, mask in zip(input_ids, attention_mask, strict=True):
            active = row[mask.bool()].tolist()
            self.input_ids.append(tuple(int(token) for token in active))
            separator = active.index(2, 1)
            candidate_start = separator + 2
            first_candidate_token = active[candidate_start]
            scores.append(self.score_by_first_candidate_token[first_candidate_token])
        logits = torch.tensor(scores, dtype=torch.float32).reshape(-1, 1)
        if self.non_finite:
            logits.fill_(float("nan"))
        return SimpleNamespace(logits=logits)


@dataclass
class _LoadedBackend:
    backend: LocalRerankerBackend
    model: _TestModel
    calls: list[tuple[str, Path, dict[str, object]]]


@pytest.fixture
def loaded_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _LoadedBackend:
    tokenizer = _TestTokenizer()
    model = _TestModel()
    calls: list[tuple[str, Path, dict[str, object]]] = []

    def load_tokenizer(path: Path, **kwargs: object) -> _TestTokenizer:
        calls.append(("tokenizer", path, kwargs))
        return tokenizer

    def load_model(path: Path, **kwargs: object) -> _TestModel:
        calls.append(("model", path, kwargs))
        return model

    monkeypatch.setattr(reranker, "_verify_local_artefacts", lambda _: None)
    monkeypatch.setattr(reranker.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        reranker.AutoModelForSequenceClassification,
        "from_pretrained",
        load_model,
    )
    backend = LocalRerankerBackend(tmp_path)
    backend.load()
    return _LoadedBackend(backend, model, calls)


def _request(
    pairs: tuple[tuple[str, str], ...], *, top_k: int | None = None
) -> RerankRequest:
    return RerankRequest(
        query="query",
        candidates=tuple(
            RerankCandidate(candidate_id=candidate_id, candidate_text=text)
            for candidate_id, text in pairs
        ),
        top_k=top_k,
    )


def test_backend_loads_only_the_pinned_local_model_and_closes_idempotently(
    loaded_backend: _LoadedBackend,
) -> None:
    backend = loaded_backend.backend

    assert isinstance(backend, RerankerBackend)
    assert isinstance(backend, MemoryAccountingAdapter)
    assert backend.metadata.supported_operations == (BackendOperation.RERANK,)
    assert backend.metadata.model_name == "BAAI/bge-reranker-v2-m3"
    assert backend.metadata.model_revision == reranker._MODEL_REVISION
    assert (
        backend.metadata.preprocessing_profile_version
        == reranker._PREPROCESSING_PROFILE_VERSION
    )
    assert backend.is_ready
    assert len(loaded_backend.calls) == 2
    for _, path, options in loaded_backend.calls:
        assert path == backend._artefact_dir
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False
    assert loaded_backend.calls[1][2]["torch_dtype"] is torch.float32

    backend.load()
    assert len(loaded_backend.calls) == 2
    backend.close()
    backend.close()
    assert backend.is_closed
    assert not backend.is_ready
    with pytest.raises(BackendClosedError):
        backend.rerank(_request((("one", "a"),)))
    with pytest.raises(BackendClosedError):
        backend.load()


def test_backend_orders_complete_scores_stably_batches_and_applies_top_k(
    loaded_backend: _LoadedBackend,
) -> None:
    pairs = (
        ("id-e", "e"),
        ("id-c", "c"),
        ("id-d", "d"),
        ("id-a", "a"),
        ("id-b", "b"),
    )
    result = loaded_backend.backend.rerank(
        _request(
            pairs,
            top_k=4,
        )
    )

    assert tuple(
        (candidate.candidate_id, candidate.score, candidate.rank)
        for candidate in result.scored_candidates
    ) == (
        ("id-e", 4.0, 1),
        ("id-b", 3.0, 2),
        ("id-d", 3.0, 3),
        ("id-c", 2.0, 4),
    )
    assert loaded_backend.model.batch_sizes == [4, 1]
    assert len(loaded_backend.model.input_ids) == 5
    all_results = loaded_backend.backend.rerank(_request(pairs, top_k=100))
    assert len(all_results.scored_candidates) == len(pairs)
    assert tuple(item.rank for item in all_results.scored_candidates) == (1, 2, 3, 4, 5)


def test_local_backend_adapter_preserves_ret06_source_provenance(
    loaded_backend: _LoadedBackend,
) -> None:
    corpus_id = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
    corpus_version = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002")
    query = RetrievalQuery("query", corpus_version, RetrievalLimit(5))
    candidates = {
        letter: RetrievalCandidate(
            record_id=RecordId.from_str(f"00000000-0000-4000-8000-{number:012d}"),
            kind=CandidateKind.ORIGINAL_PASSAGE,
            content=letter,
            sources=(
                EvidenceSource(
                    document_id=DocumentId.from_str(
                        f"00000000-0000-4000-8000-{number + 10:012d}"
                    ),
                    document_version_id=DocumentVersionId.from_str(
                        f"00000000-0000-4000-8000-{number + 20:012d}"
                    ),
                    source_span_id=SourceSpanId.from_str(
                        f"00000000-0000-4000-8000-{number + 30:012d}"
                    ),
                ),
            ),
        )
        for letter, number in (("a", 100), ("b", 101), ("c", 102))
    }
    fused_hits = tuple(
        EvidenceHit(
            candidate=candidates[letter],
            contributions=(ChannelScore(RetrievalChannel.LEXICAL, rank),),
            rank=rank,
        )
        for rank, letter in enumerate(("a", "b", "c"), start=1)
    )
    fused_result = EvidenceResult(
        corpus_version=corpus_version,
        requested_limit=query.limit,
        hits=fused_hits,
        status=ResultStatus.COMPLETED,
        failures=(),
    )

    outcome = execute_reranking(
        loaded_backend.backend,
        query,
        fused_result,
        corpus_id=corpus_id,
        evidence_count=EvidenceCount(5),
        candidate_limit=RerankCandidateLimit(3),
    )

    assert outcome.result.status is ResultStatus.COMPLETED
    assert outcome.result.requested_limit == query.limit
    assert tuple(hit.rank for hit in outcome.result.hits) == (1, 2, 3)
    assert tuple(hit.candidate for hit in outcome.result.hits) == (
        candidates["b"],
        candidates["c"],
        candidates["a"],
    )
    assert tuple(hit.contributions for hit in outcome.result.hits) == tuple(
        next(hit.contributions for hit in fused_hits if hit.candidate is candidate)
        for candidate in (candidates["b"], candidates["c"], candidates["a"])
    )
    assert tuple(hit.candidate.sources for hit in outcome.result.hits) == tuple(
        candidates[letter].sources for letter in ("b", "c", "a")
    )
    assert outcome.identity.model_name == reranker._MODEL_NAME
    assert outcome.identity.model_revision == reranker._MODEL_REVISION


def test_backend_preserves_query_and_truncates_only_candidate_tail(
    loaded_backend: _LoadedBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reranker, "_MAX_PAIR_TOKENS", 16)
    monkeypatch.setattr(reranker, "_MAX_QUERY_TOKENS", 8)
    query = "QRY"
    candidate_text = "a" * 20

    with structlog.testing.capture_logs() as events:
        result = loaded_backend.backend.rerank(
            RerankRequest(
                query=query,
                candidates=(RerankCandidate("candidate", candidate_text),),
            )
        )

    assert len(result.scored_candidates) == 1
    encoded = loaded_backend.model.input_ids[0]
    query_ids = _TestTokenizer().encode(query, add_special_tokens=False)
    assert encoded[1 : 1 + len(query_ids)] == tuple(query_ids)
    assert len(encoded) == 16
    assert encoded.count(ord("a") + 10) == 9
    event = next(
        item for item in events if item["event"] == "reranker_candidates_truncated"
    )
    assert event["affected_candidate_count"] == 1
    assert event["original_candidate_tokens"] == len(candidate_text)
    assert event["retained_candidate_tokens"] == 9
    encoded_events = json.dumps(events)
    assert query not in encoded_events
    assert candidate_text not in encoded_events


def test_backend_rejects_oversize_candidate_batches_and_queries(
    loaded_backend: _LoadedBackend,
) -> None:
    many_candidates = tuple((str(index), "a") for index in range(101))
    with pytest.raises(BackendRequestError, match="candidate limit"):
        loaded_backend.backend.rerank(_request(many_candidates))

    with pytest.raises(BackendRequestError, match="query exceeds"):
        loaded_backend.backend.rerank(
            RerankRequest(
                query="q" * (reranker._MAX_QUERY_TOKENS + 1),
                candidates=(RerankCandidate("one", "a"),),
            )
        )
    assert loaded_backend.model.batch_sizes == []


def test_backend_rejects_non_finite_or_failed_batches_without_partial_results(
    loaded_backend: _LoadedBackend,
) -> None:
    loaded_backend.model.non_finite = True
    with pytest.raises(BackendExecutionError):
        loaded_backend.backend.rerank(_request((("one", "a"),)))

    loaded_backend.model.non_finite = False
    loaded_backend.model.failure_message = "PRIVATE_MODEL_ERROR"
    with pytest.raises(BackendExecutionError) as failure:
        loaded_backend.backend.rerank(_request((("one", "a"),)))
    assert "PRIVATE_MODEL_ERROR" not in str(failure.value)
    assert failure.value.__suppress_context__


def test_backend_translates_out_of_memory_to_retryable_resource_error(
    loaded_backend: _LoadedBackend,
) -> None:
    loaded_backend.model.out_of_memory = True

    with pytest.raises(BackendResourceError) as failure:
        loaded_backend.backend.rerank(_request((("one", "a"),)))
    assert failure.value.retryable


def test_backend_requires_load_and_hides_loader_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = LocalRerankerBackend(tmp_path)
    measurement = backend.measure_memory()[0]
    assert measurement.category == "model_weights"
    assert measurement.status == "unavailable"
    assert measurement.value_bytes is None
    with pytest.raises(BackendNotReadyError):
        backend.rerank(_request((("one", "a"),)))

    monkeypatch.setattr(reranker, "_verify_local_artefacts", lambda _: None)

    def fail_loader(path: Path, **kwargs: object) -> _TestTokenizer:
        del path, kwargs
        raise RuntimeError("PRIVATE_PATH/model and PRIVATE_DETAIL")

    monkeypatch.setattr(reranker.AutoTokenizer, "from_pretrained", fail_loader)
    with pytest.raises(BackendExecutionError) as failure:
        backend.load()
    assert "PRIVATE_PATH" not in str(failure.value)
    assert "PRIVATE_DETAIL" not in str(failure.value)
    assert failure.value.__suppress_context__
    assert not backend.is_ready


def test_backend_memory_observations_keep_reranker_role(
    loaded_backend: _LoadedBackend,
) -> None:
    measurements = loaded_backend.backend.measure_memory()
    assert len(measurements) == 1
    weight = next(item for item in measurements if item.category == "model_weights")

    assert weight.component_role is MemoryComponentRole.RERANKER
    assert weight.status == "measured"
    assert weight.value_bytes == loaded_backend.model.weight.untyped_storage().nbytes()


def test_backend_failure_logs_omit_query_candidate_and_raw_errors(
    loaded_backend: _LoadedBackend,
) -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="ERROR", format="json"), stream=stream)
    loaded_backend.model.failure_message = "PRIVATE_DEPENDENCY_DETAIL"

    with pytest.raises(BackendExecutionError):
        loaded_backend.backend.rerank(
            RerankRequest(
                query="PRIVATE_QUERY",
                candidates=(RerankCandidate("private-id", "a PRIVATE_CANDIDATE"),),
            )
        )
    log_text = stream.getvalue()
    assert "PRIVATE_QUERY" not in log_text
    assert "PRIVATE_CANDIDATE" not in log_text
    assert "PRIVATE_DEPENDENCY_DETAIL" not in log_text
    records = [json.loads(line) for line in log_text.splitlines()]
    assert records[-1]["event"] == "backend_operation_failed"
    assert records[-1]["failure_category"] == "execution"


def test_artifact_verification_checks_full_inventory_and_digests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    files = {"config.json": b"{}", "assets/model-card.png": b"fixture"}
    for name, payload in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(
        reranker,
        "_EXPECTED_ARTEFACT_DIGESTS",
        tuple(
            (name, hashlib.sha256(payload).hexdigest())
            for name, payload in sorted(files.items())
        ),
    )

    reranker._verify_local_artefacts(tmp_path)
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        reranker._verify_local_artefacts(tmp_path)
    (tmp_path / "extra.txt").unlink()
    (tmp_path / "config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        reranker._verify_local_artefacts(tmp_path)
