"""Contract tests for the local Qwen embedding backend."""

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inference_lab.backends.runtime import (
    BackendClosedError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendOperation,
    BackendRequestError,
    BackendResourceError,
    EmbeddingOptions,
    EmbeddingRequest,
    QwenEmbeddingBackend,
    qwen_embedding,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig


class _TestTokenizer:
    """Small deterministic stand-in for the pinned fast Qwen tokeniser."""

    def __init__(self) -> None:
        self.pad_token_id: int | None = 0
        self.padding_side = "right"

    def __len__(self) -> int:
        return 151669

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [ord(character) + 1 for character in text]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 1

    def build_inputs_with_special_tokens(self, token_ids_0: list[int]) -> list[int]:
        return [1, *token_ids_0]

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
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for item in encoded_inputs:
            token_ids = item["input_ids"]
            padding_count = longest - len(token_ids)
            ids.append([0] * padding_count + token_ids)
            masks.append([0] * padding_count + [1] * len(token_ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class _TestModel:
    """Return token-derived hidden states and record observed input batches."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=1024)
        self.batches: list[tuple[int, ...]] = []
        self.non_finite = False
        self.out_of_memory = False

    def eval(self) -> "_TestModel":
        return self

    def to(self, device: str) -> "_TestModel":
        assert device == "cpu"
        return self

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        if self.out_of_memory:
            raise torch.OutOfMemoryError("private test failure")
        self.batches.extend(
            tuple(int(token) for token in row[mask.bool()].tolist())
            for row, mask in zip(input_ids, attention_mask, strict=True)
        )
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 1024)
        if self.non_finite:
            hidden.fill_(float("nan"))
        return SimpleNamespace(last_hidden_state=hidden)


@dataclass
class _LoadedBackend:
    backend: QwenEmbeddingBackend
    model: _TestModel
    load_calls: list[tuple[str, dict[str, object]]]


@pytest.fixture
def loaded_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> _LoadedBackend:
    tokenizer = _TestTokenizer()
    model = _TestModel()
    calls: list[tuple[str, dict[str, object]]] = []

    def load_tokenizer(path: Path, **kwargs: object) -> _TestTokenizer:
        calls.append(("tokenizer", {"path": path, **kwargs}))
        return tokenizer

    def load_model(path: Path, **kwargs: object) -> _TestModel:
        calls.append(("model", {"path": path, **kwargs}))
        return model

    monkeypatch.setattr(qwen_embedding, "_verify_local_artefacts", lambda _: None)
    monkeypatch.setattr(qwen_embedding.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(qwen_embedding.AutoModel, "from_pretrained", load_model)
    backend = QwenEmbeddingBackend(tmp_path)
    backend.load()
    return _LoadedBackend(backend, model, calls)


def test_request_defaults_to_document_purpose_and_validates_purpose() -> None:
    request = EmbeddingRequest(input_texts=("passage",))
    assert request.purpose == "document"
    assert EmbeddingRequest(input_texts=("query",), purpose="query").purpose == "query"
    with pytest.raises(ValueError, match="query.*document"):
        EmbeddingRequest(input_texts=("text",), purpose="other")  # type: ignore[arg-type]


def test_qwen_backend_loads_only_verified_local_files_and_closes_idempotently(
    loaded_backend: _LoadedBackend,
) -> None:
    backend = loaded_backend.backend
    assert backend.is_ready
    assert backend.metadata.supported_operations == (BackendOperation.EMBED,)
    assert backend.metadata.embedding_dimensions == 1024
    assert len(loaded_backend.load_calls) == 2
    for _, options in loaded_backend.load_calls:
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False
    model_options = loaded_backend.load_calls[1][1]
    assert model_options["torch_dtype"] is torch.float32

    backend.load()
    backend.close()
    backend.close()
    assert backend.is_closed
    assert not backend.is_ready
    with pytest.raises(BackendClosedError):
        backend.embed(EmbeddingRequest(input_texts=("query",)))


def test_qwen_backend_prepends_query_instruction_and_preserves_batch_order(
    loaded_backend: _LoadedBackend,
) -> None:
    backend = loaded_backend.backend
    documents = backend.embed(EmbeddingRequest(input_texts=("a", "bb", "ccc")))
    queries = backend.embed(EmbeddingRequest(input_texts=("a",), purpose="query"))

    assert tuple(vector[-1] for vector in documents.vectors) == (98.0, 99.0, 100.0)
    assert len(loaded_backend.model.batches) == 4
    assert loaded_backend.model.batches[0][-1] == ord("a") + 1
    assert loaded_backend.model.batches[-1][-1] == ord("a") + 1
    assert len(loaded_backend.model.batches[-1]) > len("a") + 1
    assert documents.usage is not None
    assert documents.usage.input_tokens == sum(
        map(len, loaded_backend.model.batches[:3])
    )
    assert queries.embedding_profile.query_prefix.startswith("Instruct: Given")
    assert queries.embedding_profile.document_prefix == ""


def test_qwen_backend_honours_dimensions_normalisation_and_rejects_whitespace(
    loaded_backend: _LoadedBackend,
) -> None:
    backend = loaded_backend.backend
    result = backend.embed(
        EmbeddingRequest(
            input_texts=("text",),
            options=EmbeddingOptions(dimensions=32, normalise=True),
        )
    )
    assert len(result.vectors[0]) == 32
    assert result.model_metadata.embedding_dimensions == 32
    assert result.embedding_profile.l2_normalised is True
    assert sum(value**2 for value in result.vectors[0]) == pytest.approx(1.0)
    with pytest.raises(BackendRequestError):
        backend.embed(EmbeddingRequest(input_texts=(" \t ",)))
    with pytest.raises(BackendRequestError):
        backend.embed(
            EmbeddingRequest(
                input_texts=("text",), options=EmbeddingOptions(dimensions=16)
            )
        )


def test_qwen_backend_truncates_text_tail_and_rejects_non_finite_vectors(
    loaded_backend: _LoadedBackend,
) -> None:
    backend = loaded_backend.backend
    tokenizer = backend._tokenizer
    assert tokenizer is not None
    prepared, counts, truncations = backend._prepare_inputs(
        tokenizer,
        ("x" * 4100,),
        prefix=qwen_embedding._QUERY_PREFIX,
    )
    assert len(prepared[0]) == 4096
    assert counts == [4096]
    assert truncations == (1, 4100 + len(qwen_embedding._QUERY_PREFIX) + 1, 4096)
    prefix_tokens = tokenizer.encode(
        qwen_embedding._QUERY_PREFIX,
        add_special_tokens=False,
    )
    assert prepared[0][: len(prefix_tokens) + 1] == [1, *prefix_tokens]

    loaded_backend.model.non_finite = True
    with pytest.raises(BackendExecutionError):
        backend.embed(EmbeddingRequest(input_texts=("text",)))


def test_qwen_truncation_log_contains_counts_without_input_text(
    loaded_backend: _LoadedBackend,
) -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="WARNING", format="json"), stream=stream)
    secret_text = "PRIVATE_QUERY_" + "x" * 4100
    loaded_backend.backend.embed(EmbeddingRequest(input_texts=(secret_text,)))

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    truncation = next(
        record for record in records if record["event"] == "embedding_input_truncated"
    )
    assert truncation["affected_input_count"] == 1
    assert truncation["original_token_count"] == 4115
    assert truncation["retained_token_count"] == 4096
    assert truncation["token_limit"] == 4096
    assert secret_text not in stream.getvalue()


def test_qwen_backend_requires_lifecycle_before_embedding(tmp_path: Path) -> None:
    backend = QwenEmbeddingBackend(tmp_path)
    with pytest.raises(BackendNotReadyError):
        backend.embed(EmbeddingRequest(input_texts=("query",)))


def test_qwen_backend_translates_torch_memory_failure(
    loaded_backend: _LoadedBackend,
) -> None:
    loaded_backend.model.out_of_memory = True
    with pytest.raises(BackendResourceError):
        loaded_backend.backend.embed(EmbeddingRequest(input_texts=("query",)))
