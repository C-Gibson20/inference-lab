#!/usr/bin/env python3
"""Compare pinned local embedding candidates on RUN-04 development cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer

from inference_lab.backends.runtime import (
    EmbeddingOptions,
    EmbeddingRequest,
    QwenEmbeddingBackend,
)

_BENCHMARK_DIR = Path(__file__).parents[3] / "benchmarks" / "run-04"
_QWEN_WEIGHT_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
_BGE_WEIGHT_SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
_QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
_BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_BATCH_SIZE = 2
_MAX_TOKENS = 4096


def _sha256(path: Path) -> str:
    """Return a streamed SHA-256 digest for one local file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_data(
    corpus_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Verify the corpus files and load all text-bearing pages and cases."""
    manifest = cast(
        dict[str, Any],
        json.loads(
            (_BENCHMARK_DIR / "development-corpus-manifest.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    dataset_path = _BENCHMARK_DIR / cast(str, manifest["dataset_file"])
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError
    cases = [json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines()]
    if len(cases) != manifest["case_count"]:
        raise ValueError

    expected = {
        cast(str, item["document_id"]): cast(dict[str, Any], item)
        for item in cast(list[dict[str, Any]], manifest["documents"])
    }
    pdfs_by_digest: dict[str, Path] = {}
    for pdf_path in corpus_dir.glob("*.pdf"):
        digest = _sha256(pdf_path)
        if digest in expected:
            if digest in pdfs_by_digest:
                raise ValueError
            pdfs_by_digest[digest] = pdf_path
    if pdfs_by_digest.keys() != expected.keys():
        raise ValueError

    pages: list[dict[str, Any]] = []
    for document_id, document in expected.items():
        reader = PdfReader(pdfs_by_digest[document_id])
        if len(reader.pages) != document["page_count"]:
            raise ValueError
        blank_pages = set(document["pages_without_extracted_text"])
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if page_number in blank_pages or not text.strip():
                continue
            pages.append(
                {
                    "document_id": document_id,
                    "page_number": page_number,
                    "text": text,
                }
            )

    page_keys = {(page["document_id"], page["page_number"]) for page in pages}
    for case in cases:
        if not case["query"].strip():
            raise ValueError
        for passage in case["relevant_passages"]:
            key = (passage["document_id"], passage["page_number"])
            if key not in page_keys:
                raise ValueError
    expected_page_count = sum(
        document["text_extractable_page_count"] for document in expected.values()
    )
    if len(pages) != expected_page_count:
        raise ValueError
    return pages, cases, expected


def _embed_qwen(
    backend: QwenEmbeddingBackend,
    texts: Sequence[str],
    *,
    purpose: Literal["query", "document"],
) -> tuple[torch.Tensor, int, float]:
    """Embed inputs with the already-loaded production Qwen backend."""
    started = time.perf_counter()
    result = backend.embed(
        EmbeddingRequest(
            input_texts=tuple(texts),
            purpose=purpose,
            options=EmbeddingOptions(normalise=True),
        )
    )
    elapsed = time.perf_counter() - started
    token_count = result.usage.input_tokens if result.usage is not None else 0
    vectors = torch.tensor(result.vectors, dtype=torch.float32)
    return vectors, token_count, elapsed


def _embed_bge(
    model_dir: Path,
    page_texts: Sequence[str],
    query_texts: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, int, int, float, float]:
    """Embed pages and queries with one BGE load and masked-mean pooling."""
    weight = model_dir / "pytorch_model.bin"
    if _sha256(weight) != _BGE_WEIGHT_SHA256:
        raise ValueError
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.to("cpu")

    def encode(texts: Sequence[str]) -> tuple[torch.Tensor, int, float]:
        batches: list[torch.Tensor] = []
        token_count = 0
        started = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(texts), _BATCH_SIZE):
                batch = texts[offset : offset + _BATCH_SIZE]
                inputs = tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=_MAX_TOKENS,
                    return_tensors="pt",
                )
                token_count += int(inputs["attention_mask"].sum().item())
                output = model(**inputs)
                mask = (
                    inputs["attention_mask"]
                    .unsqueeze(-1)
                    .to(dtype=output.last_hidden_state.dtype)
                )
                pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp(min=1)
                batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        elapsed = time.perf_counter() - started
        return torch.cat(batches, dim=0), token_count, elapsed

    page_vectors, page_tokens, page_seconds = encode(page_texts)
    query_vectors, query_tokens, query_seconds = encode(query_texts)
    return (
        page_vectors,
        query_vectors,
        page_tokens,
        query_tokens,
        page_seconds,
        query_seconds,
    )


def _score(
    page_vectors: torch.Tensor,
    query_vectors: torch.Tensor,
    pages: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
) -> dict[str, float | int]:
    """Compute deterministic page-level Recall@K and MRR."""
    similarities = query_vectors @ page_vectors.T
    hits = {1: 0, 5: 0, 10: 0}
    reciprocal_rank = 0.0
    for query_index, case in enumerate(cases):
        relevant = {
            (passage["document_id"], passage["page_number"])
            for passage in case["relevant_passages"]
        }
        ranking = sorted(
            range(len(pages)),
            key=lambda index: (
                -float(similarities[query_index, index].item()),
                pages[index]["document_id"],
                pages[index]["page_number"],
            ),
        )
        rank = next(
            (
                position
                for position, index in enumerate(ranking, start=1)
                if (pages[index]["document_id"], pages[index]["page_number"])
                in relevant
            ),
            None,
        )
        if rank is None:
            raise ValueError
        reciprocal_rank += 1 / rank
        for cutoff in hits:
            hits[cutoff] += int(rank <= cutoff)
    total = len(cases)
    return {
        "case_count": total,
        "passage_count": len(pages),
        "recall_at_1": hits[1] / total,
        "recall_at_5": hits[5] / total,
        "recall_at_10": hits[10] / total,
        "mrr": reciprocal_rank / total,
    }


def _candidate_result(
    *,
    name: str,
    model_dir: Path,
    pages: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Embed pages and queries, then return path-free aggregate evidence."""
    if name == "qwen3-embedding-0.6b":
        backend = QwenEmbeddingBackend(model_dir)
        backend.load()
        page_vectors, page_tokens, page_seconds = _embed_qwen(
            backend, [page["text"] for page in pages], purpose="document"
        )
        query_vectors, query_tokens, query_seconds = _embed_qwen(
            backend, [case["query"] for case in cases], purpose="query"
        )
        backend.close()
    else:
        (
            page_vectors,
            query_vectors,
            page_tokens,
            query_tokens,
            page_seconds,
            query_seconds,
        ) = _embed_bge(
            model_dir,
            [page["text"] for page in pages],
            [case["query"] for case in cases],
        )
    result: dict[str, Any] = _score(page_vectors, query_vectors, pages, cases)
    input_tokens = page_tokens + query_tokens
    inference_seconds = page_seconds + query_seconds
    result.update(
        {
            "inference_seconds": round(inference_seconds, 2),
            "input_tokens": input_tokens,
            "tokens_per_second": round(input_tokens / inference_seconds, 2),
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--qwen-model-dir", type=Path, required=True)
    parser.add_argument("--bge-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run both candidates locally and write aggregate metrics only."""
    args = _parser().parse_args(argv)
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pages, cases, documents = _load_data(args.corpus_dir)
        print(
            f"verified corpus: {len(pages)} text-bearing pages, "
            f"{len(cases)} development cases",
            file=sys.stderr,
        )
        print("running Qwen3-Embedding-0.6B", file=sys.stderr)
        results = {
            "qwen3-embedding-0.6b": _candidate_result(
                name="qwen3-embedding-0.6b",
                model_dir=args.qwen_model_dir,
                pages=pages,
                cases=cases,
            ),
        }
        print("running BGE-M3", file=sys.stderr)
        results["bge-m3"] = _candidate_result(
            name="bge-m3",
            model_dir=args.bge_model_dir,
            pages=pages,
            cases=cases,
        )
        report = {
            "report_schema_version": "1.0.0",
            "dataset_sha256": hashlib.sha256(
                (_BENCHMARK_DIR / "development-cases-expanded.jsonl").read_bytes()
            ).hexdigest(),
            "document_count": len(documents),
            "physical_page_count": sum(
                document["page_count"] for document in documents.values()
            ),
            "text_bearing_page_count": len(pages),
            "max_input_tokens": _MAX_TOKENS,
            "batch_size": _BATCH_SIZE,
            "device": "cpu",
            "dtype": "float32",
            "pooling": {
                "qwen3-embedding-0.6b": "last-token-v1",
                "bge-m3": "attention-mask-weighted-mean-v1",
            },
            "query_prefix": {
                "qwen3-embedding-0.6b": "pinned RUN-04 profile",
                "bge-m3": "none",
            },
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "torch": version("torch"),
                "transformers": version("transformers"),
                "pypdf": version("pypdf"),
            },
            "candidates": results,
            "model_revisions": {
                "qwen3-embedding-0.6b": _QWEN_REVISION,
                "bge-m3": _BGE_REVISION,
            },
            "weight_sha256": {
                "qwen3-embedding-0.6b": _QWEN_WEIGHT_SHA256,
                "bge-m3": _BGE_WEIGHT_SHA256,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        print("local embedding comparison failed", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "output_written": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
