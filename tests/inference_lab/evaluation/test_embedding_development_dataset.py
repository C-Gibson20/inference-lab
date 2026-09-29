"""Validate RUN-04 development queries against their corpus manifest."""

import hashlib
import json
from pathlib import Path
from typing import cast

_BENCHMARK_DIR = Path(__file__).parents[3] / "benchmarks" / "run-04"


def test_expanded_embedding_cases_match_the_pinned_corpus_manifest() -> None:
    manifest = cast(
        dict[str, object],
        json.loads(
            (_BENCHMARK_DIR / "development-corpus-manifest.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    dataset_path = _BENCHMARK_DIR / cast(str, manifest["dataset_file"])
    dataset_bytes = dataset_path.read_bytes()
    assert hashlib.sha256(dataset_bytes).hexdigest() == manifest["dataset_sha256"]

    documents = cast(list[dict[str, object]], manifest["documents"])
    inventory = {cast(str, item["document_id"]): item for item in documents}
    cases = [json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines()]
    assert len(cases) == manifest["case_count"]
    case_ids = [cast(str, case["case_id"]) for case in cases]
    assert len(set(case_ids)) == len(case_ids)

    for case in cases:
        assert isinstance(case["query"], str) and case["query"].strip()
        relevant_passages = cast(list[dict[str, object]], case["relevant_passages"])
        assert relevant_passages
        for passage in relevant_passages:
            document_id = cast(str, passage["document_id"])
            document = inventory[document_id]
            page_number = cast(int, passage["page_number"])
            page_count = cast(int, document["page_count"])
            pages_without_text = cast(
                list[int], document["pages_without_extracted_text"]
            )
            assert 1 <= page_number <= page_count
            assert page_number not in pages_without_text
