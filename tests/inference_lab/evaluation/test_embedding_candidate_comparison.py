"""Tests for RUN-04 candidate comparison scoring."""

import torch

from scripts.inference_lab.evaluation.compare_local_embedding_candidates import (
    _score,
)


def test_score_reports_recall_and_reciprocal_rank_for_page_labels() -> None:
    pages = [
        {"document_id": "a", "page_number": 1},
        {"document_id": "a", "page_number": 2},
        {"document_id": "b", "page_number": 1},
    ]
    cases = [
        {"relevant_passages": [{"document_id": "a", "page_number": 2}]},
        {"relevant_passages": [{"document_id": "b", "page_number": 1}]},
    ]
    page_vectors = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    result = _score(page_vectors, query_vectors, pages, cases)

    assert result == {
        "case_count": 2,
        "passage_count": 3,
        "recall_at_1": 0.5,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr": 0.75,
    }
