"""Reusable builders and stable identifiers for evaluation tests."""

import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

from inference_lab.evaluation.cases import EvaluationCase, QuestionType
from inference_lab.evaluation.utils import SplitName
from inference_lab.evaluation.versioning import CURRENT_SCHEMA_VERSION

CORPUS_ID = "00000000-0000-4000-8000-000000000001"
CORPUS_VERSION = "00000000-0000-4000-8000-000000000002"
DOCUMENT_ID = "00000000-0000-4000-8000-000000000003"
DOCUMENT_VERSION_ID = "00000000-0000-4000-8000-000000000004"
SOURCE_SPAN_ID = "00000000-0000-4000-8000-000000000005"


class EvidenceIds(TypedDict):
    """Typed keyword fixture for evidence-reference identifiers."""

    evidence_id: str
    corpus_version: str
    document_id: str
    document_version_id: str
    source_span_id: str


EVIDENCE_IDS: EvidenceIds = {
    "evidence_id": "evidence-primary",
    "corpus_version": CORPUS_VERSION,
    "document_id": DOCUMENT_ID,
    "document_version_id": DOCUMENT_VERSION_ID,
    "source_span_id": SOURCE_SPAN_ID,
}


def case_payload(
    case_id: str = "case-001",
    split: str = "development",
) -> dict[str, Any]:
    """Build one valid mutable evaluation-case payload."""
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "case_id": case_id,
        "question": "What assay was used?",
        "question_type": "lookup",
        "expected_answer": {
            "claims": (
                {
                    "key": "assay",
                    "description": "The assay used by the study",
                    "evidence_ids": ("evidence-primary",),
                },
            )
        },
        "evidence": [
            EVIDENCE_IDS
            | {
                "kind": "source_span",
                "role": "authoritative",
            }
        ],
        "abstention": {"expected": False, "reason": "not_applicable"},
        "split": split,
    }


def write_dataset_fixture(
    directory: Path,
    cases: list[dict[str, Any]] | None = None,
    *,
    manifest_overrides: dict[str, Any] | None = None,
    case_text: str | None = None,
) -> Path:
    """Write a mutable evaluation dataset fixture and return its manifest path."""
    case_path = directory / "cases.jsonl"
    case_records = cases if cases is not None else [case_payload()]
    contents = case_text
    if contents is None:
        contents = "".join(
            f"{json.dumps(case, sort_keys=True)}\n" for case in case_records
        )
    case_path.write_text(contents, encoding="utf-8", newline="")

    manifest: dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "dataset_id": "baseline-v1",
        "dataset_version": "1.0.0",
        "description": "Evaluation fixture",
        "corpus_id": CORPUS_ID,
        "corpus_version": CORPUS_VERSION,
        "case_file": "cases.jsonl",
        "case_count": len(case_records),
        "splits": {
            "development": [case["case_id"] for case in case_records],
            "held_out": [],
        },
        "checksums": {
            "cases.jsonl": (
                "sha256:" + hashlib.sha256(case_path.read_bytes()).hexdigest()
            )
        },
        "created_at": "2026-08-22T00:00:00Z",
        "updated_at": "2026-08-22T00:00:00Z",
    }
    manifest.update(manifest_overrides or {})
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def _repeat_question_type(
    question_type: QuestionType,
    count: int,
) -> tuple[QuestionType, ...]:
    return (question_type,) * count


_QUESTION_TYPE_DISTRIBUTION: tuple[QuestionType, ...] = (
    *_repeat_question_type("lookup", 6),
    *_repeat_question_type("comparison", 4),
    *_repeat_question_type("thematic", 4),
    *_repeat_question_type("aggregation", 5),
    *_repeat_question_type("follow_up", 4),
    *_repeat_question_type("difficult_terminology", 3),
    *_repeat_question_type("lookup", 4),
)

_DEFAULT_HELD_OUT_INDEXES = frozenset({1, 7, 11, 15, 20, 24})
_DEFAULT_ABSTENTION_INDEXES = frozenset({2, 8, 12, 16, 21, 24})


def coverage_case(
    index: int,
    question_type: QuestionType,
    *,
    split: SplitName = "development",
    abstention: bool = False,
) -> EvaluationCase:
    """Build one typed case for benchmark coverage tests."""
    case_id = f"case-{index:03d}"
    payload = case_payload(case_id, split)
    payload["question"] = f"What result is recorded for benchmark item {index}?"
    payload["question_type"] = question_type
    payload["context"] = (
        (f"Previous question for benchmark item {index}",)
        if question_type == "follow_up"
        else ()
    )
    payload["abstention"] = {
        "expected": abstention,
        "reason": "evidence_absent" if abstention else "not_applicable",
    }
    payload["evidence"][0].update(
        {
            "document_id": f"00000000-0000-4000-8000-{index:012x}",
            "document_version_id": f"00000000-0000-4001-8000-{index:012x}",
            "source_span_id": f"00000000-0000-4002-8000-{index:012x}",
        }
    )
    if abstention:
        payload["evidence"][0]["role"] = "context"
        payload["expected_answer"] = {}
    else:
        payload["expected_answer"] = {
            "claims": (
                {
                    "key": "result",
                    "description": f"Expected result for benchmark item {index}",
                    "evidence_ids": ("evidence-primary",),
                },
            )
        }
    return EvaluationCase.model_validate(payload)


def benchmark_cases(
    *,
    held_out_indexes: frozenset[int] = _DEFAULT_HELD_OUT_INDEXES,
    abstention_indexes: frozenset[int] = _DEFAULT_ABSTENTION_INDEXES,
    extra_cases: int = 0,
) -> tuple[EvaluationCase, ...]:
    """Build the complete representative benchmark case collection."""
    question_types = _QUESTION_TYPE_DISTRIBUTION + ("lookup",) * extra_cases
    return tuple(
        coverage_case(
            index,
            question_type,
            split="held_out" if index in held_out_indexes else "development",
            abstention=index in abstention_indexes,
        )
        for index, question_type in enumerate(question_types, start=1)
    )
