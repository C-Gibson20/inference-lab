"""Bounded semantic comparison of retrieval-evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.evaluation._validation import is_sha256_checksum
from inference_lab.evaluation.retrieval_contracts import (
    RetrievalEvaluationAggregate,
    RetrievalEvaluationRunIdentity,
)
from inference_lab.evaluation.retrieval_execution import RetrievalEvaluationRunResult
from inference_lab.evaluation.retrieval_report_codec import (
    retrieval_result_digest,
    serialise_case_results,
)


class RetrievalComparisonCategory(StrEnum):
    """Closed mismatch categories exposed by repeat-run comparison."""

    RUN_IDENTITY = "run_identity"
    CASE_RESULTS = "case_results"
    AGGREGATE = "aggregate"
    RESULT_DIGEST = "result_digest"


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationComparison:
    """Safe result of comparing two complete retrieval-evaluation runs."""

    equivalent: bool
    mismatch_categories: tuple[RetrievalComparisonCategory, ...]
    left_run_fingerprint: str
    right_run_fingerprint: str
    left_result_digest: str
    right_result_digest: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the bounded comparison result."""
        if type(self.equivalent) is not bool:
            raise TypeError("equivalent must be a boolean.")
        if not isinstance(self.mismatch_categories, tuple):
            raise TypeError("mismatch_categories must be a tuple.")
        if any(
            not isinstance(category, RetrievalComparisonCategory)
            for category in self.mismatch_categories
        ):
            raise TypeError(
                "mismatch_categories must contain only RetrievalComparisonCategory values."
            )
        if len(self.mismatch_categories) != len(set(self.mismatch_categories)):
            raise ValueError("mismatch_categories must be unique.")
        if self.equivalent != (not self.mismatch_categories):
            raise ValueError("equivalent must match mismatch_categories.")
        for field_name, value in (
            ("left_run_fingerprint", self.left_run_fingerprint),
            ("right_run_fingerprint", self.right_run_fingerprint),
            ("left_result_digest", self.left_result_digest),
            ("right_result_digest", self.right_result_digest),
        ):
            if not isinstance(value, str) or not is_sha256_checksum(value):
                raise ValueError(f"{field_name} must be a SHA-256 checksum.")

    def as_dict(self) -> dict[str, object]:
        """Return bounded comparison data suitable for local CLI output."""
        return {
            "equivalent": self.equivalent,
            "mismatch_categories": [
                category.value for category in self.mismatch_categories
            ],
            "left_run_fingerprint": self.left_run_fingerprint,
            "right_run_fingerprint": self.right_run_fingerprint,
            "left_result_digest": self.left_result_digest,
            "right_result_digest": self.right_result_digest,
        }


def compare_retrieval_evaluations(
    left: RetrievalEvaluationRunResult,
    right: RetrievalEvaluationRunResult,
) -> RetrievalEvaluationComparison:
    """Compare semantic run identity, cases, aggregate, and result digest."""
    if not isinstance(left, RetrievalEvaluationRunResult):
        raise TypeError("left must be a RetrievalEvaluationRunResult.")
    if not isinstance(right, RetrievalEvaluationRunResult):
        raise TypeError("right must be a RetrievalEvaluationRunResult.")
    left_fingerprint = _require_fingerprint(left.identity)
    right_fingerprint = _require_fingerprint(right.identity)
    left_digest = retrieval_result_digest(left)
    right_digest = retrieval_result_digest(right)

    mismatches: list[RetrievalComparisonCategory] = []
    if _identity_bytes(left.identity) != _identity_bytes(right.identity):
        mismatches.append(RetrievalComparisonCategory.RUN_IDENTITY)
    if serialise_case_results(left) != serialise_case_results(right):
        mismatches.append(RetrievalComparisonCategory.CASE_RESULTS)
    if _aggregate_bytes(left.aggregate) != _aggregate_bytes(right.aggregate):
        mismatches.append(RetrievalComparisonCategory.AGGREGATE)
    if left_digest != right_digest:
        mismatches.append(RetrievalComparisonCategory.RESULT_DIGEST)

    return RetrievalEvaluationComparison(
        equivalent=not mismatches,
        mismatch_categories=tuple(mismatches),
        left_run_fingerprint=left_fingerprint,
        right_run_fingerprint=right_fingerprint,
        left_result_digest=left_digest,
        right_result_digest=right_digest,
    )


def _identity_bytes(identity: RetrievalEvaluationRunIdentity) -> bytes:
    """Return canonical semantic identity bytes without filesystem metadata."""
    return canonical_json_bytes(cast(JsonObject, identity.as_dict()))


def _aggregate_bytes(aggregate: RetrievalEvaluationAggregate) -> bytes:
    """Return canonical aggregate bytes used by comparison."""
    return canonical_json_bytes(cast(JsonObject, aggregate.as_dict()))


def _require_fingerprint(identity: RetrievalEvaluationRunIdentity) -> str:
    """Return the constructor-guaranteed run fingerprint with a type guard."""
    if not isinstance(identity.run_fingerprint, str):
        raise ValueError("run identity must contain a fingerprint.")
    return identity.run_fingerprint
