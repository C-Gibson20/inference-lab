"""Deterministic dataset splitting and leakage prevention."""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final, Iterable, get_args

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.contracts import SplitName
from inference_lab.evaluation.normalisation import normalise_for_duplicate_detection
from inference_lab.evaluation.versioning import (
    CURRENT_SPLIT_ALGORITHM_VERSION,
    validate_split_algorithm_version,
)

_SPLIT_ORDER: Final[tuple[SplitName, ...]] = get_args(SplitName)
_DEFAULT_SPLIT_SEED = "inference-lab-default-split"


def _validate_unit_interval(name: str, value: object) -> None:
    """Require a finite numeric value in the closed unit interval."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Deterministic policy for assigning evaluation cases."""

    train_fraction: float
    development_fraction: float
    held_out_fraction: float
    algorithm_version: str = CURRENT_SPLIT_ALGORITHM_VERSION
    seed: str = _DEFAULT_SPLIT_SEED
    near_duplicate_threshold: float = 0.9

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate policy values and version compatibility."""
        self._validate_fractions()
        self._validate_seed()
        _validate_unit_interval(
            "near_duplicate_threshold",
            self.near_duplicate_threshold,
        )
        validate_split_algorithm_version(self.algorithm_version)

    def _validate_fractions(self) -> None:
        """Validate individual fractions and their combined total."""
        for name, fraction in self.fractions.items():
            _validate_unit_interval(name, fraction)

        if not math.isclose(
            sum(self.fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Split fractions must sum to 1.0")

    def _validate_seed(self) -> None:
        """Require a non-empty string seed."""
        if not isinstance(self.seed, str):
            raise TypeError("seed must be a string")
        if not self.seed:
            raise ValueError("seed must not be empty")

    @property
    def fractions(self) -> dict[SplitName, float]:
        """Return split fractions keyed by their persisted split names."""
        return {
            "train": self.train_fraction,
            "development": self.development_fraction,
            "held_out": self.held_out_fraction,
        }


@dataclass(frozen=True, slots=True)
class LeakageGroup:
    """A group of cases that must remain in one split."""

    group_key: str
    cases: tuple[EvaluationCase, ...]


def stable_digest(value: str) -> str:
    """Return a process-independent digest for deterministic assignment."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assignment_key(seed: str, group_key: str) -> str:
    """Return a stable key for deterministic assignment of a group."""
    return stable_digest(f"{seed}\x00{group_key}")


def question_key(question: str) -> str:
    """Return a stable key for a normalised question."""
    return stable_digest(normalise_for_duplicate_detection(question))


def token_set(value: str) -> frozenset[str]:
    """Return normalised tokens for duplicate detection."""
    return frozenset(normalise_for_duplicate_detection(value).split())


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Return the Jaccard similarity between two token sets."""
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def answer_fingerprint(case: EvaluationCase) -> str:
    """Build a stable fingerprint from typed expected-answer data."""
    payload = case.expected_answer.model_dump(mode="json")
    payload.pop("comparison_evidence_ids", None)
    for field_name in ("claims", "numeric_values", "date_values"):
        for expectation in payload[field_name]:
            expectation.pop("evidence_ids", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return stable_digest(encoded)


def evidence_group_keys(case: EvaluationCase) -> frozenset[str]:
    """Return stable evidence keys for leakage grouping."""
    keys: set[str] = set()

    for evidence in case.evidence:
        keys.add(f"document:{evidence.document_id}")

        if evidence.document_version_id is not None:
            keys.add(f"document-version:{evidence.document_version_id}")
        if evidence.source_span_id is not None:
            keys.add(f"source-span:{evidence.source_span_id}")
        if evidence.record_id is not None:
            keys.add(f"record:{evidence.record_id}")

    return frozenset(keys)


class _UnionFind:
    """Small deterministic disjoint-set implementation."""

    def __init__(self, size: int) -> None:
        self._parents = list(range(size))

    def find(self, index: int) -> int:
        """Return the root for an index with path compression."""
        parent = self._parents[index]
        if parent != index:
            self._parents[index] = self.find(parent)
        return self._parents[index]

    def union(self, left: int, right: int) -> None:
        """Join two indexes deterministically."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if left_root < right_root:
            self._parents[right_root] = left_root
        else:
            self._parents[left_root] = right_root


def _validate_leakage_group_inputs(
    cases: tuple[EvaluationCase, ...],
    near_duplicate_threshold: float,
) -> None:
    """Validate leakage-group inputs before building indexes."""
    if not cases:
        raise ValueError("At least one case is required")
    _validate_unit_interval("near_duplicate_threshold", near_duplicate_threshold)

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be unique")


def _grouping_keys(case: EvaluationCase) -> set[str]:
    """Return exact-match grouping keys for one evaluation case."""
    keys = {
        f"question:{question_key(case.question)}",
        *evidence_group_keys(case),
    }
    if case.expected_answer.has_expectations or case.expected_answer.summary:
        keys.add(f"answer:{answer_fingerprint(case)}")
    return keys


def _union_exact_matches(
    cases: tuple[EvaluationCase, ...],
    union_find: _UnionFind,
) -> None:
    """Join cases sharing an exact question, evidence, or answer key."""
    key_owner: dict[str, int] = {}
    for index, case in enumerate(cases):
        for key in _grouping_keys(case):
            owner = key_owner.setdefault(key, index)
            union_find.union(index, owner)


def _union_near_duplicate_questions(
    cases: tuple[EvaluationCase, ...],
    union_find: _UnionFind,
    near_duplicate_threshold: float,
) -> None:
    """Join cases whose normalised question tokens meet the similarity threshold."""
    tokens = [token_set(case.question) for case in cases]
    for left in range(len(cases)):
        for right in range(left + 1, len(cases)):
            if (
                jaccard_similarity(tokens[left], tokens[right])
                >= near_duplicate_threshold
            ):
                union_find.union(left, right)


def _materialise_leakage_groups(
    cases: tuple[EvaluationCase, ...],
    union_find: _UnionFind,
) -> tuple[LeakageGroup, ...]:
    """Build deterministically ordered leakage groups from union-find roots."""
    grouped: dict[int, list[EvaluationCase]] = {}
    for index, case in enumerate(cases):
        grouped.setdefault(union_find.find(index), []).append(case)

    groups = []
    for group_cases in grouped.values():
        sorted_cases = tuple(sorted(group_cases, key=lambda case: case.case_id))
        group_key = stable_digest(
            "group\x00" + "\x00".join(case.case_id for case in sorted_cases)
        )
        groups.append(LeakageGroup(group_key=group_key, cases=sorted_cases))
    return tuple(sorted(groups, key=lambda group: group.group_key))


def build_leakage_groups(
    cases: Iterable[EvaluationCase],
    *,
    near_duplicate_threshold: float = 0.9,
) -> tuple[LeakageGroup, ...]:
    """Build transitive leakage groups from cases and stable annotations."""
    ordered_cases = tuple(cases)
    _validate_leakage_group_inputs(ordered_cases, near_duplicate_threshold)
    union_find = _UnionFind(len(ordered_cases))
    _union_exact_matches(ordered_cases, union_find)
    _union_near_duplicate_questions(
        ordered_cases,
        union_find,
        near_duplicate_threshold,
    )
    return _materialise_leakage_groups(ordered_cases, union_find)


def validate_no_leakage(
    cases: Iterable[EvaluationCase],
    *,
    near_duplicate_threshold: float = 0.9,
) -> None:
    """Reject leakage groups that cross persisted split assignments."""
    groups = build_leakage_groups(
        cases,
        near_duplicate_threshold=near_duplicate_threshold,
    )

    for group in groups:
        splits = {case.split for case in group.cases}
        if len(splits) > 1:
            case_ids = ", ".join(case.case_id for case in group.cases)
            raise ValueError(f"Leakage group crosses splits: {case_ids}")


def generate_splits(
    cases: Iterable[EvaluationCase],
    policy: SplitPolicy,
) -> tuple[EvaluationCase, ...]:
    """Assign leakage groups to deterministic train, development, and held-out splits."""
    ordered_cases = tuple(cases)
    groups = build_leakage_groups(
        ordered_cases,
        near_duplicate_threshold=policy.near_duplicate_threshold,
    )
    targets: dict[SplitName, float] = {
        split: fraction * len(ordered_cases)
        for split, fraction in policy.fractions.items()
    }
    counts: dict[SplitName, int] = dict.fromkeys(_SPLIT_ORDER, 0)
    assignments: dict[str, SplitName] = {}

    ranked_groups = sorted(
        groups,
        key=lambda group: assignment_key(policy.seed, group.group_key),
    )
    eligible_splits: tuple[SplitName, ...] = tuple(
        split for split in _SPLIT_ORDER if policy.fractions[split] > 0.0
    )

    for group in ranked_groups:
        split: SplitName = min(
            eligible_splits,
            key=lambda candidate: (
                counts[candidate] - targets[candidate],
                _SPLIT_ORDER.index(candidate),
            ),
        )
        assignments.update({case.case_id: split for case in group.cases})
        counts[split] += len(group.cases)

    assigned_cases = tuple(
        EvaluationCase.model_validate(
            case.model_dump(mode="python") | {"split": assignments[case.case_id]}
        )
        for case in ordered_cases
    )
    validate_no_leakage(
        assigned_cases,
        near_duplicate_threshold=policy.near_duplicate_threshold,
    )
    return assigned_cases
