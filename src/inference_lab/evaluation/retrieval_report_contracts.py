"""Immutable aggregate metadata for published retrieval evaluations."""

from __future__ import annotations

from dataclasses import dataclass

from inference_lab.evaluation._validation import is_sha256_checksum
from inference_lab.evaluation.retrieval_contracts import (
    RetrievalEvaluationAggregate,
    RetrievalEvaluationContractError,
)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Deterministic aggregate report metadata for one evaluation run."""

    run_fingerprint: str
    result_digest: str
    aggregate: RetrievalEvaluationAggregate
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate report identities, aggregate values, and limitations."""
        for field_name, value in (
            ("run_fingerprint", self.run_fingerprint),
            ("result_digest", self.result_digest),
        ):
            if not isinstance(value, str) or not is_sha256_checksum(value):
                raise RetrievalEvaluationContractError(
                    f"{field_name} must be a SHA-256 checksum."
                )
        if not isinstance(self.aggregate, RetrievalEvaluationAggregate):
            raise TypeError("aggregate must be a RetrievalEvaluationAggregate.")
        if not isinstance(self.limitations, tuple):
            raise TypeError("limitations must be a tuple.")
        if any(
            not isinstance(limitation, str) or not limitation.strip()
            for limitation in self.limitations
        ):
            raise RetrievalEvaluationContractError(
                "limitations must contain non-blank strings."
            )
        if len(self.limitations) != len(set(self.limitations)):
            raise RetrievalEvaluationContractError("limitations must be unique.")

    def as_dict(self) -> dict[str, object]:
        """Return the path-free aggregate report payload."""
        return {
            "run_fingerprint": self.run_fingerprint,
            "result_digest": self.result_digest,
            "aggregate": self.aggregate.as_dict(),
            "limitations": list(self.limitations),
        }
