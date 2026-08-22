"""Privacy-safe structured records for evaluation operations."""

from datetime import datetime
import hashlib
import json
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from inference_lab.evaluation.utils import CaseId


EvaluationStatus = Literal["passed", "failed", "skipped"]


class EvaluationRecord(BaseModel):
    """One privacy-safe result for one evaluation unit.

    Only identifiers, status, timings, and machine-readable issue codes are
    allowed. Content fields such as questions, answers, evidence, prompts,
    and model outputs are intentionally absent from this contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(pattern=r"^eval-[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^run-[0-9a-f]{64}$")
    case_id: CaseId
    status: EvaluationStatus
    duration_ms: float | None = Field(default=None, ge=0)
    warning_code: str | None = Field(default=None, min_length=1)
    error_code: str | None = Field(default=None, min_length=1)
    timestamp: AwareDatetime
    record_schema_version: Literal[1] = 1

    @model_validator(mode="after")
    def validate_record(self) -> "EvaluationRecord":
        """Ensure the record ID and status fields are internally consistent."""
        expected_id = evaluation_record_id(self.run_id, self.case_id)
        if self.record_id != expected_id:
            raise ValueError("record_id does not match run_id and case_id")
        if self.status == "failed" and self.error_code is None:
            raise ValueError("failed evaluation records require error_code")
        if self.status == "passed" and self.error_code is not None:
            raise ValueError("passed evaluation records cannot contain error_code")
        return self


def evaluation_record_id(run_id: str, case_id: str) -> str:
    """Return the stable identifier for a run/case evaluation unit."""

    payload = json.dumps(
        {"case_id": case_id, "run_id": run_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"eval-{hashlib.sha256(payload).hexdigest()}"


class EvaluationRecorder:
    """Emit privacy-safe evaluation records and operational issues."""

    def __init__(self, logger: object) -> None:
        """Create a recorder backed by a structlog-compatible logger."""
        self._logger = logger

    def record(
        self,
        *,
        run_id: str,
        case_id: str,
        status: EvaluationStatus,
        timestamp: AwareDatetime | datetime,
        duration_ms: float | None = None,
        warning_code: str | None = None,
        error_code: str | None = None,
    ) -> EvaluationRecord:
        """Validate and emit one result record without content fields."""

        record = EvaluationRecord(
            record_id=evaluation_record_id(run_id, case_id),
            run_id=run_id,
            case_id=case_id,
            status=status,
            duration_ms=duration_ms,
            warning_code=warning_code,
            error_code=error_code,
            timestamp=timestamp,
        )
        level = (
            "error"
            if record.status == "failed"
            else "warning"
            if record.warning_code
            else "info"
        )
        log_method = getattr(self._logger, level, None)
        if not callable(log_method):
            raise TypeError(f"logger does not provide a {level} method")
        fields = record.model_dump(mode="json")
        event = (
            "evaluation_unit_failed"
            if status == "failed"
            else "evaluation_unit_recorded"
        )
        log_method(event, **fields)
        return record

    def invalid_input(
        self, *, issue_code: str, timestamp: AwareDatetime | datetime
    ) -> None:
        """Emit a warning for a rejected evaluation input without its contents."""

        log_method = getattr(self._logger, "warning", None)
        if not callable(log_method):
            raise TypeError("logger does not provide a warning method")
        log_method(
            "evaluation_input_rejected", issue_code=issue_code, timestamp=timestamp
        )


__all__ = [
    "EvaluationRecord",
    "EvaluationRecorder",
    "EvaluationStatus",
    "evaluation_record_id",
]
