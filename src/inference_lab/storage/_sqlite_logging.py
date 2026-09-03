"""Privacy-safe logging shared by SQLite repository lifecycle owners."""

import structlog

from inference_lab.storage.errors import StorageError

_LOGGER = structlog.get_logger("inference_lab.storage._sqlite_repository")


def safe_log(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded repository metadata without changing storage outcomes."""
    try:
        getattr(_LOGGER, method_name)(event, pipeline_stage="storage", **fields)
    except Exception:
        pass


def safe_log_failure(
    event: str,
    error: StorageError,
    **fields: object,
) -> None:
    """Emit a typed storage failure without changing its outcome."""
    failure_fields = dict(fields)
    failure_fields.update(
        error_category=error.category.value,
        retryable=error.retryable,
    )
    safe_log("error", event, **failure_fields)
