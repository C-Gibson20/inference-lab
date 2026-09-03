"""Privacy-safe logging shared by SQLite repository lifecycle owners."""

import structlog

_LOGGER = structlog.get_logger("inference_lab.storage._sqlite_repository")


def safe_log(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded repository metadata without changing storage outcomes."""
    try:
        getattr(_LOGGER, method_name)(event, pipeline_stage="storage", **fields)
    except Exception:
        pass
