"""Shared privacy-safe structured logging for source discovery."""

import structlog

from inference_lab.ingest.discovery_failures import DiscoveryFailure


def emit_discovery_event(
    module: str,
    method_name: str,
    event: str,
    **fields: object,
) -> None:
    """Emit bounded discovery metadata without changing pipeline outcomes."""
    try:
        logger = structlog.get_logger(module)
        getattr(logger, method_name)(event, pipeline_stage="ingest", **fields)
    except Exception:
        pass


def failure_log_fields(failure: DiscoveryFailure) -> dict[str, object]:
    """Return the bounded failure fields permitted in operational logs."""
    fields: dict[str, object] = {
        "source_ordinal": failure.ordinal,
        "error_category": failure.category.value,
        "failure_scope": failure.scope.value,
        "failure_stage": failure.stage.value,
        "retryable": failure.retryable,
    }
    if failure.file_kind is not None:
        fields["file_kind"] = failure.file_kind.value
    if failure.media_type is not None:
        fields["media_type"] = failure.media_type
    return fields
