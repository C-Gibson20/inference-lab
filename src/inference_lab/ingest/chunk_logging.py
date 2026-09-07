"""Privacy-safe structured logging for observed chunking operations."""

import structlog


def emit_chunk_event(
    module: str,
    method_name: str,
    event: str,
    **fields: object,
) -> None:
    """Emit bounded chunk metadata without changing the operation outcome."""
    try:
        logger = structlog.get_logger(module)
        getattr(logger, method_name)(event, pipeline_stage="chunk", **fields)
    except Exception:
        pass
