"""Local, provenance-preserving paper and patent inference tool."""

import structlog

from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import get_settings


def main() -> None:
    """Entry point for the command-line interface."""
    settings = get_settings()
    configure_logging(settings.logging)

    log = structlog.get_logger(__name__)
    log.info("startup_completed")
