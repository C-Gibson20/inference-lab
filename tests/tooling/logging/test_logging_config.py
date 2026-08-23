import json
import logging
from io import StringIO

import pytest
import structlog

from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig


class FakeStream(StringIO):
    def __init__(self, *, is_tty: bool) -> None:
        super().__init__()
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def test_json_format_emits_structured_record() -> None:
    stream = StringIO()
    configure_logging(
        LoggingConfig(level="INFO", format="json"),
        stream=stream,
    )

    log = structlog.get_logger("ingest")
    log.info("stage_completed", pipeline_stage="parse")

    record = json.loads(stream.getvalue())

    assert record["event"] == "stage_completed"
    assert record["level"] == "info"
    assert record["module"] == "ingest"
    assert record["pipeline_stage"] == "parse"


def test_console_format_is_readable() -> None:
    stream = StringIO()
    configure_logging(
        LoggingConfig(level="INFO", format="console"),
        stream=stream,
    )

    structlog.get_logger("ingest").info(
        "stage_completed",
        pipeline_stage="parse",
    )

    output = stream.getvalue()

    assert "stage_completed" in output
    assert "pipeline_stage=parse" in output


def test_info_level_suppresses_debug() -> None:
    stream = StringIO()
    configure_logging(
        LoggingConfig(level="INFO", format="json"),
        stream=stream,
    )

    log = structlog.get_logger("ingest")
    log.debug("chunk_detail")
    log.info("stage_completed")

    records = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert [record["event"] for record in records] == ["stage_completed"]


def test_required_context_fields_default_to_null() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(format="json"), stream=stream)

    structlog.get_logger("tooling").info("startup_completed")

    record = json.loads(stream.getvalue())

    assert record["module"] == "tooling"
    assert record["document_id"] is None
    assert record["corpus_version"] is None
    assert record["pipeline_stage"] is None


def test_auto_uses_json_for_non_tty() -> None:
    stream = FakeStream(is_tty=False)
    configure_logging(LoggingConfig(format="auto"), stream=stream)

    structlog.get_logger("tooling").info("startup_completed")

    record = json.loads(stream.getvalue())
    assert record["event"] == "startup_completed"


def test_auto_uses_console_for_tty() -> None:
    stream = FakeStream(is_tty=True)
    configure_logging(LoggingConfig(format="auto"), stream=stream)

    structlog.get_logger("tooling").info("startup_completed")

    assert "startup_completed" in stream.getvalue()
    assert not stream.getvalue().lstrip().startswith("{")


def test_stdlib_records_use_structured_renderer() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(format="json"), stream=stream)

    logging.getLogger("dependency").warning("dependency_degraded")

    record = json.loads(stream.getvalue())
    assert record["event"] == "dependency_degraded"
    assert record["level"] == "warning"
    assert record["module"] == "dependency"


def test_sensitive_evaluation_fields_are_rejected() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(format="json"), stream=stream)

    with pytest.raises(ValueError, match="sensitive content fields"):
        structlog.get_logger("evaluation").info(
            "evaluation_unit_recorded", model_output="private answer"
        )

    assert stream.getvalue() == ""


def test_sensitive_event_names_are_rejected() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(format="json"), stream=stream)

    with pytest.raises(ValueError, match="sensitive content fields"):
        structlog.get_logger("evaluation").info("model_output")


def test_reconfiguring_does_not_duplicate_handlers() -> None:
    stream = StringIO()
    config = LoggingConfig(format="json")

    configure_logging(config, stream=stream)
    configure_logging(config, stream=stream)
    structlog.get_logger("tooling").info("startup_completed")

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [record["event"] for record in records] == ["startup_completed"]
