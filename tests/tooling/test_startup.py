"""Tests for application startup."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from inference_lab import main


@pytest.fixture
def clean_application_logging() -> Iterator[None]:
    """Remove application-owned handlers after testing process startup."""
    root_logger = logging.getLogger()
    existing_handlers = set(root_logger.handlers)

    yield

    for handler in root_logger.handlers[:]:
        if handler not in existing_handlers:
            root_logger.removeHandler(handler)
            handler.close()


def test_main_configures_logging_and_emits_startup_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    clean_application_logging: None,
) -> None:
    (tmp_path / "config.toml").write_text(
        """
        [models]
        embedder_path = "/models/embedder"
        reranker_path = "/models/reranker"
        generator_path = "/models/generator"

        [runtime]
        vram_budget_gb = 24

        [storage]
        index_dir = "/data/indexes"
        metadata_db_path = "/data/metadata.db"

        [logging]
        level = "INFO"
        format = "json"
        """
    )
    monkeypatch.chdir(tmp_path)

    main()

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    startup_record = next(
        record for record in records if record["event"] == "startup_completed"
    )

    assert startup_record["level"] == "info"
    assert startup_record["module"] == "inference_lab"
