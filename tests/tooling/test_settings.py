from pathlib import Path
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from inference_lab.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_settings_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    (tmp_path / "config.toml").write_text(
        """
        [models]
        embedder_path = "/models/qwen3-embedding-0.6b"
        reranker_path = "/models/bge-reranker-v2"
        generator_path = "/models/qwen3-27b-q4"

        [runtime]
        vram_budget_gb = 24
        max_context_tokens = 8192

        [storage]
        index_dir = "/data/indexes"
        metadata_db_path = "/data/metadata.db"

        [logging]
        level = "DEBUG"
        format = "json"
        """
    )
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_explicit_runtime_settings_override_toml() -> None:
    configured = Settings(runtime={"vram_budget_gb": 12})

    assert configured.runtime.vram_budget_gb == 12
    assert configured.runtime.max_context_tokens == 8192


def test_explicit_nested_model_setting_overrides_toml() -> None:
    configured = Settings(models={"generator_path": "/tmp/test-generator"})

    assert configured.models.generator_path == Path("/tmp/test-generator")
    assert configured.models.embedder_path == Path("/models/qwen3-embedding-0.6b")


def test_nested_environment_setting_overrides_toml(monkeypatch) -> None:
    monkeypatch.setenv("MODELS__GENERATOR_PATH", "/tmp/test-generator")

    configured = get_settings()

    assert configured.models.generator_path == Path("/tmp/test-generator")


def test_nested_runtime_environment_setting_overrides_toml(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME__VRAM_BUDGET_GB", "12")

    configured = get_settings()

    assert configured.runtime.vram_budget_gb == 12


def test_toml_configuration_loads() -> None:
    configured = get_settings()

    assert configured.models.generator_path == Path("/models/qwen3-27b-q4")
    assert configured.runtime.vram_budget_gb == 24
    assert configured.storage.metadata_db_path == Path("/data/metadata.db")


def test_settings_can_be_reloaded(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME__VRAM_BUDGET_GB", "12")
    config_1 = get_settings()

    monkeypatch.setenv("RUNTIME__VRAM_BUDGET_GB", "16")
    get_settings.cache_clear()
    config_2 = get_settings()

    assert config_1.runtime.vram_budget_gb == 12
    assert config_2.runtime.vram_budget_gb == 16


def test_missing_required_setting_is_rejected(tmp_path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
        [runtime]
        vram_budget_gb = 24
        """
    )

    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["toml_file"] = config_file

    with pytest.raises(ValidationError):
        TestSettings()


def test_invalid_setting_type_is_rejected(tmp_path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
        [runtime]
        vram_budget_gb = "not an int"
        """
    )

    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["toml_file"] = config_file

    with pytest.raises(ValidationError):
        TestSettings()


def test_logging_settings_default() -> None:
    configured = get_settings()

    assert configured.logging.level == "INFO"
    assert configured.logging.format == "auto"


def test_logging_settings_load_from_toml() -> None:
    configured = get_settings()

    assert configured.logging.level == "DEBUG"
    assert configured.logging.format == "json"


def test_explicit_logging_settings_override_toml() -> None:
    configured = Settings(logging={"level": "WARNING"})

    assert configured.logging.level == "WARNING"
    assert configured.logging.format == "json"


def test_nested_logging_level_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOGGING__LEVEL", "ERROR")

    configured = get_settings()

    assert configured.logging.level == "ERROR"
    assert configured.logging.format == "json"


def test_nested_logging_format_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOGGING__FORMAT", "console")

    configured = get_settings()

    assert configured.logging.level == "DEBUG"
    assert configured.logging.format == "console"


def test_invalid_logging_level_is_rejected(tmp_path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
        [logging]
        level = "TRACE"
        """
    )

    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["toml_file"] = config_file

    with pytest.raises(ValidationError):
        TestSettings()


def test_invalid_logging_format_is_rejected(tmp_path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
        [logging]
        format = "xml"
        """
    )

    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["toml_file"] = config_file

    with pytest.raises(ValidationError):
        TestSettings()
