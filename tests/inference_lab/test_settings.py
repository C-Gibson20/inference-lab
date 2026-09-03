from collections.abc import Iterator
from pathlib import Path

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

        [ingest]
        max_source_bytes = 1048576
        """
    )
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _load_settings(settings_type: type[Settings]) -> Settings:
    """Load a Settings subclass from its configured external sources."""
    return settings_type()  # pyright: ignore[reportCallIssue]


def test_explicit_runtime_settings_override_toml() -> None:
    configured = Settings(  # pyright: ignore[reportCallIssue]
        runtime={"vram_budget_gb": 12}
    )

    assert configured.runtime.vram_budget_gb == 12
    assert configured.runtime.max_context_tokens == 8192


def test_explicit_nested_model_setting_overrides_toml() -> None:
    configured = Settings(  # pyright: ignore[reportCallIssue]
        models={"generator_path": "/tmp/test-generator"}
    )

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
        _load_settings(TestSettings)


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
        _load_settings(TestSettings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vram_budget_gb", 0),
        ("vram_budget_gb", -1),
        ("max_context_tokens", 0),
        ("max_context_tokens", -1),
    ],
)
def test_non_positive_runtime_setting_is_rejected(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(  # pyright: ignore[reportCallIssue]
            runtime={field: value}
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("models", "embedder_path"),
        ("models", "reranker_path"),
        ("models", "generator_path"),
        ("storage", "index_dir"),
        ("storage", "metadata_db_path"),
    ],
)
def test_empty_path_is_rejected(section: str, field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(**{section: {field: ""}})  # pyright: ignore


def test_non_local_model_path_shapes_are_rejected(
    rejected_local_path_shape: str | Path,
) -> None:
    with pytest.raises(ValidationError):
        Settings(  # pyright: ignore[reportCallIssue]
            models={"generator_path": rejected_local_path_shape}
        )


def test_unknown_nested_setting_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(  # pyright: ignore[reportCallIssue]
            storage={
                "index_dir": "/data/indexes",
                "metadata_db_path": "/data/metadata.db",
                "backup_dir": "/data/backups",
            }
        )


def test_unknown_nested_toml_setting_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
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
        backup_dir = "/data/backups"
        """
    )

    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["toml_file"] = config_file

    with pytest.raises(ValidationError):
        _load_settings(TestSettings)


def test_logging_settings_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
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
        """
    )

    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["toml_file"] = config_file

    configured = _load_settings(TestSettings)

    assert configured.logging.level == "INFO"
    assert configured.logging.format == "auto"


def test_logging_settings_load_from_toml() -> None:
    configured = get_settings()

    assert configured.logging.level == "DEBUG"
    assert configured.logging.format == "json"


def test_explicit_logging_settings_override_toml() -> None:
    configured = Settings(  # pyright: ignore[reportCallIssue]
        logging={"level": "WARNING"}
    )

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
        _load_settings(TestSettings)


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
        _load_settings(TestSettings)


def test_ingest_size_setting_loads_from_toml() -> None:
    configured = get_settings()

    assert configured.ingest.max_source_bytes == 1048576


def test_ingest_size_setting_has_safe_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
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
        """
    )

    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["toml_file"] = config_file

    configured = _load_settings(TestSettings)

    assert configured.ingest.max_source_bytes == 50 * 1024 * 1024


def test_ingest_size_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST__MAX_SOURCE_BYTES", "2048")

    configured = get_settings()

    assert configured.ingest.max_source_bytes == 2048


@pytest.mark.parametrize("value", [0, -1, "not-an-int"])
def test_invalid_ingest_size_is_rejected(value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(  # pyright: ignore[reportCallIssue]
            ingest={"max_source_bytes": value}
        )
