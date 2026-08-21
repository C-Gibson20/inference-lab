from functools import lru_cache
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


_URI_LOCATION = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _validate_local_path(value: object) -> object:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("path must not be empty")
        if _URI_LOCATION.match(value.strip()):
            raise ValueError("path must be a local filesystem path")
    return value


class ModelPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedder_path: Path
    reranker_path: Path
    generator_path: Path

    _validate_paths = field_validator(
        "embedder_path", "reranker_path", "generator_path", mode="before"
    )(_validate_local_path)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vram_budget_gb: int = Field(gt=0)
    max_context_tokens: int = Field(default=8192, gt=0)


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_dir: Path
    metadata_db_path: Path

    _validate_paths = field_validator("index_dir", "metadata_db_path", mode="before")(
        _validate_local_path
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["auto", "console", "json"] = "auto"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file="config.toml",
        env_nested_delimiter="__",
        extra="forbid",
    )

    models: ModelPaths
    runtime: RuntimeConfig
    storage: StorageConfig
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
