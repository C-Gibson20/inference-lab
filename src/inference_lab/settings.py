from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ModelPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedder_path: Path
    reranker_path: Path
    generator_path: Path


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vram_budget_gb: int
    max_context_tokens: int = 8192


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_dir: Path
    metadata_db_path: Path


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
