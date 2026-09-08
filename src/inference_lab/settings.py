"""Typed application settings and configuration-source precedence."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from inference_lab._validation import is_local_path_shape


def _validate_local_path(value: object) -> object:
    if isinstance(value, (str, Path)) and not is_local_path_shape(value):
        raise ValueError("path must be a local filesystem path")
    return value


def _absolute_lexical_path(path: Path) -> Path:
    """Return a host-normalised absolute path without resolving filesystem aliases."""
    return Path(os.path.normcase(os.path.abspath(os.fspath(path))))


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether two lexical paths are equal or contain one another."""
    return first == second or first in second.parents or second in first.parents


class ModelPaths(BaseModel):
    """Filesystem locations for the local embedding, reranking, and generator models."""

    model_config = ConfigDict(extra="forbid")

    embedder_path: Path
    reranker_path: Path
    generator_path: Path

    _validate_paths = field_validator(
        "embedder_path", "reranker_path", "generator_path", mode="before"
    )(_validate_local_path)


class RuntimeConfig(BaseModel):
    """Hardware and context limits used by the runtime."""

    model_config = ConfigDict(extra="forbid")

    vram_budget_gb: int = Field(gt=0)
    max_context_tokens: int = Field(default=8192, gt=0)


class StorageConfig(BaseModel):
    """Filesystem locations for persistent indexes, metadata, and source files."""

    model_config = ConfigDict(extra="forbid")

    index_dir: Path
    metadata_db_path: Path
    source_dir: Path

    _validate_paths = field_validator(
        "index_dir",
        "metadata_db_path",
        "source_dir",
        mode="before",
    )(_validate_local_path)

    @model_validator(mode="after")
    def _validate_path_separation(self) -> Self:
        """Reject equal or nested deployment storage locations."""
        paths = (
            ("index_dir", _absolute_lexical_path(self.index_dir)),
            ("metadata_db_path", _absolute_lexical_path(self.metadata_db_path)),
            ("source_dir", _absolute_lexical_path(self.source_dir)),
        )
        for index, (first_name, first_path) in enumerate(paths):
            for second_name, second_path in paths[index + 1 :]:
                if _paths_overlap(first_path, second_path):
                    raise ValueError(f"{first_name} and {second_name} must not overlap")
        return self


class LoggingConfig(BaseModel):
    """Log level and renderer settings for application output."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["auto", "console", "json"] = "auto"


class IngestConfig(BaseModel):
    """Limits and controls for batch document ingestion."""

    model_config = ConfigDict(extra="forbid")

    max_source_bytes: int = Field(default=50 * 1024 * 1024, gt=0)


class Settings(BaseSettings):
    """Validated application settings loaded from explicit, environment, or TOML values."""

    model_config = SettingsConfigDict(
        toml_file="config.toml",
        env_nested_delimiter="__",
        extra="forbid",
    )

    models: ModelPaths
    runtime: RuntimeConfig
    storage: StorageConfig
    logging: LoggingConfig = LoggingConfig()
    ingest: IngestConfig = IngestConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Return configuration sources in descending precedence order."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the validated settings for the current process."""
    return Settings()  # pyright: ignore[reportCallIssue]
