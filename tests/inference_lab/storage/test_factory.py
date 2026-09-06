"""Tests for metadata repository construction."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.provenance import CorpusId, CorpusMetadata
from inference_lab.settings import StorageConfig
from inference_lab.storage import (
    MetadataRepository,
    MigrationRequiredError,
    open_metadata_repository,
)

_CORPUS = CorpusMetadata(
    CorpusId("b779bb09-f656-41c7-bc19-df4b6a04bcfd"),
    "Factory acceptance corpus",
)


def test_factory_opens_configured_repository_across_restart(tmp_path: Path) -> None:
    config = StorageConfig(
        index_dir=tmp_path / "indexes",
        metadata_db_path=tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "source-data",
    )

    first = open_metadata_repository(
        config.metadata_db_path,
        source_dir=config.source_dir,
        apply_migrations=True,
    )
    try:
        assert isinstance(first, MetadataRepository)
        with first.transaction() as transaction:
            transaction.add_corpus(_CORPUS)
    finally:
        first.close()

    second = open_metadata_repository(
        config.metadata_db_path,
        source_dir=config.source_dir,
    )
    try:
        assert isinstance(second, MetadataRepository)
        assert second.get_corpus(_CORPUS.corpus_id) == _CORPUS
    finally:
        second.close()


def test_factory_requires_explicit_schema_update_permission(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    config = StorageConfig(
        index_dir=tmp_path / "indexes",
        metadata_db_path=database_path,
        source_dir=tmp_path / "source-data",
    )

    with pytest.raises(MigrationRequiredError):
        open_metadata_repository(
            config.metadata_db_path,
            source_dir=config.source_dir,
        )

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_factory_rejects_invalid_source_directory_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="source_dir"):
        open_metadata_repository(
            tmp_path / "metadata.sqlite3",
            source_dir=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("max_entries", "error_type"),
    [
        (True, TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_factory_rejects_invalid_reconciliation_bound_before_opening_storage(
    tmp_path: Path,
    max_entries: object,
    error_type: type[Exception],
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"

    with pytest.raises(error_type, match="source_reconciliation_max_entries"):
        open_metadata_repository(
            database_path,
            source_dir=source_dir,
            source_reconciliation_max_entries=max_entries,  # type: ignore[arg-type]
        )

    assert not database_path.exists()
    assert not source_dir.exists()
