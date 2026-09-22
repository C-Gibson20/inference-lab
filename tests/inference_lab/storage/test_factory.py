"""Tests for metadata repository construction."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_lab.provenance import CorpusId, CorpusMetadata
from inference_lab.settings import StorageConfig
from inference_lab.storage import (
    MetadataRepository,
    MigrationRequiredError,
    RepositoryClosedError,
    RepositoryUnavailableError,
    WorkerRepositoryOpener,
    open_metadata_repository,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

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


def test_worker_repository_owns_connection_without_reconciliation(
    tmp_path: Path,
) -> None:
    """Worker opens on its own thread without repeating startup cleanup."""
    database = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    parent = open_metadata_repository(
        database,
        source_dir=source_dir,
        apply_migrations=True,
    )
    assert isinstance(parent, WorkerRepositoryOpener)
    child_corpus = CorpusMetadata(CorpusId.generate(), "Worker-owned corpus")
    try:

        def work() -> bool:
            child = parent.open_worker_repository()
            try:
                with child.transaction() as transaction:
                    transaction.add_corpus(child_corpus)
                return child.get_corpus(child_corpus.corpus_id) == child_corpus
            finally:
                child.close()

        with patch.object(
            SQLiteMetadataRepository,
            "reconcile_source_storage",
            side_effect=AssertionError("worker must not run startup reconciliation"),
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(work).result()
        assert parent.get_corpus(child_corpus.corpus_id) == child_corpus
    finally:
        parent.close()
    with pytest.raises(RepositoryClosedError):
        parent.open_worker_repository()


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
    ("database_suffix", "source_suffix"),
    [
        (("state",), ("state",)),
        (("state", "metadata.sqlite3"), ("state",)),
        (("state",), ("state", "sources")),
    ],
)
def test_factory_rejects_resolved_storage_path_overlap_before_creation(
    tmp_path: Path,
    database_suffix: tuple[str, ...],
    source_suffix: tuple[str, ...],
) -> None:
    database_path = tmp_path.joinpath(*database_suffix)
    source_dir = tmp_path.joinpath(*source_suffix)

    with pytest.raises(ValueError, match="must not overlap"):
        open_metadata_repository(
            database_path,
            source_dir=source_dir,
            apply_migrations=True,
        )

    assert not database_path.exists()
    assert not source_dir.exists()


def test_factory_rejects_symlink_alias_overlap_before_opening_storage(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source-data"
    source_dir.mkdir()
    database_alias = tmp_path / "metadata-alias.sqlite3"
    database_alias.symlink_to(source_dir / "metadata.sqlite3")

    with pytest.raises(ValueError, match="must not overlap"):
        open_metadata_repository(
            database_alias,
            source_dir=source_dir,
            apply_migrations=True,
        )

    assert database_alias.is_symlink()
    assert not (source_dir / "metadata.sqlite3").exists()


def test_factory_hides_path_resolution_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolution(
        _path: Path,
        *,
        strict: bool = False,
    ) -> Path:
        del strict
        raise OSError("private-storage-path")

    monkeypatch.setattr(Path, "resolve", fail_resolution)

    with pytest.raises(RepositoryUnavailableError) as caught:
        open_metadata_repository(
            tmp_path / "metadata.sqlite3",
            source_dir=tmp_path / "source-data",
        )

    assert "private-storage-path" not in str(caught.value)


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
