"""Tests for fail-closed SQLite corpus-version reads."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.storage import (
    CorruptStorageError,
)
from tests.inference_lab.storage.corpus_publication_fixtures import (
    _CORPUS_A,
    _FIRST_MEMBERSHIP,
    _VERSION_A1,
    _add_catalogue,
    _commit,
    _open,
    _prepare_update,
)


def test_version_reads_fail_closed_when_committed_lifecycle_is_corrupt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    _prepare_update(
        repository,
        _VERSION_A1,
        expected_corpus_version=None,
        memberships=(_FIRST_MEMBERSHIP,),
    )
    _commit(repository, _VERSION_A1)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP TRIGGER corpus_updates_lifecycle_update")
        connection.execute(
            """UPDATE corpus_updates SET state = 'ready'
            WHERE corpus_id = ? AND corpus_version = ?""",
            (_CORPUS_A.corpus_id.as_str(), _VERSION_A1.as_str()),
        )
        connection.commit()
    finally:
        connection.close()

    try:
        with pytest.raises(CorruptStorageError):
            repository.get_active_corpus_version(_CORPUS_A.corpus_id)
        with pytest.raises(CorruptStorageError):
            repository.get_corpus_version(_CORPUS_A.corpus_id, _VERSION_A1)
        with pytest.raises(CorruptStorageError):
            repository.list_corpus_versions(_CORPUS_A.corpus_id)
    finally:
        repository.close()


def test_active_read_fails_closed_when_corpus_pointer_row_is_missing(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP TRIGGER corpus_active_versions_delete_immutable")
        connection.execute(
            "DELETE FROM corpus_active_versions WHERE corpus_id = ?",
            (_CORPUS_A.corpus_id.as_str(),),
        )
        connection.commit()
    finally:
        connection.close()

    try:
        with pytest.raises(CorruptStorageError):
            repository.get_active_corpus_version(_CORPUS_A.corpus_id)
    finally:
        repository.close()
