from datetime import UTC, datetime
from pathlib import Path

import pytest

from inference_lab.provenance import DocumentFamilyId
from inference_lab.storage import RepositoryClosedError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 2, 9, 30, tzinfo=UTC)
_FAMILY_ID = DocumentFamilyId("4a020bbc-5e41-46f8-8442-37a4aef86552")


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _open_repository(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )


def test_document_family_round_trips_after_restart_without_row_identity(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    first = _open_repository(database_path)
    with first.transaction() as transaction:
        assert transaction.get_document_family(_FAMILY_ID) is None
        transaction.add_document_family(_FAMILY_ID)
        transaction.add_document_family(_FAMILY_ID)
        assert transaction.get_document_family(_FAMILY_ID) == _FAMILY_ID
    first.close()

    second = SQLiteMetadataRepository.open(database_path)
    try:
        stored = second.get_document_family(_FAMILY_ID)

        assert stored == _FAMILY_ID
        assert isinstance(stored, DocumentFamilyId)
        assert not hasattr(stored, "row_id")
        assert not hasattr(stored, "storage_id")
    finally:
        second.close()


def test_missing_document_family_returns_none(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        assert repository.get_document_family(_FAMILY_ID) is None
    finally:
        repository.close()


def test_document_family_operations_reject_closed_repository(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    repository.close()

    with pytest.raises(RepositoryClosedError):
        repository.get_document_family(_FAMILY_ID)
