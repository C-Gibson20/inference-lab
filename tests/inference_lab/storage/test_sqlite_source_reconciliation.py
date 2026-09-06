"""Tests for bounded, conservative managed-source reconciliation."""

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
)
from inference_lab.storage import SourceArtefact, open_metadata_repository
from inference_lab.storage import _sqlite_repository as repository_module
from inference_lab.storage._source_blob_store import ManagedSourceBlobStore
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    SOURCE_BYTES,
    SOURCE_HASH,
)

_CORPUS = CorpusMetadata(
    CorpusId("b0000000-0000-4000-8000-000000000001"),
    "Source reconciliation corpus",
)
_VERSION = DocumentVersion(DOCUMENT_VERSION_ID, DOCUMENT_ID, SOURCE_HASH)
_ARTEFACT = SourceArtefact(
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    content_hash=SOURCE_HASH,
    byte_count=len(SOURCE_BYTES),
    media_type="application/pdf",
    storage_key=f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}",
)


def _repository(tmp_path: Path) -> tuple[SQLiteMetadataRepository, Path]:
    source_dir = tmp_path / "source-data"
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        source_dir=source_dir,
        apply_migrations=True,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS)
        transaction.add_document(DocumentIdentity(DOCUMENT_ID))
        transaction.add_document_version(_VERSION)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS.corpus_id,
                DOCUMENT_ID,
                DOCUMENT_VERSION_ID,
            )
        )
    return repository, source_dir


def _staging_path(source_dir: Path, digit: str) -> Path:
    return source_dir / ".staging" / f".{digit * 32}.tmp"


def test_reconciliation_cleans_staging_and_reports_final_blob_inventory(
    tmp_path: Path,
) -> None:
    repository, source_dir = _repository(tmp_path)
    repository.persist_source_artefact(
        _CORPUS.corpus_id,
        _ARTEFACT,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )
    abandoned = _staging_path(source_dir, "1")
    abandoned.parent.mkdir(exist_ok=True)
    abandoned.write_bytes(b"partial")
    ambiguous_staging = abandoned.parent / "keep.me"
    ambiguous_staging.write_bytes(b"unknown")
    orphan_digest = "f" * 64
    orphan = source_dir / "sha256" / "ff" / orphan_digest
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    ambiguous_final = orphan.parent / "not-a-digest"
    ambiguous_final.write_bytes(b"unknown")

    try:
        with capture_logs() as logs:
            repository.reconcile_source_storage(max_entries=100)

        assert not abandoned.exists()
        assert ambiguous_staging.is_file()
        assert (source_dir / _ARTEFACT.storage_key).is_file()
        assert orphan.is_file()
        assert ambiguous_final.is_file()
        assert [record["event"] for record in logs] == [
            "source_reconciliation_started",
            "source_reconciliation_completed",
        ]
        assert logs[-1]["staging_files_removed"] == 1
        assert logs[-1]["referenced_blobs_found"] == 1
        assert logs[-1]["orphan_blobs_found"] == 1
        assert logs[-1]["ambiguous_entries_retained"] == 2
        assert logs[-1]["limit_reached"] is False
        assert str(source_dir) not in repr(logs)
        assert SOURCE_HASH.digest not in repr(logs)
    finally:
        repository.close()


def test_reconciliation_stops_at_the_entry_bound(
    tmp_path: Path,
) -> None:
    repository, source_dir = _repository(tmp_path)
    staging_files = tuple(_staging_path(source_dir, digit) for digit in "123")
    staging_files[0].parent.mkdir(parents=True)
    for path in staging_files:
        path.write_bytes(b"partial")

    try:
        with capture_logs() as logs:
            repository.reconcile_source_storage(max_entries=1)

        assert sum(path.is_file() for path in staging_files) == 2
        assert logs[-1]["entries_examined"] == 1
        assert logs[-1]["staging_files_removed"] == 1
        assert logs[-1]["limit_reached"] is True
    finally:
        repository.close()


def test_repository_startup_runs_bounded_source_reconciliation(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    initial = SQLiteMetadataRepository.open(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    initial.close()
    staging_files = tuple(_staging_path(source_dir, digit) for digit in "123")
    staging_files[0].parent.mkdir(parents=True, exist_ok=True)
    for path in staging_files:
        path.write_bytes(b"partial")

    with capture_logs() as logs:
        repository = open_metadata_repository(
            database_path,
            source_dir=source_dir,
            source_reconciliation_max_entries=1,
        )
    try:
        assert sum(path.is_file() for path in staging_files) == 2
        reconciliation = [
            record
            for record in logs
            if record["event"] == "source_reconciliation_completed"
        ]
        assert len(reconciliation) == 1
        assert reconciliation[0]["entries_examined"] == 1
        assert reconciliation[0]["staging_files_removed"] == 1
        assert reconciliation[0]["limit_reached"] is True
    finally:
        repository.close()


def test_repository_startup_closes_database_without_replacing_recovery_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    initial = SQLiteMetadataRepository.open(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    initial.close()
    primary = RuntimeError("private-reconciliation-payload")
    closed_connections: list[sqlite3.Connection] = []
    real_close = repository_module.close_preserving_primary

    def fail_reconciliation(
        self: ManagedSourceBlobStore,
        is_referenced: Callable[[str], bool],
        *,
        max_entries: int,
    ) -> object:
        raise primary

    def record_close(connection: sqlite3.Connection | None) -> bool:
        if connection is not None:
            closed_connections.append(connection)
        return real_close(connection)

    monkeypatch.setattr(ManagedSourceBlobStore, "reconcile_locked", fail_reconciliation)
    monkeypatch.setattr(repository_module, "close_preserving_primary", record_close)

    with capture_logs() as logs:
        with pytest.raises(RuntimeError) as raised:
            SQLiteMetadataRepository.open(database_path, source_dir=source_dir)

    assert raised.value is primary
    assert len(closed_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        closed_connections[0].execute("SELECT 1")
    assert [record["event"] for record in logs[-2:]] == [
        "source_reconciliation_started",
        "source_reconciliation_failed",
    ]
    assert "private-reconciliation-payload" not in repr(logs)


def test_reconciliation_does_not_report_an_exactly_exhausted_complete_scan_as_limited(
    tmp_path: Path,
) -> None:
    repository, source_dir = _repository(tmp_path)
    staging_file = _staging_path(source_dir, "1")
    staging_file.parent.mkdir(parents=True)
    staging_file.write_bytes(b"partial")

    try:
        with capture_logs() as logs:
            repository.reconcile_source_storage(max_entries=1)

        assert not staging_file.exists()
        assert logs[-1]["entries_examined"] == 1
        assert logs[-1]["limit_reached"] is False
    finally:
        repository.close()
