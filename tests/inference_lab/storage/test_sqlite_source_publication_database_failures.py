"""Injected SQLite failures for atomic managed-source publication."""

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from structlog.contextvars import merge_contextvars
from structlog.testing import capture_logs

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
)
from inference_lab.storage import (
    ContentOrigin,
    CoordinateOrigin,
    CoordinateUnit,
    PageLayoutDescriptor,
    PageRotation,
    ParsedRepresentation,
    ParseFailureCategory,
    ParseFailureRecord,
    ParseFailureSeverity,
    ParseFailureStage,
    ParseFailureUnitKind,
    ParseOutcome,
    ParseStatus,
    RepositoryClosedError,
    SourceArtefact,
    StorageConcurrencyError,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage._source_blob_store import ManagedSourceBlobStore
from inference_lab.storage._sqlite_migrations import MIGRATIONS
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    PARSE_ID,
    PARSED_RECORD_ID,
    PARSER,
    SOURCE_BYTES,
    SOURCE_HASH,
    SOURCE_TEXT,
    parsed_representation,
)

_CORPUS = CorpusMetadata(
    CorpusId("c0000000-0000-4000-8000-000000000001"),
    "Database failure corpus",
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
_LAYOUT = PageLayoutDescriptor(
    page_number=1,
    coordinate_unit=CoordinateUnit.PDF_POINTS,
    coordinate_origin=CoordinateOrigin.TOP_LEFT,
    page_width=612.0,
    page_height=792.0,
    rotation=PageRotation.DEGREES_0,
    content_origin=ContentOrigin.NATIVE,
    producer=PARSER,
)
_FAILURE = ParseFailureRecord(
    category=ParseFailureCategory.MALFORMED_UNIT,
    severity=ParseFailureSeverity.RECOVERABLE,
    stage=ParseFailureStage.STRUCTURE,
    unit_kind=ParseFailureUnitKind.PARAGRAPH,
    unit_index=1,
)
_OUTCOME = ParseOutcome(
    parse_id=PARSE_ID,
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    status=ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
    parser=PARSER,
    normalisation_version="1.0.0",
    parsed_record_id=PARSED_RECORD_ID,
    failures=(_FAILURE,),
)
_PUBLICATION_TABLES = (
    "source_artefacts",
    "parsed_representations",
    "page_layouts",
    "source_spans",
    "structural_blocks",
    "parse_outcomes",
    "parse_failures",
)
_PRIVATE_DATABASE_PAYLOAD = "private-database-publication-failure"


def _representation() -> ParsedRepresentation:
    representation = parsed_representation()
    return replace(
        representation,
        page_count=1,
        source_locations=tuple(
            replace(location, page_number=1)
            for location in representation.source_locations
        ),
        page_layouts=(_LAYOUT,),
    )


def _open_catalogued_repository(
    database_path: Path,
    source_dir: Path,
) -> SQLiteMetadataRepository:
    repository = SQLiteMetadataRepository.open(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
        busy_timeout_ms=25,
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
    return repository


def _install_rejecting_trigger(database_path: Path, table: str) -> None:
    if table not in _PUBLICATION_TABLES:
        raise ValueError("table must be a publication table.")
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            f"""CREATE TRIGGER reject_{table}_publication
            BEFORE INSERT ON {table}
            BEGIN
                SELECT RAISE(ABORT, '{_PRIVATE_DATABASE_PAYLOAD}');
            END"""  # noqa: S608
        )
        connection.commit()
    finally:
        connection.close()


def _drop_rejecting_trigger(database_path: Path, table: str) -> None:
    if table not in _PUBLICATION_TABLES:
        raise ValueError("table must be a publication table.")
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(f"DROP TRIGGER reject_{table}_publication")  # noqa: S608
        connection.commit()
    finally:
        connection.close()


def _table_counts(database_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(database_path)
    try:
        return {
            table: connection.execute(
                f"SELECT count(*) FROM {table}"  # noqa: S608
            ).fetchone()[0]
            for table in _PUBLICATION_TABLES
        }
    finally:
        connection.close()


def _publish(repository: SQLiteMetadataRepository) -> None:
    repository.persist_source_artefact(
        _CORPUS.corpus_id,
        _ARTEFACT,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
        parse_outcome=_OUTCOME,
        parsed_representation=_representation(),
    )


def _assert_no_publication(repository: SQLiteMetadataRepository) -> None:
    assert (
        repository.get_source_artefact(
            _CORPUS.corpus_id,
            DOCUMENT_VERSION_ID,
        )
        is None
    )
    assert (
        repository.get_parsed_representation(
            _CORPUS.corpus_id,
            PARSED_RECORD_ID,
        )
        is None
    )
    assert repository.get_parse_outcome(_CORPUS.corpus_id, PARSE_ID) is None


@pytest.mark.parametrize("rejected_table", _PUBLICATION_TABLES)
def test_each_database_stage_failure_rolls_back_the_complete_publication(
    tmp_path: Path,
    rejected_table: str,
) -> None:
    database_path = tmp_path / "private-database" / "metadata.sqlite3"
    source_dir = tmp_path / "private-source"
    repository = _open_catalogued_repository(database_path, source_dir)
    _install_rejecting_trigger(database_path, rejected_table)

    try:
        with capture_logs(processors=(merge_contextvars,)) as logs:
            with pytest.raises(StorageConflictError) as raised:
                _publish(repository)

        _assert_no_publication(repository)
        assert repository.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert _table_counts(database_path) == dict.fromkeys(_PUBLICATION_TABLES, 0)
        assert (source_dir / _ARTEFACT.storage_key).read_bytes() == SOURCE_BYTES
        assert logs[-1]["event"] == "source_persistence_failed"
        assert logs[-1]["error_category"] == "conflict"
        for private_value in (
            _PRIVATE_DATABASE_PAYLOAD,
            str(database_path),
            str(source_dir),
            SOURCE_TEXT,
        ):
            assert private_value not in str(raised.value)
            assert private_value not in repr(logs)
    finally:
        repository.close()

    _drop_rejecting_trigger(database_path, rejected_table)
    reopened = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    try:
        _assert_no_publication(reopened)
        assert reopened.get_corpus(_CORPUS.corpus_id) == _CORPUS
    finally:
        reopened.close()


def test_transaction_start_failure_leaves_orphan_reusable_and_repository_usable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-database" / "metadata.sqlite3"
    source_dir = tmp_path / "private-source"
    repository = _open_catalogued_repository(database_path, source_dir)
    blocker = sqlite3.connect(database_path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with capture_logs(processors=(merge_contextvars,)) as logs:
            with pytest.raises(StorageConcurrencyError) as raised:
                _publish(repository)
    finally:
        blocker.rollback()
        blocker.close()

    try:
        _assert_no_publication(repository)
        assert repository.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert (source_dir / _ARTEFACT.storage_key).read_bytes() == SOURCE_BYTES
        assert _PRIVATE_DATABASE_PAYLOAD not in str(raised.value)
        assert "locked" not in str(raised.value).lower()
        assert logs[-1]["error_category"] == "concurrency"
        assert "locked" not in repr(logs).lower()
        assert str(database_path) not in repr(logs)
        assert str(source_dir) not in repr(logs)

        _publish(repository)
        assert (
            repository.read_source_bytes(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == SOURCE_BYTES
        )
    finally:
        repository.close()


class _CommitFailingConnection(sqlite3.Connection):
    def commit(self) -> None:
        raise sqlite3.OperationalError(_PRIVATE_DATABASE_PAYLOAD)


def test_commit_failure_rolls_back_publication_and_allows_reopen_and_retry(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-database" / "metadata.sqlite3"
    source_dir = tmp_path / "private-source"
    initial = _open_catalogued_repository(database_path, source_dir)
    initial.close()
    connection = sqlite3.connect(
        database_path,
        isolation_level=None,
        factory=_CommitFailingConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    repository = SQLiteMetadataRepository(
        connection,
        MIGRATIONS,
        source_blob_store=ManagedSourceBlobStore(source_dir),
    )

    try:
        with capture_logs(processors=(merge_contextvars,)) as logs:
            with pytest.raises(StorageTransactionError) as raised:
                _publish(repository)

        _assert_no_publication(repository)
        assert repository.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert _table_counts(database_path) == dict.fromkeys(_PUBLICATION_TABLES, 0)
        assert (source_dir / _ARTEFACT.storage_key).read_bytes() == SOURCE_BYTES
        assert _PRIVATE_DATABASE_PAYLOAD not in str(raised.value)
        assert _PRIVATE_DATABASE_PAYLOAD not in repr(logs)
        assert str(database_path) not in repr(logs)
        assert str(source_dir) not in repr(logs)
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    try:
        _assert_no_publication(reopened)
        assert reopened.get_corpus(_CORPUS.corpus_id) == _CORPUS
        _publish(reopened)
        assert (
            reopened.get_parsed_representation(
                _CORPUS.corpus_id,
                PARSED_RECORD_ID,
            )
            == _representation()
        )
        assert reopened.get_parse_outcome(_CORPUS.corpus_id, PARSE_ID) == _OUTCOME
    finally:
        reopened.close()


class _RollbackFailingConnection(sqlite3.Connection):
    def rollback(self) -> None:
        raise sqlite3.OperationalError(_PRIVATE_DATABASE_PAYLOAD)


class _CommitAndRollbackFailingConnection(_CommitFailingConnection):
    def rollback(self) -> None:
        raise sqlite3.OperationalError(_PRIVATE_DATABASE_PAYLOAD)


def test_commit_cleanup_failure_invalidates_connection_and_preserves_commit_error(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-database" / "metadata.sqlite3"
    source_dir = tmp_path / "private-source"
    initial = _open_catalogued_repository(database_path, source_dir)
    initial.close()
    connection = sqlite3.connect(
        database_path,
        isolation_level=None,
        factory=_CommitAndRollbackFailingConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    repository = SQLiteMetadataRepository(
        connection,
        MIGRATIONS,
        source_blob_store=ManagedSourceBlobStore(source_dir),
    )

    with capture_logs(processors=(merge_contextvars,)) as logs:
        with pytest.raises(StorageTransactionError) as raised:
            _publish(repository)

    assert repository.is_closed is True
    with pytest.raises(RepositoryClosedError):
        repository.get_corpus(_CORPUS.corpus_id)
    assert (source_dir / _ARTEFACT.storage_key).read_bytes() == SOURCE_BYTES
    assert _PRIVATE_DATABASE_PAYLOAD not in str(raised.value)
    assert _PRIVATE_DATABASE_PAYLOAD not in repr(logs)
    assert str(database_path) not in repr(logs)
    assert str(source_dir) not in repr(logs)

    reopened = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    try:
        _assert_no_publication(reopened)
        assert reopened.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert _table_counts(database_path) == dict.fromkeys(_PUBLICATION_TABLES, 0)
    finally:
        reopened.close()


def test_rollback_failure_preserves_primary_and_invalidates_partial_connection(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-database" / "metadata.sqlite3"
    source_dir = tmp_path / "private-source"
    initial = _open_catalogued_repository(database_path, source_dir)
    initial.close()
    _install_rejecting_trigger(database_path, "parse_failures")
    connection = sqlite3.connect(
        database_path,
        isolation_level=None,
        factory=_RollbackFailingConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    repository = SQLiteMetadataRepository(
        connection,
        MIGRATIONS,
        source_blob_store=ManagedSourceBlobStore(source_dir),
    )

    with capture_logs(processors=(merge_contextvars,)) as logs:
        with pytest.raises(StorageConflictError) as raised:
            _publish(repository)

    assert repository.is_closed is True
    with pytest.raises(RepositoryClosedError):
        repository.get_corpus(_CORPUS.corpus_id)
    assert (source_dir / _ARTEFACT.storage_key).read_bytes() == SOURCE_BYTES
    assert _PRIVATE_DATABASE_PAYLOAD not in str(raised.value)
    assert _PRIVATE_DATABASE_PAYLOAD not in repr(logs)
    assert str(database_path) not in repr(logs)
    assert str(source_dir) not in repr(logs)

    _drop_rejecting_trigger(database_path, "parse_failures")
    reopened = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    try:
        _assert_no_publication(reopened)
        assert reopened.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert _table_counts(database_path) == dict.fromkeys(_PUBLICATION_TABLES, 0)
    finally:
        reopened.close()
