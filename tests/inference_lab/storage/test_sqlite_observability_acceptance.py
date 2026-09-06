"""Acceptance tests for SQLite observability."""

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pytest

from inference_lab.logging.logging_config import configure_logging
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    ExternalIdentifier,
    HashAlgorithm,
)
from inference_lab.settings import LoggingConfig
from inference_lab.storage import (
    CorruptStorageError,
    MetadataRepository,
    ParseFailureCategory,
    ParseFailureRecord,
    ParseFailureSeverity,
    ParseFailureStage,
    ParseFailureUnitKind,
    ParseOutcome,
    ParseStatus,
    RepositoryUnavailableError,
    SourceArtefact,
    StorageConflictError,
    StorageMigrationError,
    open_metadata_repository,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import (  # noqa: TID252
    APPLIED_AT,
    CompleteMetadataGraph,
    add_complete_graph,
)
from .fixtures import (  # noqa: TID252
    fixed_applied_at as _fixed_applied_at,
)
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

_CORPUS_ID = CorpusId("ea304c92-db98-4112-9976-38d2d67a28d4")
_CORPUS = CorpusMetadata(_CORPUS_ID, "Private test corpus")
_FAMILY_ID = DocumentFamilyId("815236bc-44b7-4784-84cc-f56bf72bb6d2")
_FIRST_DOCUMENT_ID = DocumentId("0d64c130-cea0-4c1d-8a44-298941893072")
_SECOND_DOCUMENT_ID = DocumentId("354c3b10-9a1a-47f2-91c9-76640ea573c2")
_FIRST_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_SECOND_VERSION_ID = DocumentVersionId("22222222-2222-4222-8222-222222222222")
_PRIVATE_DIGEST = "c" * 64
_PRIVATE_EXTERNAL_IDENTIFIER = "10.1000/private-observability"
_FIRST_VERSION = DocumentVersion(
    _FIRST_VERSION_ID,
    _FIRST_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, _PRIVATE_DIGEST),
    (ExternalIdentifier("doi", _PRIVATE_EXTERNAL_IDENTIFIER),),
)
_SECOND_VERSION = DocumentVersion(
    _SECOND_VERSION_ID,
    _SECOND_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, _PRIVATE_DIGEST),
)
_DUPLICATE = DuplicateRelationship(
    _FIRST_VERSION_ID,
    _SECOND_VERSION_ID,
    DuplicateClass.BYTE_IDENTICAL,
)
_MEMBERSHIP = DocumentFamilyMembership(
    _FAMILY_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
)
_REQUIRED_LOG_FIELDS = {
    "timestamp",
    "level",
    "event",
    "module",
    "pipeline_stage",
    "document_id",
    "corpus_id",
    "corpus_version",
}
_SOURCE_CORPUS = CorpusMetadata(
    CorpusId("ca000000-0000-4000-8000-000000000001"),
    "Private source observability corpus",
)
_SOURCE_EXTERNAL_IDENTIFIER = "10.1000/private-source-publication"
_SOURCE_VERSION = DocumentVersion(
    DOCUMENT_VERSION_ID,
    DOCUMENT_ID,
    SOURCE_HASH,
    (ExternalIdentifier("doi", _SOURCE_EXTERNAL_IDENTIFIER),),
)
_SOURCE_ARTEFACT = SourceArtefact(
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    content_hash=SOURCE_HASH,
    byte_count=len(SOURCE_BYTES),
    media_type="application/pdf",
    storage_key=f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}",
)
_PRIVATE_FAILURE_UNIT_INDEX = 987_654_321
_SOURCE_FAILURE = ParseFailureRecord(
    category=ParseFailureCategory.MALFORMED_UNIT,
    severity=ParseFailureSeverity.RECOVERABLE,
    stage=ParseFailureStage.STRUCTURE,
    unit_kind=ParseFailureUnitKind.PAGE,
    unit_index=_PRIVATE_FAILURE_UNIT_INDEX,
)
_SOURCE_OUTCOME = ParseOutcome(
    parse_id=PARSE_ID,
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    status=ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
    parser=PARSER,
    normalisation_version="1.0.0",
    parsed_record_id=PARSED_RECORD_ID,
    failures=(_SOURCE_FAILURE,),
)


def _records(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def _assert_storage_log_shape(records: list[dict[str, object]]) -> None:
    assert records
    for record in records:
        assert _REQUIRED_LOG_FIELDS <= record.keys()
        assert record["pipeline_stage"] == "storage"
        assert record["document_id"] is None
        assert record["corpus_id"] is None
        assert record["corpus_version"] is None
        assert isinstance(record["module"], str)
        assert record["module"].startswith("inference_lab.storage.")
        assert isinstance(record["timestamp"], str)
        assert datetime.fromisoformat(record["timestamp"]).utcoffset() == UTC.utcoffset(
            APPLIED_AT
        )


_COMPLETE_GRAPH = CompleteMetadataGraph(
    corpus=_CORPUS,
    family_id=_FAMILY_ID,
    versions=(_FIRST_VERSION, _SECOND_VERSION),
    duplicate_relationships=(_DUPLICATE,),
    family_memberships=(_MEMBERSHIP,),
)


def test_configured_logs_are_complete_and_payload_free_across_restart(
    tmp_path: Path,
) -> None:
    private_path_marker = "private-observability-path"
    private_exception = "private-transaction-exception"
    database_path = tmp_path / private_path_marker / "metadata.sqlite3"
    stream = StringIO()
    configure_logging(LoggingConfig(level="INFO", format="json"), stream=stream)

    repository = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    with repository.transaction() as transaction:
        add_complete_graph(transaction, _COMPLETE_GRAPH)
    with repository.transaction() as transaction:
        add_complete_graph(transaction, _COMPLETE_GRAPH)
    with pytest.raises(RuntimeError, match=private_exception):
        with repository.transaction():
            raise RuntimeError(private_exception)
    with pytest.raises(StorageConflictError):
        with repository.transaction() as transaction:
            transaction.add_document_version(
                DocumentVersion(
                    DocumentVersionId("99999999-9999-4999-8999-999999999999"),
                    DocumentId("99999999-9999-4999-8999-999999999999"),
                    ContentHash(HashAlgorithm.SHA256, "e" * 64),
                )
            )
    repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert reopened.get_document_version(_FIRST_VERSION_ID) == _FIRST_VERSION
        assert reopened.list_duplicate_relationships(_CORPUS_ID, _FIRST_VERSION_ID) == (
            _DUPLICATE,
        )
        assert reopened.list_family_memberships_by_family(_CORPUS_ID, _FAMILY_ID) == (
            _MEMBERSHIP,
        )
    finally:
        reopened.close()

    records = _records(stream)
    _assert_storage_log_shape(records)
    assert [(record["event"], record["level"]) for record in records] == [
        ("repository_open_started", "info"),
        ("repository_initialisation_started", "info"),
        ("repository_initialisation_completed", "info"),
        ("repository_open_completed", "info"),
        ("transaction_completed", "info"),
        ("transaction_completed", "info"),
        ("transaction_rolled_back", "info"),
        ("transaction_failed", "error"),
        ("repository_close_completed", "info"),
        ("repository_open_started", "info"),
        ("repository_open_completed", "info"),
        ("repository_close_completed", "info"),
    ]

    rendered = stream.getvalue()
    canaries = (
        str(database_path),
        private_path_marker,
        private_exception,
        _PRIVATE_DIGEST,
        _PRIVATE_EXTERNAL_IDENTIFIER,
        "INSERT INTO document_versions",
        _CORPUS_ID.as_str(),
        _FAMILY_ID.as_str(),
        _FIRST_DOCUMENT_ID.as_str(),
        _FIRST_VERSION_ID.as_str(),
    )
    assert all(canary not in rendered for canary in canaries)


def test_source_publication_logs_and_errors_exclude_private_payloads(
    tmp_path: Path,
) -> None:
    private_database_filename = "private-metadata-filename.sqlite3"
    private_source_path = "private-source-path"
    private_exception = "private-source-iterator-exception"
    database_path = tmp_path / private_database_filename
    source_dir = tmp_path / private_source_path
    stream = StringIO()
    configure_logging(LoggingConfig(level="INFO", format="json"), stream=stream)

    repository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    assert isinstance(repository, MetadataRepository)
    corrupt_error: CorruptStorageError | None = None
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_SOURCE_CORPUS)
            transaction.add_document(DocumentIdentity(DOCUMENT_ID))
            transaction.add_document_version(_SOURCE_VERSION)
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _SOURCE_CORPUS.corpus_id,
                    DOCUMENT_ID,
                    DOCUMENT_VERSION_ID,
                )
            )

        def interrupted_chunks() -> Iterator[bytes]:
            yield SOURCE_BYTES[:3]
            raise RuntimeError(private_exception)

        with pytest.raises(RuntimeError, match=private_exception):
            repository.persist_source_artefact(
                _SOURCE_CORPUS.corpus_id,
                _SOURCE_ARTEFACT,
                interrupted_chunks(),
                max_source_bytes=len(SOURCE_BYTES),
                parse_outcome=_SOURCE_OUTCOME,
                parsed_representation=parsed_representation(),
            )

        for _ in range(2):
            repository.persist_source_artefact(
                _SOURCE_CORPUS.corpus_id,
                _SOURCE_ARTEFACT,
                (SOURCE_BYTES,),
                max_source_bytes=len(SOURCE_BYTES),
                parse_outcome=_SOURCE_OUTCOME,
                parsed_representation=parsed_representation(),
            )

        blob_path = source_dir / _SOURCE_ARTEFACT.storage_key
        blob_path.chmod(0o600)
        blob_path.write_bytes(b"x" * len(SOURCE_BYTES))
        with pytest.raises(CorruptStorageError) as raised:
            repository.read_source_bytes(
                _SOURCE_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
        corrupt_error = raised.value
    finally:
        repository.close()

    records = _records(stream)
    assert records
    for record in records:
        assert _REQUIRED_LOG_FIELDS <= record.keys()
        assert record["pipeline_stage"] == "storage"
        assert record["corpus_version"] is None
        assert isinstance(record["module"], str)
        assert str(record["module"]).startswith("inference_lab.storage.")
        assert isinstance(record["timestamp"], str)
        assert datetime.fromisoformat(str(record["timestamp"])).utcoffset() == (
            UTC.utcoffset(APPLIED_AT)
        )

    source_records = [
        record
        for record in records
        if str(record["event"]).startswith("source_persistence_")
    ]
    assert [(record["event"], record["level"]) for record in source_records] == [
        ("source_persistence_started", "info"),
        ("source_persistence_failed", "error"),
        ("source_persistence_started", "info"),
        ("source_persistence_completed", "info"),
        ("source_persistence_started", "info"),
        ("source_persistence_completed", "info"),
    ]
    assert all(
        record["corpus_id"] == _SOURCE_CORPUS.corpus_id.as_str()
        and record["document_id"] == DOCUMENT_ID.as_str()
        for record in source_records
    )
    assert source_records[1]["error_category"] == "transaction"
    assert source_records[1]["retryable"] is False
    assert [
        record["byte_count"]
        for record in source_records
        if record["event"] == "source_persistence_completed"
    ] == [len(SOURCE_BYTES), len(SOURCE_BYTES)]

    assert corrupt_error is not None
    rendered_error = str(corrupt_error)
    rendered_logs = stream.getvalue()
    canaries = (
        str(database_path),
        private_database_filename,
        str(source_dir),
        private_source_path,
        private_exception,
        SOURCE_BYTES.decode(),
        SOURCE_TEXT,
        parsed_representation().text,
        SOURCE_HASH.digest,
        _SOURCE_ARTEFACT.storage_key,
        _SOURCE_EXTERNAL_IDENTIFIER,
        _SOURCE_CORPUS.display_name,
        str(_PRIVATE_FAILURE_UNIT_INDEX),
    )
    assert all(canary not in rendered_logs for canary in canaries)
    assert all(canary not in rendered_error for canary in canaries)


def test_configured_migration_failure_logs_exclude_sql_and_raw_errors(
    tmp_path: Path,
) -> None:
    private_path_marker = "private-migration-path"
    private_sql_marker = "private_sql_canary"
    database_path = tmp_path / private_path_marker / "metadata.sqlite3"
    stream = StringIO()
    configure_logging(LoggingConfig(level="INFO", format="json"), stream=stream)
    initial = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        migrations=(MIGRATIONS[0],),
        applied_at_factory=_fixed_applied_at,
    )
    initial.close()
    stream.seek(0)
    stream.truncate(0)

    failing = MigrationDefinition(
        revision=2,
        migration_id="test_observability_failure",
        statements=(
            f"CREATE TABLE {private_sql_marker} (value TEXT) STRICT",
            "CREATE TABLE documents (conflicting_value TEXT)",
        ),
    )
    with pytest.raises(StorageMigrationError) as raised:
        SQLiteMetadataRepository.open(
            database_path,
            apply_migrations=True,
            migrations=(MIGRATIONS[0], failing),
            applied_at_factory=_fixed_applied_at,
        )

    records = _records(stream)
    _assert_storage_log_shape(records)
    assert [(record["event"], record["level"]) for record in records] == [
        ("repository_open_started", "info"),
        ("migration_started", "info"),
        ("migration_failed", "error"),
        ("repository_open_failed", "error"),
    ]
    assert records[-1]["error_category"] == "migration"
    assert records[-1]["retryable"] is False
    rendered = stream.getvalue()
    assert private_path_marker not in rendered
    assert private_sql_marker not in rendered
    assert "CREATE TABLE" not in rendered
    assert "already exists" not in rendered
    assert "already exists" not in str(raised.value)


def test_configured_repository_close_failure_is_bounded(
    tmp_path: Path,
) -> None:
    private_error_marker = "private-close-error"

    class CloseFailingConnection(sqlite3.Connection):
        def close(self) -> None:
            raise sqlite3.OperationalError(private_error_marker)

    database_path = tmp_path / "metadata.sqlite3"
    initial = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    initial.close()
    connection = sqlite3.connect(
        database_path,
        isolation_level=None,
        factory=CloseFailingConnection,
    )
    repository = SQLiteMetadataRepository(connection, MIGRATIONS)
    stream = StringIO()
    configure_logging(LoggingConfig(level="INFO", format="json"), stream=stream)
    try:
        with pytest.raises(RepositoryUnavailableError) as raised:
            repository.close()

        records = _records(stream)
        _assert_storage_log_shape(records)
        assert [(record["event"], record["level"]) for record in records] == [
            ("repository_close_failed", "error"),
        ]
        assert records[0]["error_category"] == "repository_unavailable"
        assert records[0]["retryable"] is False
        assert private_error_marker not in stream.getvalue()
        assert private_error_marker not in str(raised.value)
    finally:
        sqlite3.Connection.close(connection)
