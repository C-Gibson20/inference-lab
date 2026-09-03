import json
import sqlite3
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pytest

from inference_lab.logging.logging_config import configure_logging
from inference_lab.provenance import (
    ContentHash,
    CorpusId,
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
    MetadataTransaction,
    RepositoryUnavailableError,
    StorageConflictError,
    StorageMigrationError,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 2, 15, 0, tzinfo=UTC)
_CORPUS_ID = CorpusId("ea304c92-db98-4112-9976-38d2d67a28d4")
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
    "corpus_version",
}


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _records(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def _assert_storage_log_shape(records: list[dict[str, object]]) -> None:
    assert records
    for record in records:
        assert _REQUIRED_LOG_FIELDS <= record.keys()
        assert record["pipeline_stage"] == "storage"
        assert record["document_id"] is None
        assert record["corpus_version"] is None
        assert isinstance(record["module"], str)
        assert record["module"].startswith("inference_lab.storage.")
        assert isinstance(record["timestamp"], str)
        assert datetime.fromisoformat(record["timestamp"]).utcoffset() == UTC.utcoffset(
            _APPLIED_AT
        )


def _add_complete_graph(transaction: MetadataTransaction) -> None:
    transaction.add_corpus(_CORPUS_ID)
    transaction.add_document_family(_FAMILY_ID)
    transaction.add_document(DocumentIdentity(_FIRST_DOCUMENT_ID))
    transaction.add_document(DocumentIdentity(_SECOND_DOCUMENT_ID))
    transaction.add_document_version(_FIRST_VERSION)
    transaction.add_document_version(_SECOND_VERSION)
    transaction.add_duplicate_relationship(_DUPLICATE)
    transaction.add_family_membership(_MEMBERSHIP)


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
        _add_complete_graph(transaction)
    with repository.transaction() as transaction:
        _add_complete_graph(transaction)
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
        assert reopened.list_duplicate_relationships(_FIRST_VERSION_ID) == (_DUPLICATE,)
        assert reopened.list_family_memberships_by_family(_FAMILY_ID) == (_MEMBERSHIP,)
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
