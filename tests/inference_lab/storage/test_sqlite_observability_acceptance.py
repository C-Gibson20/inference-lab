"""Acceptance tests for SQLite observability."""

import json
import sqlite3
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pytest

import inference_lab.storage._sqlite_corpus_snapshots as sqlite_corpus_snapshots
from inference_lab.logging.logging_config import configure_logging
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentVersion,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    ExternalIdentifier,
    HashAlgorithm,
)
from inference_lab.settings import LoggingConfig
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorruptStorageError,
    IncompatibleSchemaError,
    MigrationRequiredError,
    RepositoryClosedError,
    RepositoryUnavailableError,
    RollbackCorpusVersionOutcome,
    SealCorpusUpdateOutcome,
    StorageConcurrencyError,
    StorageConflictError,
    StorageError,
    StorageMigrationError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_logging import safe_log_failure
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
_CORPUS_VERSION_MEMBERSHIP_ONE = CorpusDocumentVersionMembership(
    _CORPUS_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
)
_CORPUS_VERSION_MEMBERSHIP_TWO = CorpusDocumentVersionMembership(
    _CORPUS_ID,
    _SECOND_DOCUMENT_ID,
    _SECOND_VERSION_ID,
)
_CORPUS_VERSION_ONE = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_CORPUS_VERSION_TWO = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_ROLLBACK_VERSION = CorpusVersionId("55555555-5555-4555-8555-555555555555")
_ABORTED_VERSION = CorpusVersionId("66666666-6666-4666-8666-666666666666")
_FAILED_VERSION = CorpusVersionId("77777777-7777-4777-8777-777777777777")
_CONFLICT_VERSION = CorpusVersionId("88888888-8888-4888-8888-888888888888")
_UNKNOWN_VERSION = CorpusVersionId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_LEXICAL = ComponentKind("lexical")
_PRIVATE_COMPONENT_CANARY = "private-component-payload"
_PRIVATE_COMPONENT_IMPLEMENTATION = "private-index"
_PRIVATE_CONFIGURATION_DIGEST = "d" * 64
_PRIVATE_SOURCE_MANIFEST_DIGEST = "e" * 64
_PRIVATE_VALIDATOR_CANARY = "private-component-validator-failure"
_PRIVATE_ERROR_CANARY = "/private/error/payload-canary"
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
_CORPUS_VERSION_EVENTS = {
    "corpus_snapshot_close_failed",
    "corpus_snapshot_closed",
    "corpus_snapshot_open_failed",
    "corpus_snapshot_opened",
    "corpus_update_aborted",
    "corpus_update_ready",
    "corpus_update_started",
    "corpus_version_committed",
    "corpus_version_conflict",
    "corpus_version_rollback_committed",
}


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


def _assert_corpus_version_log_shape(records: list[dict[str, object]]) -> None:
    assert records
    for record in records:
        assert _REQUIRED_LOG_FIELDS <= record.keys()
        assert record["pipeline_stage"] == "storage"
        assert record["document_id"] is None
        assert isinstance(record["module"], str)
        assert record["module"].startswith("inference_lab.storage.")
        assert isinstance(record["timestamp"], str)
        assert datetime.fromisoformat(record["timestamp"]).utcoffset() == UTC.utcoffset(
            APPLIED_AT
        )


class _AcceptingComponentValidator:
    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        return True


class _FailingComponentValidator:
    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        raise RuntimeError(_PRIVATE_VALIDATOR_CANARY)


def _component_generation(
    corpus_version: CorpusVersionId,
    suffix: str,
) -> ComponentGenerationReference:
    return ComponentGenerationReference(
        component_kind=_LEXICAL,
        generation_id=f"{_PRIVATE_COMPONENT_CANARY}-{suffix}",
        implementation=_PRIVATE_COMPONENT_IMPLEMENTATION,
        format_version="1.0.0",
        contract_version="1.0.0",
        configuration_digest=_PRIVATE_CONFIGURATION_DIGEST,
        source_manifest_digest=_PRIVATE_SOURCE_MANIFEST_DIGEST,
        corpus_id=_CORPUS_ID,
        corpus_version=corpus_version,
        state=ComponentGenerationState.PREPARING,
    )


def _seal_corpus_version(
    repository: SQLiteMetadataRepository,
    corpus_version: CorpusVersionId,
    *,
    expected_corpus_version: CorpusVersionId | None,
    memberships: tuple[CorpusDocumentVersionMembership, ...],
    generation: ComponentGenerationReference,
    restores_corpus_version: CorpusVersionId | None = None,
) -> SealCorpusUpdateOutcome:
    sealed: SealCorpusUpdateOutcome | None = None
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS_ID,
            corpus_version,
            expected_corpus_version=expected_corpus_version,
            required_component_kinds=(_LEXICAL,),
            restores_corpus_version=restores_corpus_version,
        )
        for membership in memberships:
            transaction.add_corpus_update_document_membership(
                _CORPUS_ID,
                corpus_version,
                membership,
            )
        transaction.register_corpus_update_component_generation(
            _CORPUS_ID,
            corpus_version,
            generation,
            validator=_AcceptingComponentValidator(),
        )
        result = transaction.seal_corpus_update(_CORPUS_ID, corpus_version)
        assert isinstance(result, SealCorpusUpdateOutcome)
        sealed = result
    assert sealed is not None
    return sealed


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


def test_corpus_version_events_are_complete_and_payload_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path_marker = "private-corpus-version-path"
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
    stream.seek(0)
    stream.truncate(0)

    try:
        _seal_corpus_version(
            repository,
            _CORPUS_VERSION_ONE,
            expected_corpus_version=None,
            memberships=(_CORPUS_VERSION_MEMBERSHIP_ONE,),
            generation=_component_generation(_CORPUS_VERSION_ONE, "v1"),
        )
        with repository.transaction() as transaction:
            transaction.commit_corpus_update(_CORPUS_ID, _CORPUS_VERSION_ONE)

        with repository.open_active_corpus_snapshot(_CORPUS_ID):
            pass

        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_ID,
                _ABORTED_VERSION,
                expected_corpus_version=_CORPUS_VERSION_ONE,
            )
            transaction.abort_corpus_update(_CORPUS_ID, _ABORTED_VERSION)

        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_ID,
                _FAILED_VERSION,
                expected_corpus_version=_CORPUS_VERSION_ONE,
                required_component_kinds=(_LEXICAL,),
            )
        with pytest.raises(RepositoryUnavailableError) as validator_failure:
            with repository.transaction() as transaction:
                transaction.register_corpus_update_component_generation(
                    _CORPUS_ID,
                    _FAILED_VERSION,
                    _component_generation(_FAILED_VERSION, "failed"),
                    validator=_FailingComponentValidator(),
                )
        assert _PRIVATE_VALIDATOR_CANARY not in str(validator_failure.value)

        _seal_corpus_version(
            repository,
            _CORPUS_VERSION_TWO,
            expected_corpus_version=_CORPUS_VERSION_ONE,
            memberships=(_CORPUS_VERSION_MEMBERSHIP_TWO,),
            generation=_component_generation(_CORPUS_VERSION_TWO, "v2"),
        )
        with repository.transaction() as transaction:
            transaction.commit_corpus_update(_CORPUS_ID, _CORPUS_VERSION_TWO)

        _seal_corpus_version(
            repository,
            _ROLLBACK_VERSION,
            expected_corpus_version=_CORPUS_VERSION_TWO,
            memberships=(),
            generation=_component_generation(_ROLLBACK_VERSION, "rollback"),
            restores_corpus_version=_CORPUS_VERSION_ONE,
        )
        rollback: RollbackCorpusVersionOutcome | None = None
        with repository.transaction() as transaction:
            rollback = transaction.commit_corpus_rollback(
                _CORPUS_ID,
                _ROLLBACK_VERSION,
            )
        assert rollback is not None

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.begin_corpus_update(
                    _CORPUS_ID,
                    _CONFLICT_VERSION,
                    expected_corpus_version=None,
                )

        unavailable = repository.open_corpus_snapshot(
            _CORPUS_ID,
            _UNKNOWN_VERSION,
        )
        with pytest.raises(StorageConflictError):
            with unavailable:
                pytest.fail("an unavailable corpus snapshot must not open")

        close_failure = repository.open_active_corpus_snapshot(_CORPUS_ID)
        close_failure.__enter__()
        monkeypatch.setattr(
            sqlite_corpus_snapshots,
            "rollback_preserving_primary",
            lambda connection: False,
        )
        with pytest.raises(StorageTransactionError):
            close_failure.close()
    finally:
        repository.close()

    records = _records(stream)
    _assert_corpus_version_log_shape(records)
    events = {str(record["event"]) for record in records}
    assert _CORPUS_VERSION_EVENTS <= events
    assert {
        str(record["error_category"])
        for record in records
        if record["event"] == "transaction_failed"
    } == {"conflict", "repository_unavailable"}
    assert {
        str(record["error_category"])
        for record in records
        if record["event"] == "corpus_snapshot_open_failed"
    } == {"conflict"}
    assert {
        str(record["error_category"])
        for record in records
        if record["event"] == "corpus_snapshot_close_failed"
    } == {"transaction"}

    rendered = stream.getvalue()
    canaries = (
        str(database_path),
        private_path_marker,
        _CORPUS.display_name,
        _PRIVATE_DIGEST,
        _PRIVATE_EXTERNAL_IDENTIFIER,
        _PRIVATE_COMPONENT_CANARY,
        _PRIVATE_COMPONENT_IMPLEMENTATION,
        _PRIVATE_CONFIGURATION_DIGEST,
        _PRIVATE_SOURCE_MANIFEST_DIGEST,
        _PRIVATE_VALIDATOR_CANARY,
        _FAMILY_ID.as_str(),
        _FIRST_DOCUMENT_ID.as_str(),
        _SECOND_DOCUMENT_ID.as_str(),
        _FIRST_VERSION_ID.as_str(),
        _SECOND_VERSION_ID.as_str(),
    )
    assert all(canary not in rendered for canary in canaries)


@pytest.mark.parametrize(
    "error_type",
    [
        RepositoryUnavailableError,
        RepositoryClosedError,
        StorageConflictError,
        StorageConcurrencyError,
        StorageTransactionError,
        CorruptStorageError,
        StorageMigrationError,
        IncompatibleSchemaError,
        MigrationRequiredError,
    ],
)
def test_every_storage_error_rejects_and_never_logs_payload(
    error_type: type[StorageError],
) -> None:
    with pytest.raises(TypeError) as rejected_payload:
        error_type(_PRIVATE_ERROR_CANARY)  # type: ignore[call-arg]
    assert _PRIVATE_ERROR_CANARY not in str(rejected_payload.value)

    stream = StringIO()
    configure_logging(LoggingConfig(level="INFO", format="json"), stream=stream)
    error = error_type()
    safe_log_failure("storage_error_observed", error)

    records = _records(stream)
    _assert_storage_log_shape(records)
    assert len(records) == 1
    assert records[0]["event"] == "storage_error_observed"
    assert records[0]["error_category"] == error.category.value
    assert records[0]["retryable"] is error.retryable
    assert _PRIVATE_ERROR_CANARY not in str(error)
    assert _PRIVATE_ERROR_CANARY not in stream.getvalue()
