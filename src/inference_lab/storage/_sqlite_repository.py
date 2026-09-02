"""Internal SQLite metadata repository lifecycle owner."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Self

import structlog

from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    DocumentFamilyId,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateRelationship,
    ExternalIdentifier,
)
from inference_lab.storage._sqlite_connection import (
    DEFAULT_BUSY_TIMEOUT_MS,
    translate_sqlite_error,
)
from inference_lab.storage._sqlite_core import (
    add_corpus,
    add_document,
    add_document_family,
    get_corpus,
    get_document,
    get_document_family,
)
from inference_lab.storage._sqlite_inspection import (
    SchemaInspection,
    validate_database_schema,
)
from inference_lab.storage._sqlite_lifecycle import open_sqlite_database
from inference_lab.storage._sqlite_migration_execution import (
    AppliedAtFactory,
    utc_now,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage._sqlite_relationships import (
    add_duplicate_relationship,
    list_duplicate_relationships,
)
from inference_lab.storage._sqlite_versions import (
    add_document_version,
    find_document_versions_by_content_hash,
    find_document_versions_by_external_identifiers,
    get_document_version,
    list_document_versions,
)
from inference_lab.storage.errors import (
    RepositoryClosedError,
    RepositoryUnavailableError,
    StorageTransactionError,
)

_LOGGER = structlog.get_logger(__name__)


def _safe_log(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded repository lifecycle metadata without changing outcomes."""
    try:
        getattr(_LOGGER, method_name)(event, pipeline_stage="storage", **fields)
    except Exception:
        pass


class SQLiteMetadataRepository:
    """Own one validated SQLite connection and its explicit lifecycle."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        migrations: tuple[MigrationDefinition, ...],
    ) -> None:
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("connection must be a sqlite3.Connection.")
        if not isinstance(migrations, tuple):
            raise TypeError("migrations must be a tuple.")
        self._connection: sqlite3.Connection | None = connection
        self._migrations = migrations
        self._transaction_active = False

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        apply_migrations: bool = False,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
        applied_at_factory: AppliedAtFactory = utc_now,
    ) -> Self:
        """Open a validated repository under the explicit schema policy."""
        connection = open_sqlite_database(
            path,
            apply_migrations=apply_migrations,
            busy_timeout_ms=busy_timeout_ms,
            migrations=migrations,
            applied_at_factory=applied_at_factory,
        )
        return cls(connection, migrations)

    @property
    def is_closed(self) -> bool:
        """Return whether this repository has been permanently closed."""
        return self._connection is None

    @property
    def schema_inspection(self) -> SchemaInspection:
        """Return the validated current schema state."""
        return validate_database_schema(
            self._connection_or_raise(),
            migrations=self._migrations,
        )

    def get_corpus(self, corpus_id: CorpusId) -> CorpusId | None:
        """Return a persistent corpus identity, or ``None`` when absent."""
        return get_corpus(self._connection_or_raise(), corpus_id)

    def get_document(self, document_id: DocumentId) -> DocumentIdentity | None:
        """Return a logical document, or ``None`` when absent."""
        return get_document(self._connection_or_raise(), document_id)

    def get_document_family(
        self,
        family_id: DocumentFamilyId,
    ) -> DocumentFamilyId | None:
        """Return a document-family identity, or ``None`` when absent."""
        return get_document_family(self._connection_or_raise(), family_id)

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        """Return an immutable document version, or ``None`` when absent."""
        return get_document_version(
            self._connection_or_raise(),
            document_version_id,
        )

    def list_document_versions(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentVersion, ...]:
        """Return a document's versions in canonical identifier order."""
        return list_document_versions(self._connection_or_raise(), document_id)

    def find_document_versions_by_content_hash(
        self,
        content_hash: ContentHash,
    ) -> tuple[DocumentVersion, ...]:
        """Return exact-hash candidates in canonical request order."""
        return find_document_versions_by_content_hash(
            self._connection_or_raise(),
            content_hash,
        )

    def find_document_versions_by_external_identifiers(
        self,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        """Return unique versions matching any supplied identifier."""
        return find_document_versions_by_external_identifiers(
            self._connection_or_raise(),
            external_identifiers,
        )

    def list_duplicate_relationships(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[DuplicateRelationship, ...]:
        """Return a version's duplicate relationships in canonical order."""
        return list_duplicate_relationships(
            self._connection_or_raise(),
            document_version_id,
        )

    def transaction(self) -> SQLiteMetadataTransaction:
        """Return a new write transaction that starts on context entry."""
        self._connection_or_raise()
        return SQLiteMetadataTransaction(self)

    def _connection_or_raise(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            raise RepositoryClosedError()
        return connection

    def close(self) -> None:
        """Close the owned connection idempotently and reject future use."""
        connection = self._connection
        if connection is None:
            return
        if self._transaction_active:
            error = StorageTransactionError()
            _safe_log(
                "error",
                "repository_close_failed",
                error_category=error.category.value,
                retryable=error.retryable,
            )
            raise error
        self._connection = None

        try:
            connection.close()
        except sqlite3.Error as error:
            translated = translate_sqlite_error(
                error,
                default=RepositoryUnavailableError,
            )
            _safe_log(
                "error",
                "repository_close_failed",
                error_category=translated.category.value,
                retryable=translated.retryable,
            )
            raise translated from None

        _safe_log("info", "repository_close_completed")


class SQLiteMetadataTransaction:
    """One non-nestable SQLite metadata write transaction."""

    def __init__(self, repository: SQLiteMetadataRepository) -> None:
        self._repository = repository
        self._active = False
        self._finished = False

    def __enter__(self) -> Self:
        """Acquire the write transaction, rejecting nesting and reuse."""
        if self._active or self._finished or self._repository._transaction_active:
            raise StorageTransactionError()

        connection = self._repository._connection_or_raise()
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as sqlite_error:
            error = translate_sqlite_error(
                sqlite_error,
                default=StorageTransactionError,
            )
            _safe_log(
                "error",
                "transaction_failed",
                error_category=error.category.value,
                retryable=error.retryable,
            )
            raise error from None

        self._active = True
        self._repository._transaction_active = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Commit success or roll back while preserving a primary failure."""
        del exception_type, traceback
        connection = self._connection_or_raise()

        if exception is not None:
            rolled_back = False
            try:
                connection.rollback()
                rolled_back = True
            except sqlite3.Error:
                _safe_log(
                    "error",
                    "transaction_failed",
                    error_category=StorageTransactionError.category.value,
                    retryable=StorageTransactionError.retryable,
                )
            finally:
                self._finish()
            if rolled_back:
                _safe_log("info", "transaction_rolled_back")
            return False

        try:
            connection.commit()
        except sqlite3.Error as sqlite_error:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            error = translate_sqlite_error(
                sqlite_error,
                default=StorageTransactionError,
            )
            self._finish()
            _safe_log(
                "error",
                "transaction_failed",
                error_category=error.category.value,
                retryable=error.retryable,
            )
            raise error from None

        self._finish()
        _safe_log("info", "transaction_completed")
        return False

    def get_corpus(self, corpus_id: CorpusId) -> CorpusId | None:
        """Return a corpus identity within this transaction."""
        return get_corpus(self._connection_or_raise(), corpus_id)

    def add_corpus(self, corpus_id: CorpusId) -> None:
        """Create a corpus identity within this transaction."""
        add_corpus(self._connection_or_raise(), corpus_id)

    def get_document(self, document_id: DocumentId) -> DocumentIdentity | None:
        """Return a logical document within this transaction."""
        return get_document(self._connection_or_raise(), document_id)

    def add_document(self, document: DocumentIdentity) -> None:
        """Create a logical document within this transaction."""
        add_document(self._connection_or_raise(), document)

    def get_document_family(
        self,
        family_id: DocumentFamilyId,
    ) -> DocumentFamilyId | None:
        """Return a family identity within this transaction."""
        return get_document_family(self._connection_or_raise(), family_id)

    def add_document_family(self, family_id: DocumentFamilyId) -> None:
        """Create a document-family identity within this transaction."""
        add_document_family(self._connection_or_raise(), family_id)

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        """Return an immutable version within this transaction."""
        return get_document_version(
            self._connection_or_raise(),
            document_version_id,
        )

    def list_document_versions(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentVersion, ...]:
        """Return a document's versions within this transaction."""
        return list_document_versions(self._connection_or_raise(), document_id)

    def find_document_versions_by_content_hash(
        self,
        content_hash: ContentHash,
    ) -> tuple[DocumentVersion, ...]:
        """Return exact-hash candidates within this transaction."""
        return find_document_versions_by_content_hash(
            self._connection_or_raise(),
            content_hash,
        )

    def find_document_versions_by_external_identifiers(
        self,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        """Return identifier candidates within this transaction."""
        return find_document_versions_by_external_identifiers(
            self._connection_or_raise(),
            external_identifiers,
        )

    def add_document_version(self, document_version: DocumentVersion) -> None:
        """Create a version and its external identifiers atomically."""
        add_document_version(self._connection_or_raise(), document_version)

    def list_duplicate_relationships(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[DuplicateRelationship, ...]:
        """Return duplicate relationships within this transaction."""
        return list_duplicate_relationships(
            self._connection_or_raise(),
            document_version_id,
        )

    def add_duplicate_relationship(
        self,
        relationship: DuplicateRelationship,
    ) -> None:
        """Create a duplicate relationship within this transaction."""
        add_duplicate_relationship(self._connection_or_raise(), relationship)

    def _connection_or_raise(self) -> sqlite3.Connection:
        if not self._active or self._finished:
            raise StorageTransactionError()
        return self._repository._connection_or_raise()

    def _finish(self) -> None:
        self._active = False
        self._finished = True
        self._repository._transaction_active = False
