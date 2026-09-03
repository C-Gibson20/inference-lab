"""Internal SQLite metadata repository lifecycle owner."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Self

from inference_lab._clock import utc_now
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
    DuplicateRelationship,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage._sqlite_connection import (
    DEFAULT_BUSY_TIMEOUT_MS,
    translate_sqlite_error,
)
from inference_lab.storage._sqlite_core import (
    get_corpus,
    get_document,
    get_document_family,
    list_corpora,
)
from inference_lab.storage._sqlite_corpus_memberships import (
    get_corpus_document_version_membership,
    list_corpus_document_version_memberships,
)
from inference_lab.storage._sqlite_identifier_assertions import (
    list_external_identifier_assertions,
)
from inference_lab.storage._sqlite_inspection import (
    SchemaInspection,
    validate_database_schema,
)
from inference_lab.storage._sqlite_lifecycle import open_sqlite_database
from inference_lab.storage._sqlite_logging import safe_log, safe_log_failure
from inference_lab.storage._sqlite_migration_execution import AppliedAtFactory
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage._sqlite_observations import (
    get_source_observation,
    list_source_observation_history,
)
from inference_lab.storage._sqlite_relationships import (
    list_duplicate_relationships,
    list_family_memberships_by_document,
    list_family_memberships_by_document_version,
    list_family_memberships_by_family,
)
from inference_lab.storage._sqlite_transaction import SQLiteMetadataTransaction
from inference_lab.storage._sqlite_versions import (
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

    def get_corpus(self, corpus_id: CorpusId) -> CorpusMetadata | None:
        """Return a named corpus, or ``None`` when absent."""
        return get_corpus(self._connection_or_raise(), corpus_id)

    def list_corpora(self) -> tuple[CorpusMetadata, ...]:
        """Return all named corpora in canonical identifier order."""
        return list_corpora(self._connection_or_raise())

    def get_corpus_document_version_membership(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> CorpusDocumentVersionMembership | None:
        """Return one corpus catalogue membership, or ``None``."""
        return get_corpus_document_version_membership(
            self._connection_or_raise(),
            corpus_id,
            document_version_id,
        )

    def list_corpus_document_version_memberships(
        self,
        corpus_id: CorpusId,
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        """Return a corpus catalogue in canonical order."""
        return list_corpus_document_version_memberships(
            self._connection_or_raise(),
            corpus_id,
        )

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
        corpus_id: CorpusId,
        content_hash: ContentHash,
    ) -> tuple[DocumentVersion, ...]:
        """Return exact-hash candidates in canonical request order."""
        return find_document_versions_by_content_hash(
            self._connection_or_raise(),
            corpus_id,
            content_hash,
        )

    def find_document_versions_by_external_identifiers(
        self,
        corpus_id: CorpusId,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        """Return unique versions matching any supplied identifier."""
        return find_document_versions_by_external_identifiers(
            self._connection_or_raise(),
            corpus_id,
            external_identifiers,
        )

    def get_source_observation(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> SourceObservationState | None:
        """Return the current state for an observed source, or ``None``."""
        return get_source_observation(
            self._connection_or_raise(),
            corpus_id,
            source_observation_id,
        )

    def list_source_observation_history(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> tuple[SourceObservationState, ...]:
        """Return an observed source's append-only state history."""
        return list_source_observation_history(
            self._connection_or_raise(),
            corpus_id,
            source_observation_id,
        )

    def list_external_identifier_assertions(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        """Return append-only assertions for one source version."""
        return list_external_identifier_assertions(
            self._connection_or_raise(),
            corpus_id,
            document_version_id,
        )

    def list_duplicate_relationships(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[DuplicateRelationship, ...]:
        """Return a version's duplicate relationships in canonical order."""
        return list_duplicate_relationships(
            self._connection_or_raise(),
            corpus_id,
            document_version_id,
        )

    def list_family_memberships_by_family(
        self,
        corpus_id: CorpusId,
        family_id: DocumentFamilyId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a family's memberships in canonical key order."""
        return list_family_memberships_by_family(
            self._connection_or_raise(),
            corpus_id,
            family_id,
        )

    def list_family_memberships_by_document(
        self,
        corpus_id: CorpusId,
        document_id: DocumentId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a document's memberships in canonical key order."""
        return list_family_memberships_by_document(
            self._connection_or_raise(),
            corpus_id,
            document_id,
        )

    def list_family_memberships_by_document_version(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a version's memberships in canonical key order."""
        return list_family_memberships_by_document_version(
            self._connection_or_raise(),
            corpus_id,
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
            safe_log_failure("repository_close_failed", error)
            raise error
        self._connection = None

        try:
            connection.close()
        except sqlite3.Error as error:
            translated = translate_sqlite_error(
                error,
                default=RepositoryUnavailableError,
            )
            safe_log_failure("repository_close_failed", translated)
            raise translated from None

        safe_log("info", "repository_close_completed")
