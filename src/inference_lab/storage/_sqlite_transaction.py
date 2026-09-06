"""SQLite metadata write-transaction implementation."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Concatenate, ParamSpec, Self, TypeVar

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
from inference_lab.storage._sqlite_cleanup import rollback_preserving_primary
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_core import (
    add_corpus,
    add_document,
    add_document_family,
    get_corpus,
    get_document,
    get_document_family,
    list_corpora,
    set_corpus_display_name,
)
from inference_lab.storage._sqlite_corpus_memberships import (
    add_corpus_document_version_membership,
    get_corpus_document_version_membership,
    list_corpus_document_version_memberships,
)
from inference_lab.storage._sqlite_identifier_assertions import (
    add_external_identifier_assertions,
    list_external_identifier_assertions,
)
from inference_lab.storage._sqlite_logging import safe_log, safe_log_failure
from inference_lab.storage._sqlite_observations import (
    add_source_observation,
    advance_source_observation,
    get_source_observation,
    list_source_observation_history,
)
from inference_lab.storage._sqlite_parsed_representations import (
    add_parsed_representation,
)
from inference_lab.storage._sqlite_relationships import (
    add_duplicate_relationship,
    add_family_membership,
    list_duplicate_relationships,
    list_family_memberships_by_document,
    list_family_memberships_by_document_version,
    list_family_memberships_by_family,
)
from inference_lab.storage._sqlite_source_artefacts import (
    add_source_artefact,
    get_source_artefact,
    read_source_bytes,
    stream_source_bytes,
)
from inference_lab.storage._sqlite_versions import (
    add_document_version,
    find_document_versions_by_content_hash,
    find_document_versions_by_external_identifiers,
    get_document_version,
    list_document_versions,
)
from inference_lab.storage.errors import (
    StorageError,
    StorageTransactionError,
)
from inference_lab.storage.parsed_records import ParsedRepresentation
from inference_lab.storage.source_records import SourceArtefact

if TYPE_CHECKING:
    from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_OperationParameters = ParamSpec("_OperationParameters")
_OperationResult = TypeVar("_OperationResult")


class SQLiteMetadataTransaction:
    """One non-nestable SQLite metadata write transaction."""

    def __init__(self, repository: SQLiteMetadataRepository) -> None:
        self._repository = repository
        self._active = False
        self._finished = False
        self._failed = False

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
            safe_log_failure("transaction_failed", error)
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

        if exception is not None or self._failed:
            rolled_back = rollback_preserving_primary(connection)
            if not rolled_back and exception is not None:
                safe_log_failure("transaction_failed", StorageTransactionError())
            self._finish()
            if exception is not None:
                if self._failed and rolled_back:
                    error = (
                        exception
                        if isinstance(exception, StorageError)
                        else StorageTransactionError()
                    )
                    safe_log_failure("transaction_failed", error)
                elif rolled_back:
                    safe_log("info", "transaction_rolled_back")
                return False

            error = StorageTransactionError()
            safe_log_failure("transaction_failed", error)
            raise error from None

        try:
            connection.commit()
        except sqlite3.Error as sqlite_error:
            rollback_preserving_primary(connection)
            error = translate_sqlite_error(
                sqlite_error,
                default=StorageTransactionError,
            )
            self._finish()
            safe_log_failure("transaction_failed", error)
            raise error from None

        self._finish()
        safe_log("info", "transaction_completed")
        return False

    def get_corpus(self, corpus_id: CorpusId) -> CorpusMetadata | None:
        """Return a named corpus within this transaction."""
        return self._execute(get_corpus, corpus_id)

    def list_corpora(self) -> tuple[CorpusMetadata, ...]:
        """Return all named corpora within this transaction."""
        return self._execute(list_corpora)

    def add_corpus(self, corpus: CorpusMetadata) -> None:
        """Create a named corpus within this transaction."""
        self._execute(add_corpus, corpus)

    def set_corpus_display_name(
        self,
        corpus_id: CorpusId,
        display_name: str,
    ) -> CorpusMetadata:
        """Replace a corpus display name within this transaction."""
        return self._execute(set_corpus_display_name, corpus_id, display_name)

    def get_corpus_document_version_membership(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> CorpusDocumentVersionMembership | None:
        """Return one corpus catalogue membership within this transaction."""
        return self._execute(
            get_corpus_document_version_membership,
            corpus_id,
            document_version_id,
        )

    def list_corpus_document_version_memberships(
        self,
        corpus_id: CorpusId,
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        """Return a corpus catalogue within this transaction."""
        return self._execute(list_corpus_document_version_memberships, corpus_id)

    def add_corpus_document_version_membership(
        self,
        membership: CorpusDocumentVersionMembership,
    ) -> None:
        """Add a corpus catalogue membership within this transaction."""
        self._execute(add_corpus_document_version_membership, membership)

    def get_document(self, document_id: DocumentId) -> DocumentIdentity | None:
        """Return a logical document within this transaction."""
        return self._execute(get_document, document_id)

    def add_document(self, document: DocumentIdentity) -> None:
        """Create a logical document within this transaction."""
        self._execute(add_document, document)

    def get_document_family(
        self,
        family_id: DocumentFamilyId,
    ) -> DocumentFamilyId | None:
        """Return a family identity within this transaction."""
        return self._execute(get_document_family, family_id)

    def add_document_family(self, family_id: DocumentFamilyId) -> None:
        """Create a document-family identity within this transaction."""
        self._execute(add_document_family, family_id)

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        """Return an immutable version within this transaction."""
        return self._execute(get_document_version, document_version_id)

    def list_document_versions(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentVersion, ...]:
        """Return a document's versions within this transaction."""
        return self._execute(list_document_versions, document_id)

    def get_source_artefact(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        """Return corpus-visible managed-source metadata in this transaction."""
        return self._execute(
            get_source_artefact,
            corpus_id,
            document_version_id,
        )

    def read_source_bytes(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> bytes | None:
        """Return verified corpus-visible bytes in this transaction."""
        return self._execute(
            read_source_bytes,
            self._repository._source_blob_store,
            corpus_id,
            document_version_id,
        )

    def stream_source_bytes(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> Iterator[bytes] | None:
        """Return fully verified, detached chunks in this transaction."""
        return self._execute(
            stream_source_bytes,
            self._repository._source_blob_store,
            corpus_id,
            document_version_id,
        )

    def find_document_versions_by_content_hash(
        self,
        corpus_id: CorpusId,
        content_hash: ContentHash,
    ) -> tuple[DocumentVersion, ...]:
        """Return exact-hash candidates within this transaction."""
        return self._execute(
            find_document_versions_by_content_hash,
            corpus_id,
            content_hash,
        )

    def find_document_versions_by_external_identifiers(
        self,
        corpus_id: CorpusId,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        """Return identifier candidates within this transaction."""
        return self._execute(
            find_document_versions_by_external_identifiers,
            corpus_id,
            external_identifiers,
        )

    def get_source_observation(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> SourceObservationState | None:
        """Return the current observed-source state within this transaction."""
        return self._execute(
            get_source_observation,
            corpus_id,
            source_observation_id,
        )

    def list_source_observation_history(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> tuple[SourceObservationState, ...]:
        """Return observed-source history within this transaction."""
        return self._execute(
            list_source_observation_history,
            corpus_id,
            source_observation_id,
        )

    def list_external_identifier_assertions(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        """Return identifier assertions within this transaction."""
        return self._execute(
            list_external_identifier_assertions,
            corpus_id,
            document_version_id,
        )

    def add_document_version(self, document_version: DocumentVersion) -> None:
        """Create a version and its external identifiers atomically."""
        self._execute(add_document_version, document_version)

    def add_source_artefact(
        self,
        corpus_id: CorpusId,
        artefact: SourceArtefact,
    ) -> None:
        """Add corpus-authorised managed-source metadata."""
        self._execute(add_source_artefact, corpus_id, artefact)

    def add_parsed_representation(
        self,
        corpus_id: CorpusId,
        representation: ParsedRepresentation,
    ) -> None:
        """Add one complete corpus-authorised parsed aggregate."""
        self._execute(add_parsed_representation, corpus_id, representation)

    def add_source_observation(self, state: SourceObservationState) -> None:
        """Create an observed source at revision one."""
        self._execute(add_source_observation, state)

    def advance_source_observation(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
        *,
        expected_version_id: DocumentVersionId,
        new_version_id: DocumentVersionId,
    ) -> SourceObservationState:
        """Atomically advance an observed source from its expected version."""
        return self._execute(
            advance_source_observation,
            corpus_id,
            source_observation_id,
            expected_version_id=expected_version_id,
            new_version_id=new_version_id,
        )

    def add_external_identifier_assertions(
        self,
        corpus_id: CorpusId,
        assertions: tuple[ExternalIdentifierAssertion, ...],
    ) -> None:
        """Append a canonical identifier-evidence batch."""
        self._execute(add_external_identifier_assertions, corpus_id, assertions)

    def list_duplicate_relationships(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[DuplicateRelationship, ...]:
        """Return duplicate relationships within this transaction."""
        return self._execute(
            list_duplicate_relationships,
            corpus_id,
            document_version_id,
        )

    def add_duplicate_relationship(
        self,
        relationship: DuplicateRelationship,
    ) -> None:
        """Create a duplicate relationship within this transaction."""
        self._execute(add_duplicate_relationship, relationship)

    def list_family_memberships_by_family(
        self,
        corpus_id: CorpusId,
        family_id: DocumentFamilyId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a family's memberships within this transaction."""
        return self._execute(
            list_family_memberships_by_family,
            corpus_id,
            family_id,
        )

    def list_family_memberships_by_document(
        self,
        corpus_id: CorpusId,
        document_id: DocumentId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a document's memberships within this transaction."""
        return self._execute(
            list_family_memberships_by_document,
            corpus_id,
            document_id,
        )

    def list_family_memberships_by_document_version(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return a version's memberships within this transaction."""
        return self._execute(
            list_family_memberships_by_document_version,
            corpus_id,
            document_version_id,
        )

    def add_family_membership(
        self,
        membership: DocumentFamilyMembership,
    ) -> None:
        """Create a family membership within this transaction."""
        self._execute(add_family_membership, membership)

    def _execute(
        self,
        operation: Callable[
            Concatenate[sqlite3.Connection, _OperationParameters],
            _OperationResult,
        ],
        *args: _OperationParameters.args,
        **kwargs: _OperationParameters.kwargs,
    ) -> _OperationResult:
        connection = self._connection_or_raise()
        if self._failed:
            raise StorageTransactionError()
        try:
            return operation(connection, *args, **kwargs)
        except BaseException:
            self._failed = True
            raise

    def _connection_or_raise(self) -> sqlite3.Connection:
        if not self._active or self._finished:
            raise StorageTransactionError()
        return self._repository._connection_or_raise()

    def _finish(self) -> None:
        self._active = False
        self._finished = True
        self._repository._transaction_active = False
