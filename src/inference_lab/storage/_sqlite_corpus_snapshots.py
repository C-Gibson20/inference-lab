"""Dedicated SQLite snapshot lifecycle for version-pinned corpus readers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from inference_lab.provenance import (
    ConfigurationIdentity,
    ContentHash,
    CorpusId,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateRelationship,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    RecordId,
    SourceObservationId,
    SourceObservationState,
    SourceSpanId,
)
from inference_lab.storage._source_blob_store import ManagedSourceBlobStore
from inference_lab.storage._sqlite_chunk_sets import (
    get_chunk_set_selection as read_chunk_set_selection,
)
from inference_lab.storage._sqlite_cleanup import (
    close_preserving_primary,
    rollback_preserving_primary,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_core import get_document as read_document
from inference_lab.storage._sqlite_core import (
    get_document_family as read_document_family,
)
from inference_lab.storage._sqlite_corpus_publication import _published_manifest
from inference_lab.storage._sqlite_corpus_update_reads import get_corpus_update
from inference_lab.storage._sqlite_corpus_versions import (
    get_active_corpus_version,
    get_corpus_version,
)
from inference_lab.storage._sqlite_identifier_assertions import (
    list_external_identifier_assertions as read_external_identifier_assertions,
)
from inference_lab.storage._sqlite_inspection import open_current_database
from inference_lab.storage._sqlite_logging import safe_log, safe_log_failure
from inference_lab.storage._sqlite_migrations import MigrationDefinition
from inference_lab.storage._sqlite_observations import (
    list_source_observation_history as read_source_observation_history,
)
from inference_lab.storage._sqlite_parse_outcomes import get_parse_outcome
from inference_lab.storage._sqlite_parsed_representations import (
    get_parsed_representation,
)
from inference_lab.storage._sqlite_relationships import (
    list_duplicate_relationships as read_duplicate_relationships,
)
from inference_lab.storage._sqlite_relationships import (
    list_family_memberships_by_document as read_family_memberships_by_document,
)
from inference_lab.storage._sqlite_relationships import (
    list_family_memberships_by_document_version as read_family_memberships_by_version,
)
from inference_lab.storage._sqlite_relationships import (
    list_family_memberships_by_family as read_family_memberships_by_family,
)
from inference_lab.storage._sqlite_source_artefacts import (
    get_source_artefact,
    read_source_bytes,
    stream_source_bytes,
)
from inference_lab.storage._sqlite_versions import (
    find_document_versions_by_content_hash as read_versions_by_content_hash,
)
from inference_lab.storage._sqlite_versions import (
    find_document_versions_by_external_identifiers as read_versions_by_identifiers,
)
from inference_lab.storage._sqlite_versions import (
    get_document_version as read_document_version,
)
from inference_lab.storage._sqlite_versions import (
    list_document_versions as read_document_versions,
)
from inference_lab.storage.chunk_records import ChunkSetSelectionRecord
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryClosedError,
    RepositoryUnavailableError,
    StorageConflictError,
    StorageError,
    StorageTransactionError,
)
from inference_lab.storage.parse_records import ParseOutcome
from inference_lab.storage.parsed_records import ParsedRepresentation
from inference_lab.storage.source_locations import (
    PageLayoutDescriptor,
    ResolvedSourceSpan,
    SourceLocation,
)
from inference_lab.storage.source_records import SourceArtefact
from inference_lab.storage.versioning import (
    CorpusUpdateState,
    CorpusVersion,
    CorpusVersionManifest,
)


@dataclass(frozen=True, slots=True)
class _SourceSpanIndex:
    """Compact source-location lookup built from one validated representation."""

    document_id: DocumentId
    locations: dict[SourceSpanId, SourceLocation]
    layouts: dict[int, PageLayoutDescriptor]

    @classmethod
    def from_representation(
        cls, representation: ParsedRepresentation
    ) -> _SourceSpanIndex:
        locations = {
            location.source_span_id: location
            for location in representation.source_locations
        }
        layouts = {layout.page_number: layout for layout in representation.page_layouts}
        if len(locations) != len(representation.source_locations):
            raise CorruptStorageError()
        if len(layouts) != len(representation.page_layouts):
            raise CorruptStorageError()
        return cls(representation.document_id, locations, layouts)


def _database_details(connection: sqlite3.Connection) -> tuple[Path, int]:
    try:
        database_rows = connection.execute("PRAGMA database_list").fetchall()
        main_rows = tuple(row for row in database_rows if row["name"] == "main")
        timeout_row = connection.execute("PRAGMA busy_timeout").fetchone()
        if len(main_rows) != 1 or timeout_row is None:
            raise RepositoryUnavailableError()
        database_value = main_rows[0]["file"]
        timeout_value = timeout_row[0]
        if not isinstance(database_value, str) or not database_value:
            raise RepositoryUnavailableError()
        if (
            not isinstance(timeout_value, int)
            or isinstance(timeout_value, bool)
            or timeout_value <= 0
        ):
            raise RepositoryUnavailableError()
        return Path(database_value), timeout_value
    except RepositoryUnavailableError:
        raise
    except (KeyError, IndexError, TypeError):
        raise RepositoryUnavailableError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


class SQLiteCorpusSnapshot:
    """Own one dedicated read transaction pinned to a committed version."""

    def __init__(
        self,
        database_path: Path,
        busy_timeout_ms: int,
        migrations: tuple[MigrationDefinition, ...],
        source_blob_store: ManagedSourceBlobStore | None,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId | None,
    ) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a Path.")
        if not isinstance(busy_timeout_ms, int) or isinstance(busy_timeout_ms, bool):
            raise TypeError("busy_timeout_ms must be an integer.")
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be greater than zero.")
        if not isinstance(migrations, tuple):
            raise TypeError("migrations must be a tuple.")
        if source_blob_store is not None and not isinstance(
            source_blob_store, ManagedSourceBlobStore
        ):
            raise TypeError(
                "source_blob_store must be a ManagedSourceBlobStore or None."
            )
        if not isinstance(corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if corpus_version is not None and not isinstance(
            corpus_version, CorpusVersionId
        ):
            raise TypeError("corpus_version must be a CorpusVersionId or None.")
        self._database_path = database_path
        self._busy_timeout_ms = busy_timeout_ms
        self._migrations = migrations
        self._source_blob_store = source_blob_store
        self._requested_corpus_id = corpus_id
        self._requested_corpus_version = corpus_version
        self._connection: sqlite3.Connection | None = None
        self._version: CorpusVersion | None = None
        self._manifest: CorpusVersionManifest | None = None
        self._visible_document_ids: frozenset[DocumentId] = frozenset()
        self._visible_document_version_ids: frozenset[DocumentVersionId] = frozenset()
        self._visible_membership_keys: frozenset[
            tuple[DocumentId, DocumentVersionId]
        ] = frozenset()
        self._source_span_indexes: dict[RecordId, _SourceSpanIndex] = {}
        self._closed = False

    @property
    def corpus_id(self) -> CorpusId:
        """Return the selected corpus identity while the snapshot is open."""
        return self._open_version().corpus_id

    @property
    def version(self) -> CorpusVersion:
        """Return the committed version selected on entry."""
        return self._open_version()

    @property
    def manifest(self) -> CorpusVersionManifest:
        """Return the complete committed manifest selected on entry."""
        self._open_version()
        manifest = self._manifest
        if manifest is None:
            raise StorageTransactionError()
        return manifest

    def get_document(self, document_id: DocumentId) -> DocumentIdentity | None:
        """Return a document only when the pinned manifest contains it."""
        connection = self._reader_connection()
        if not isinstance(document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if document_id not in self._visible_document_ids:
            return None
        document = read_document(connection, document_id)
        if document is None:
            raise CorruptStorageError()
        return document

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        """Return a version only when the pinned manifest contains it."""
        connection = self._reader_connection()
        if not isinstance(document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if document_version_id not in self._visible_document_version_ids:
            return None
        document_version = read_document_version(connection, document_version_id)
        if document_version is None or not self._is_visible_document_version(
            document_version
        ):
            raise CorruptStorageError()
        return document_version

    def list_document_versions(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentVersion, ...]:
        """Return only versions included in the pinned manifest."""
        connection = self._reader_connection()
        if not isinstance(document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if document_id not in self._visible_document_ids:
            return ()
        return tuple(
            version
            for version in read_document_versions(connection, document_id)
            if self._is_visible_document_version(version)
        )

    def get_source_artefact(
        self,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        """Return source metadata only for a version in the pinned manifest."""
        connection = self._reader_connection()
        self._require_document_version_id(document_version_id)
        if document_version_id not in self._visible_document_version_ids:
            return None
        return get_source_artefact(connection, self.corpus_id, document_version_id)

    def get_chunk_set_selection(
        self,
        document_version_id: DocumentVersionId,
        configuration_identity: ConfigurationIdentity,
    ) -> ChunkSetSelectionRecord | None:
        """Return chunk metadata only for a version in the pinned manifest."""
        connection = self._reader_connection()
        self._require_document_version_id(document_version_id)
        if not isinstance(configuration_identity, ConfigurationIdentity):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")
        if document_version_id not in self._visible_document_version_ids:
            return None

        selection = read_chunk_set_selection(
            connection,
            self.corpus_id,
            document_version_id,
            configuration_identity,
        )
        if selection is None:
            return None
        if (
            selection.document_id,
            selection.document_version_id,
        ) not in self._visible_membership_keys:
            raise CorruptStorageError()
        return selection

    def read_source_bytes(
        self,
        document_version_id: DocumentVersionId,
    ) -> bytes | None:
        """Return verified source bytes only for a visible source version."""
        connection = self._reader_connection()
        self._require_document_version_id(document_version_id)
        if document_version_id not in self._visible_document_version_ids:
            return None
        return read_source_bytes(
            connection,
            self._source_blob_store,
            self.corpus_id,
            document_version_id,
        )

    def stream_source_bytes(
        self,
        document_version_id: DocumentVersionId,
    ) -> Iterator[bytes] | None:
        """Return detached verified chunks only for a visible source version."""
        connection = self._reader_connection()
        self._require_document_version_id(document_version_id)
        if document_version_id not in self._visible_document_version_ids:
            return None
        return stream_source_bytes(
            connection,
            self._source_blob_store,
            self.corpus_id,
            document_version_id,
        )

    def get_parse_outcome(self, parse_id: RecordId) -> ParseOutcome | None:
        """Return an outcome only when its source version is visible."""
        connection = self._reader_connection()
        if not isinstance(parse_id, RecordId):
            raise TypeError("parse_id must be a RecordId.")
        if not self._record_source_is_visible(
            connection,
            "SELECT document_version_id FROM parse_outcomes WHERE parse_id = ?",
            parse_id.as_str(),
        ):
            return None
        outcome = get_parse_outcome(connection, self.corpus_id, parse_id)
        if outcome is None:
            raise CorruptStorageError()
        return outcome

    def get_parsed_representation(
        self,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        """Return a parsed aggregate only when its source version is visible."""
        connection = self._reader_connection()
        if not isinstance(record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        if not self._record_source_is_visible(
            connection,
            "SELECT document_version_id FROM parsed_representations WHERE record_id = ?",
            record_id.as_str(),
        ):
            return None
        representation = get_parsed_representation(
            connection,
            self.corpus_id,
            record_id,
        )
        if representation is None:
            raise CorruptStorageError()
        return representation

    def resolve_source_span(
        self,
        source_span_id: SourceSpanId,
    ) -> ResolvedSourceSpan | None:
        """Resolve a span only when its source version is visible."""
        connection = self._reader_connection()
        if not isinstance(source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")
        try:
            row = connection.execute(
                """SELECT span.parsed_record_id
                FROM source_spans AS span
                JOIN parsed_representations AS representation
                    ON representation.record_id = span.parsed_record_id
                    AND representation.document_version_id = span.document_version_id
                JOIN corpus_document_version_memberships AS membership
                    ON membership.document_version_id = representation.document_version_id
                    AND membership.document_id = representation.document_id
                WHERE membership.corpus_id = ? AND span.source_span_id = ?""",
                (self.corpus_id.as_str(), source_span_id.as_str()),
            ).fetchone()
        except sqlite3.Error as error:
            raise translate_sqlite_error(
                error,
                default=RepositoryUnavailableError,
            ) from None
        if row is None:
            return None
        try:
            record_id = RecordId.from_str(row["parsed_record_id"])
        except (TypeError, ValueError, KeyError, IndexError):
            raise CorruptStorageError()
        index = self._source_span_indexes.get(record_id)
        if index is None:
            representation = self.get_parsed_representation(record_id)
            if representation is None:
                raise CorruptStorageError()
            index = _SourceSpanIndex.from_representation(representation)
            self._source_span_indexes[record_id] = index
        location = index.locations.get(source_span_id)
        if location is None:
            raise CorruptStorageError()
        resolved = ResolvedSourceSpan(
            document_id=index.document_id,
            location=location,
            page_layout=(
                None
                if location.page_number is None
                else index.layouts.get(location.page_number)
            ),
        )
        return resolved

    def find_document_versions_by_content_hash(
        self,
        content_hash: ContentHash,
    ) -> tuple[DocumentVersion, ...]:
        """Return exact-hash candidates included in the pinned manifest."""
        connection = self._reader_connection()
        return tuple(
            version
            for version in read_versions_by_content_hash(
                connection,
                self.corpus_id,
                content_hash,
            )
            if self._is_visible_document_version(version)
        )

    def find_document_versions_by_external_identifiers(
        self,
        external_identifiers: tuple[ExternalIdentifier, ...],
    ) -> tuple[DocumentVersion, ...]:
        """Return identifier candidates included in the pinned manifest."""
        connection = self._reader_connection()
        return tuple(
            version
            for version in read_versions_by_identifiers(
                connection,
                self.corpus_id,
                external_identifiers,
            )
            if self._is_visible_document_version(version)
        )

    def get_source_observation(
        self,
        source_observation_id: SourceObservationId,
    ) -> SourceObservationState | None:
        """Return the latest observation state visible in the pinned manifest."""
        history = self.list_source_observation_history(source_observation_id)
        return None if not history else history[-1]

    def list_source_observation_history(
        self,
        source_observation_id: SourceObservationId,
    ) -> tuple[SourceObservationState, ...]:
        """Return only observation states whose exact version is visible."""
        connection = self._reader_connection()
        return tuple(
            state
            for state in read_source_observation_history(
                connection,
                self.corpus_id,
                source_observation_id,
            )
            if (state.document_id, state.document_version_id)
            in self._visible_membership_keys
        )

    def list_external_identifier_assertions(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[ExternalIdentifierAssertion, ...]:
        """Return assertions only for a version in the pinned manifest."""
        connection = self._reader_connection()
        if not isinstance(document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if document_version_id not in self._visible_document_version_ids:
            return ()
        return read_external_identifier_assertions(
            connection,
            self.corpus_id,
            document_version_id,
        )

    def list_duplicate_relationships(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[DuplicateRelationship, ...]:
        """Return duplicate relationships with two visible endpoints."""
        connection = self._reader_connection()
        if not isinstance(document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if document_version_id not in self._visible_document_version_ids:
            return ()
        return tuple(
            relationship
            for relationship in read_duplicate_relationships(
                connection,
                self.corpus_id,
                document_version_id,
            )
            if relationship.first_version_id in self._visible_document_version_ids
            and relationship.second_version_id in self._visible_document_version_ids
        )

    def get_document_family(
        self,
        family_id: DocumentFamilyId,
    ) -> DocumentFamilyId | None:
        """Return a family only when it has visible membership."""
        connection = self._reader_connection()
        if not isinstance(family_id, DocumentFamilyId):
            raise TypeError("family_id must be a DocumentFamilyId.")
        if not self._visible_family_memberships(
            read_family_memberships_by_family(connection, self.corpus_id, family_id)
        ):
            return None
        family = read_document_family(connection, family_id)
        if family is None:
            raise CorruptStorageError()
        return family

    def list_family_memberships_by_family(
        self,
        family_id: DocumentFamilyId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return only a family's memberships visible in this version."""
        connection = self._reader_connection()
        return self._visible_family_memberships(
            read_family_memberships_by_family(connection, self.corpus_id, family_id)
        )

    def list_family_memberships_by_document(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return family memberships only for a visible document."""
        connection = self._reader_connection()
        if not isinstance(document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if document_id not in self._visible_document_ids:
            return ()
        return self._visible_family_memberships(
            read_family_memberships_by_document(
                connection,
                self.corpus_id,
                document_id,
            )
        )

    def list_family_memberships_by_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> tuple[DocumentFamilyMembership, ...]:
        """Return family memberships only for a visible version."""
        connection = self._reader_connection()
        if not isinstance(document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if document_version_id not in self._visible_document_version_ids:
            return ()
        return self._visible_family_memberships(
            read_family_memberships_by_version(
                connection,
                self.corpus_id,
                document_version_id,
            )
        )

    @property
    def is_closed(self) -> bool:
        """Return whether this snapshot has permanently released its reader."""
        return self._closed

    def __enter__(self) -> Self:
        """Open and pin the dedicated SQLite read transaction."""
        if self._closed:
            raise RepositoryClosedError()
        if self._connection is not None:
            raise StorageTransactionError()

        connection: sqlite3.Connection | None = None
        try:
            connection = open_current_database(
                self._database_path,
                busy_timeout_ms=self._busy_timeout_ms,
                migrations=self._migrations,
            )
            connection.execute("PRAGMA query_only = ON")
            query_only = connection.execute("PRAGMA query_only").fetchone()
            if query_only is None or query_only[0] != 1:
                raise RepositoryUnavailableError()
            connection.execute("BEGIN")
            version = self._select_version(connection)
            update = get_corpus_update(
                connection,
                version.corpus_id,
                version.corpus_version,
            )
            if update is None or update.state is not CorpusUpdateState.COMMITTED:
                raise CorruptStorageError()
            manifest = _published_manifest(connection, update)
        except StorageError as error:
            close_preserving_primary(connection)
            self._closed = True
            safe_log_failure(
                "corpus_snapshot_open_failed",
                error,
                corpus_id=self._requested_corpus_id.as_str(),
                corpus_version=(
                    None
                    if self._requested_corpus_version is None
                    else self._requested_corpus_version.as_str()
                ),
            )
            raise
        except sqlite3.Error as sqlite_error:
            close_preserving_primary(connection)
            self._closed = True
            error = translate_sqlite_error(
                sqlite_error,
                default=StorageTransactionError,
            )
            safe_log_failure(
                "corpus_snapshot_open_failed",
                error,
                corpus_id=self._requested_corpus_id.as_str(),
                corpus_version=(
                    None
                    if self._requested_corpus_version is None
                    else self._requested_corpus_version.as_str()
                ),
            )
            raise error from None

        self._connection = connection
        self._version = version
        self._manifest = manifest
        self._visible_document_ids = frozenset(
            membership.document_id for membership in manifest.document_memberships
        )
        self._visible_document_version_ids = frozenset(
            membership.document_version_id
            for membership in manifest.document_memberships
        )
        self._visible_membership_keys = frozenset(
            (membership.document_id, membership.document_version_id)
            for membership in manifest.document_memberships
        )
        safe_log(
            "info",
            "corpus_snapshot_opened",
            corpus_id=version.corpus_id.as_str(),
            corpus_version=version.corpus_version.as_str(),
            sequence=version.sequence,
            document_count=len(manifest.document_memberships),
            component_count=len(manifest.component_generations),
        )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Close the reader without replacing a primary exception."""
        del exception_type, traceback
        try:
            self.close()
        except StorageError:
            if exception is None:
                raise
        return False

    def close(self) -> None:
        """Release the reader idempotently and reject subsequent use."""
        if self._closed:
            return
        connection = self._connection
        version = self._version
        self._connection = None
        self._closed = True
        self._source_span_indexes.clear()
        if connection is None:
            return

        rolled_back = rollback_preserving_primary(connection)
        closed = close_preserving_primary(connection)
        if not rolled_back or not closed:
            error: StorageError = (
                StorageTransactionError()
                if not rolled_back
                else RepositoryUnavailableError()
            )
            safe_log_failure(
                "corpus_snapshot_close_failed",
                error,
                corpus_id=(
                    self._requested_corpus_id.as_str()
                    if version is None
                    else version.corpus_id.as_str()
                ),
                corpus_version=(
                    None if version is None else version.corpus_version.as_str()
                ),
            )
            raise error

        if version is not None:
            safe_log(
                "info",
                "corpus_snapshot_closed",
                corpus_id=version.corpus_id.as_str(),
                corpus_version=version.corpus_version.as_str(),
                sequence=version.sequence,
            )

    def _select_version(self, connection: sqlite3.Connection) -> CorpusVersion:
        requested = self._requested_corpus_version
        version = (
            get_active_corpus_version(connection, self._requested_corpus_id)
            if requested is None
            else get_corpus_version(connection, self._requested_corpus_id, requested)
        )
        if version is None:
            raise StorageConflictError()
        return version

    def _open_version(self) -> CorpusVersion:
        if self._closed:
            raise RepositoryClosedError()
        version = self._version
        if self._connection is None or version is None:
            raise StorageTransactionError()
        return version

    def _reader_connection(self) -> sqlite3.Connection:
        self._open_version()
        connection = self._connection
        if connection is None:
            raise StorageTransactionError()
        return connection

    def _is_visible_document_version(self, version: DocumentVersion) -> bool:
        return (
            version.document_id,
            version.document_version_id,
        ) in self._visible_membership_keys

    @staticmethod
    def _require_document_version_id(document_version_id: object) -> None:
        if not isinstance(document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")

    def _record_source_is_visible(
        self,
        connection: sqlite3.Connection,
        statement: str,
        record_id: str,
    ) -> bool:
        try:
            row = connection.execute(statement, (record_id,)).fetchone()
            if row is None:
                return False
            document_version_id = DocumentVersionId.from_str(row["document_version_id"])
        except (TypeError, ValueError, KeyError, IndexError):
            raise CorruptStorageError() from None
        except sqlite3.Error as error:
            raise translate_sqlite_error(
                error,
                default=RepositoryUnavailableError,
            ) from None
        return document_version_id in self._visible_document_version_ids

    def _visible_family_memberships(
        self,
        memberships: tuple[DocumentFamilyMembership, ...],
    ) -> tuple[DocumentFamilyMembership, ...]:
        return tuple(
            membership
            for membership in memberships
            if (
                membership.document_id in self._visible_document_ids
                if membership.document_version_id is None
                else (
                    membership.document_id,
                    membership.document_version_id,
                )
                in self._visible_membership_keys
            )
        )


def create_active_corpus_snapshot(
    connection: sqlite3.Connection,
    migrations: tuple[MigrationDefinition, ...],
    source_blob_store: ManagedSourceBlobStore | None,
    corpus_id: CorpusId,
) -> SQLiteCorpusSnapshot:
    """Create an unopened owner that will select the active version on entry."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    database_path, busy_timeout_ms = _database_details(connection)
    return SQLiteCorpusSnapshot(
        database_path,
        busy_timeout_ms,
        migrations,
        source_blob_store,
        corpus_id,
        None,
    )


def create_corpus_snapshot(
    connection: sqlite3.Connection,
    migrations: tuple[MigrationDefinition, ...],
    source_blob_store: ManagedSourceBlobStore | None,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> SQLiteCorpusSnapshot:
    """Create an unopened owner for one explicit historical version."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")
    database_path, busy_timeout_ms = _database_details(connection)
    return SQLiteCorpusSnapshot(
        database_path,
        busy_timeout_ms,
        migrations,
        source_blob_store,
        corpus_id,
        corpus_version,
    )
