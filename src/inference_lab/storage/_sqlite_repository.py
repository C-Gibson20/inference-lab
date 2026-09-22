"""Internal SQLite metadata repository lifecycle owner."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final, Self

from structlog.contextvars import bound_contextvars

from inference_lab._clock import utc_now
from inference_lab.provenance import (
    ConfigurationIdentity,
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
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
    KnownSourceSpan,
    RecordId,
    SourceObservationId,
    SourceObservationState,
    SourceSpanId,
)
from inference_lab.storage._source_blob_store import (
    ManagedSourceBlobStore,
    SourceBlobReconciliation,
)
from inference_lab.storage._sqlite_baseline_checkpoints import (
    get_baseline_checkpoint,
    get_baseline_manifest,
    list_baseline_checkpoints,
)
from inference_lab.storage._sqlite_chunk_sets import get_chunk_set_selection
from inference_lab.storage._sqlite_cleanup import close_preserving_primary
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
from inference_lab.storage._sqlite_corpus_snapshots import (
    SQLiteCorpusSnapshot,
    create_active_corpus_snapshot,
    create_corpus_snapshot,
)
from inference_lab.storage._sqlite_corpus_update_components import (
    list_corpus_update_component_generations,
)
from inference_lab.storage._sqlite_corpus_update_lifecycle import CreatedAtFactory
from inference_lab.storage._sqlite_corpus_update_memberships import (
    list_corpus_update_document_memberships,
)
from inference_lab.storage._sqlite_corpus_update_reads import (
    get_corpus_update,
    list_non_terminal_corpus_updates,
)
from inference_lab.storage._sqlite_corpus_versions import (
    get_active_corpus_version,
    get_corpus_version,
    list_corpus_versions,
)
from inference_lab.storage._sqlite_identifier_assertions import (
    list_external_identifier_assertions,
)
from inference_lab.storage._sqlite_inspection import (
    SchemaInspection,
    validate_database_schema,
)
from inference_lab.storage._sqlite_lifecycle import open_sqlite_database
from inference_lab.storage._sqlite_locator_continuity import (
    find_source_locator_by_path,
    find_source_locators_by_identity_hint,
    get_source_locator,
    list_source_locator_history,
)
from inference_lab.storage._sqlite_logging import safe_log, safe_log_failure
from inference_lab.storage._sqlite_migration_execution import AppliedAtFactory
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage._sqlite_observations import (
    find_source_observations_by_document_version,
    get_source_observation,
    list_source_observation_history,
)
from inference_lab.storage._sqlite_parse_outcomes import get_parse_outcome
from inference_lab.storage._sqlite_parsed_representations import (
    get_parsed_representation,
    list_parsed_representations,
)
from inference_lab.storage._sqlite_relationships import (
    list_duplicate_relationships,
    list_family_memberships_by_document,
    list_family_memberships_by_document_version,
    list_family_memberships_by_family,
)
from inference_lab.storage._sqlite_source_artefacts import (
    get_source_artefact,
    read_source_bytes,
    stream_source_bytes,
)
from inference_lab.storage._sqlite_source_spans import (
    list_known_source_spans,
    resolve_source_span,
)
from inference_lab.storage._sqlite_transaction import SQLiteMetadataTransaction
from inference_lab.storage._sqlite_versions import (
    find_document_versions_by_content_hash,
    find_document_versions_by_external_identifiers,
    get_document_version,
    list_document_versions,
)
from inference_lab.storage.baseline_checkpoint_records import (
    BaselineCheckpointRecord,
    BaselineCheckpointRecordStage,
    BaselineManifestRecord,
)
from inference_lab.storage.chunk_records import ChunkSetSelectionRecord
from inference_lab.storage.errors import (
    RepositoryClosedError,
    RepositoryUnavailableError,
    StorageConflictError,
    StorageError,
    StorageTransactionError,
)
from inference_lab.storage.locator_continuity import (
    SourceFileIdentityHint,
    SourceLocatorRevision,
)
from inference_lab.storage.parse_records import ParseOutcome
from inference_lab.storage.parsed_records import ParsedRepresentation
from inference_lab.storage.source_locations import ResolvedSourceSpan
from inference_lab.storage.source_records import SourceArtefact
from inference_lab.storage.versioning import (
    ComponentGenerationReference,
    CorpusUpdate,
    CorpusVersion,
)

DEFAULT_SOURCE_RECONCILIATION_MAX_ENTRIES: Final = 10_000


def _validate_source_reconciliation_max_entries(
    max_entries: int,
    *,
    parameter_name: str,
) -> None:
    if not isinstance(max_entries, int) or isinstance(max_entries, bool):
        raise TypeError(f"{parameter_name} must be an integer.")
    if max_entries <= 0:
        raise ValueError(f"{parameter_name} must be greater than zero.")


class SQLiteMetadataRepository:
    """Own one validated SQLite connection and its explicit lifecycle."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        migrations: tuple[MigrationDefinition, ...],
        *,
        created_at_factory: CreatedAtFactory = utc_now,
        source_blob_store: ManagedSourceBlobStore | None = None,
        database_path: Path | None = None,
        source_dir: Path | None = None,
    ) -> None:
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("connection must be a sqlite3.Connection.")
        if not isinstance(migrations, tuple):
            raise TypeError("migrations must be a tuple.")
        if not callable(created_at_factory):
            raise TypeError("created_at_factory must be callable.")
        if source_blob_store is not None and not isinstance(
            source_blob_store, ManagedSourceBlobStore
        ):
            raise TypeError(
                "source_blob_store must be a ManagedSourceBlobStore or None."
            )
        if database_path is not None and not isinstance(database_path, Path):
            raise TypeError("database_path must be a Path or None.")
        if source_dir is not None and not isinstance(source_dir, Path):
            raise TypeError("source_dir must be a Path or None.")
        self._connection: sqlite3.Connection | None = connection
        self._migrations = migrations
        self._created_at_factory = created_at_factory
        self._source_blob_store = source_blob_store
        self._database_path = database_path
        self._source_dir = source_dir
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
        created_at_factory: CreatedAtFactory = utc_now,
        source_dir: Path | None = None,
        source_reconciliation_max_entries: int = (
            DEFAULT_SOURCE_RECONCILIATION_MAX_ENTRIES
        ),
        reconcile_sources: bool = True,
    ) -> Self:
        """Open a validated repository under the explicit schema policy."""
        if not isinstance(path, Path):
            raise TypeError("path must be a Path.")
        if not callable(created_at_factory):
            raise TypeError("created_at_factory must be callable.")
        _validate_source_reconciliation_max_entries(
            source_reconciliation_max_entries,
            parameter_name="source_reconciliation_max_entries",
        )
        if not isinstance(reconcile_sources, bool):
            raise TypeError("reconcile_sources must be a boolean.")
        source_blob_store = (
            ManagedSourceBlobStore(source_dir) if source_dir is not None else None
        )
        try:
            resolved_database_path = path.expanduser().resolve(strict=False)
            resolved_source_dir = (
                source_dir.expanduser().resolve(strict=False)
                if source_dir is not None
                else None
            )
        except (OSError, RuntimeError):
            raise RepositoryUnavailableError() from None
        connection = open_sqlite_database(
            path,
            apply_migrations=apply_migrations,
            busy_timeout_ms=busy_timeout_ms,
            migrations=migrations,
            applied_at_factory=applied_at_factory,
        )
        repository = cls(
            connection,
            migrations,
            created_at_factory=created_at_factory,
            source_blob_store=source_blob_store,
            database_path=resolved_database_path,
            source_dir=resolved_source_dir,
        )
        if source_blob_store is not None and reconcile_sources:
            try:
                repository.reconcile_source_storage(
                    max_entries=source_reconciliation_max_entries
                )
            except BaseException:
                close_preserving_primary(connection)
                raise
        return repository

    def open_worker_repository(self) -> Self:
        """Open a validated worker connection without startup reconciliation."""
        if self._connection is None:
            raise RepositoryClosedError()
        if self._database_path is None or self._source_dir is None:
            raise RepositoryUnavailableError()
        return type(self).open(
            self._database_path,
            migrations=self._migrations,
            created_at_factory=self._created_at_factory,
            source_dir=self._source_dir,
            reconcile_sources=False,
        )

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

    def get_active_corpus_version(
        self,
        corpus_id: CorpusId,
    ) -> CorpusVersion | None:
        """Return a corpus's active committed version, or ``None``."""
        return get_active_corpus_version(self._connection_or_raise(), corpus_id)

    def get_corpus_version(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CorpusVersion | None:
        """Return one committed version owned by a corpus, or ``None``."""
        return get_corpus_version(
            self._connection_or_raise(),
            corpus_id,
            corpus_version,
        )

    def list_corpus_versions(
        self,
        corpus_id: CorpusId,
    ) -> tuple[CorpusVersion, ...]:
        """Return committed history in ascending sequence order."""
        return list_corpus_versions(self._connection_or_raise(), corpus_id)

    def get_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> CorpusUpdate | None:
        """Return one owned durable corpus update, or ``None``."""
        return get_corpus_update(
            self._connection_or_raise(),
            corpus_id,
            corpus_version,
        )

    def list_non_terminal_corpus_updates(self) -> tuple[CorpusUpdate, ...]:
        """Return durable preparing and ready work for explicit recovery."""
        return list_non_terminal_corpus_updates(self._connection_or_raise())

    def get_baseline_manifest(
        self,
        corpus_id: CorpusId,
        proposed_corpus_version: CorpusVersionId,
    ) -> BaselineManifestRecord | None:
        """Return one durable baseline manifest, or ``None`` when absent."""
        return get_baseline_manifest(
            self._connection_or_raise(),
            corpus_id,
            proposed_corpus_version,
        )

    def get_baseline_checkpoint(
        self,
        corpus_id: CorpusId,
        proposed_corpus_version: CorpusVersionId,
        document_version_id: DocumentVersionId,
        stage: BaselineCheckpointRecordStage,
        configuration_identity: ConfigurationIdentity,
    ) -> BaselineCheckpointRecord | None:
        """Return one exact baseline checkpoint, or ``None`` when absent."""
        return get_baseline_checkpoint(
            self._connection_or_raise(),
            corpus_id,
            proposed_corpus_version,
            document_version_id,
            stage,
            configuration_identity,
        )

    def list_baseline_checkpoints(
        self,
        corpus_id: CorpusId,
        proposed_corpus_version: CorpusVersionId,
    ) -> tuple[BaselineCheckpointRecord, ...]:
        """Return proposal checkpoints in canonical work-key order."""
        return list_baseline_checkpoints(
            self._connection_or_raise(),
            corpus_id,
            proposed_corpus_version,
        )

    def list_corpus_update_document_memberships(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        """Return deterministic proposed membership for one update."""
        return list_corpus_update_document_memberships(
            self._connection_or_raise(),
            corpus_id,
            corpus_version,
        )

    def list_corpus_update_component_generations(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> tuple[ComponentGenerationReference, ...]:
        """Return deterministic ready generations staged for one update."""
        return list_corpus_update_component_generations(
            self._connection_or_raise(),
            corpus_id,
            corpus_version,
        )

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

    def get_chunk_set_selection(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
        configuration_identity: ConfigurationIdentity,
    ) -> ChunkSetSelectionRecord | None:
        """Return one exact corpus-visible chunk-set selection, or ``None``."""
        return get_chunk_set_selection(
            self._connection_or_raise(),
            corpus_id,
            document_version_id,
            configuration_identity,
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

    def get_source_artefact(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        """Return corpus-visible managed-source metadata, or ``None``."""
        return get_source_artefact(
            self._connection_or_raise(),
            corpus_id,
            document_version_id,
        )

    def read_source_bytes(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> bytes | None:
        """Return fully verified corpus-visible source bytes, or ``None``."""
        return read_source_bytes(
            self._connection_or_raise(),
            self._source_blob_store,
            corpus_id,
            document_version_id,
        )

    def stream_source_bytes(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> Iterator[bytes] | None:
        """Return fully verified, detached source chunks, or ``None``."""
        return stream_source_bytes(
            self._connection_or_raise(),
            self._source_blob_store,
            corpus_id,
            document_version_id,
        )

    def get_parse_outcome(
        self,
        corpus_id: CorpusId,
        parse_id: RecordId,
    ) -> ParseOutcome | None:
        """Return a complete corpus-visible parser outcome, or ``None``."""
        return get_parse_outcome(self._connection_or_raise(), corpus_id, parse_id)

    def get_parsed_representation(
        self,
        corpus_id: CorpusId,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        """Return a complete corpus-visible parsed record, or ``None``."""
        return get_parsed_representation(
            self._connection_or_raise(),
            corpus_id,
            record_id,
        )

    def list_parsed_representations(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[ParsedRepresentation, ...]:
        """Return a version's parsed records in canonical identity order."""
        return list_parsed_representations(
            self._connection_or_raise(),
            corpus_id,
            document_version_id,
        )

    def resolve_source_span(
        self,
        corpus_id: CorpusId,
        source_span_id: SourceSpanId,
    ) -> ResolvedSourceSpan | None:
        """Return a corpus-visible span and its coordinate descriptor."""
        return resolve_source_span(
            self._connection_or_raise(),
            corpus_id,
            source_span_id,
        )

    def list_known_source_spans(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[KnownSourceSpan, ...]:
        """Return a version's validated PROV-03 source-span catalogue."""
        return list_known_source_spans(
            self._connection_or_raise(),
            corpus_id,
            document_version_id,
        )

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

    def find_source_observations_by_document_version(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> tuple[SourceObservationState, ...]:
        """Return current observations of one version in canonical ID order."""
        return find_source_observations_by_document_version(
            self._connection_or_raise(),
            corpus_id,
            document_version_id,
        )

    def get_source_locator(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> SourceLocatorRevision | None:
        """Return an observed source's current private locator, or ``None``."""
        return get_source_locator(
            self._connection_or_raise(),
            corpus_id,
            source_observation_id,
        )

    def list_source_locator_history(
        self,
        corpus_id: CorpusId,
        source_observation_id: SourceObservationId,
    ) -> tuple[SourceLocatorRevision, ...]:
        """Return an observed source's append-only private locator history."""
        return list_source_locator_history(
            self._connection_or_raise(),
            corpus_id,
            source_observation_id,
        )

    def find_source_locator_by_path(
        self,
        corpus_id: CorpusId,
        path: Path,
    ) -> SourceLocatorRevision | None:
        """Return an exact current private-locator match within one corpus."""
        return find_source_locator_by_path(
            self._connection_or_raise(),
            corpus_id,
            path,
        )

    def find_source_locators_by_identity_hint(
        self,
        corpus_id: CorpusId,
        identity_hint: SourceFileIdentityHint,
    ) -> tuple[SourceLocatorRevision, ...]:
        """Return current filesystem-hint candidates within one corpus."""
        return find_source_locators_by_identity_hint(
            self._connection_or_raise(),
            corpus_id,
            identity_hint,
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

    def open_active_corpus_snapshot(
        self,
        corpus_id: CorpusId,
    ) -> SQLiteCorpusSnapshot:
        """Return an unopened reader that will pin the active version."""
        if self._transaction_active:
            raise StorageTransactionError()
        return create_active_corpus_snapshot(
            self._connection_or_raise(),
            self._migrations,
            self._source_blob_store,
            corpus_id,
        )

    def open_corpus_snapshot(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> SQLiteCorpusSnapshot:
        """Return an unopened reader for one committed historical version."""
        if self._transaction_active:
            raise StorageTransactionError()
        return create_corpus_snapshot(
            self._connection_or_raise(),
            self._migrations,
            self._source_blob_store,
            corpus_id,
            corpus_version,
        )

    def reconcile_source_storage(self, *, max_entries: int) -> None:
        """Remove abandoned staging and report bounded final-blob inventory."""
        _validate_source_reconciliation_max_entries(
            max_entries,
            parameter_name="max_entries",
        )
        safe_log("info", "source_reconciliation_started")
        try:
            result = self._reconcile_source_storage(max_entries=max_entries)
        except StorageError as error:
            safe_log_failure("source_reconciliation_failed", error)
            raise
        except Exception:
            safe_log_failure(
                "source_reconciliation_failed",
                StorageTransactionError(),
            )
            raise
        safe_log(
            "info",
            "source_reconciliation_completed",
            entries_examined=result.entries_examined,
            staging_files_removed=result.staging_files_removed,
            referenced_blobs_found=result.referenced_blobs_found,
            orphan_blobs_found=result.orphan_blobs_found,
            ambiguous_entries_retained=result.ambiguous_entries_retained,
            limit_reached=result.limit_reached,
        )

    def _reconcile_source_storage(
        self,
        *,
        max_entries: int,
    ) -> SourceBlobReconciliation:
        connection = self._connection_or_raise()
        if self._transaction_active:
            raise StorageTransactionError()
        blob_store = self._source_blob_store
        if blob_store is None:
            raise RepositoryUnavailableError()
        with blob_store.publication_lock():

            def is_referenced(storage_key: str) -> bool:
                try:
                    row = connection.execute(
                        """SELECT 1 FROM source_artefacts
                        WHERE storage_key = ? LIMIT 1""",
                        (storage_key,),
                    ).fetchone()
                except sqlite3.Error as error:
                    raise translate_sqlite_error(
                        error,
                        default=RepositoryUnavailableError,
                    ) from None
                return row is not None

            return blob_store.reconcile_locked(
                is_referenced,
                max_entries=max_entries,
            )

    def persist_source_artefact(
        self,
        corpus_id: CorpusId,
        artefact: SourceArtefact,
        chunks: Iterable[bytes],
        *,
        max_source_bytes: int,
        parse_outcome: ParseOutcome | None = None,
        parsed_representation: ParsedRepresentation | None = None,
    ) -> None:
        """Publish bytes, source metadata, and optional parse records safely."""
        if not isinstance(corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(artefact, SourceArtefact):
            raise TypeError("artefact must be a SourceArtefact.")
        fields = {
            "corpus_id": corpus_id.as_str(),
            "document_id": artefact.document_id.as_str(),
        }
        with bound_contextvars(**fields):
            safe_log("info", "source_persistence_started")
            try:
                self._persist_source_artefact(
                    corpus_id,
                    artefact,
                    chunks,
                    max_source_bytes=max_source_bytes,
                    parse_outcome=parse_outcome,
                    parsed_representation=parsed_representation,
                )
            except StorageError as error:
                safe_log_failure("source_persistence_failed", error)
                raise
            except Exception:
                safe_log_failure(
                    "source_persistence_failed",
                    StorageTransactionError(),
                )
                raise

            safe_log(
                "info",
                "source_persistence_completed",
                byte_count=artefact.byte_count,
            )

    def _persist_source_artefact(
        self,
        corpus_id: CorpusId,
        artefact: SourceArtefact,
        chunks: Iterable[bytes],
        *,
        max_source_bytes: int,
        parse_outcome: ParseOutcome | None,
        parsed_representation: ParsedRepresentation | None,
    ) -> None:
        """Execute the preflight, blob promotion, and metadata commit sequence."""
        connection = self._connection_or_raise()
        if self._transaction_active:
            raise StorageTransactionError()
        blob_store = self._source_blob_store
        if blob_store is None:
            raise RepositoryUnavailableError()
        with blob_store.publication_lock():
            self._validate_parse_publication(
                artefact,
                parse_outcome,
                parsed_representation,
            )

            membership = get_corpus_document_version_membership(
                connection,
                corpus_id,
                artefact.document_version_id,
            )
            version = get_document_version(connection, artefact.document_version_id)
            if (
                membership is None
                or membership.document_id != artefact.document_id
                or version is None
                or version.document_id != artefact.document_id
                or version.content_hash != artefact.content_hash
            ):
                raise StorageConflictError()
            stored = get_source_artefact(
                connection,
                corpus_id,
                artefact.document_version_id,
            )
            if stored is not None and stored != artefact:
                raise StorageConflictError()

            blob_store.write_chunks(
                artefact,
                chunks,
                max_source_bytes=max_source_bytes,
            )
            with self.transaction() as transaction:
                transaction.add_source_artefact(corpus_id, artefact)
                if parsed_representation is not None:
                    transaction.add_parsed_representation(
                        corpus_id,
                        parsed_representation,
                    )
                if parse_outcome is not None:
                    transaction.add_parse_outcome(corpus_id, parse_outcome)

    @staticmethod
    def _validate_parse_publication(
        artefact: SourceArtefact,
        parse_outcome: ParseOutcome | None,
        parsed_representation: ParsedRepresentation | None,
    ) -> None:
        """Reject an incomplete or mismatched parse bundle before promotion."""
        if parse_outcome is not None and not isinstance(parse_outcome, ParseOutcome):
            raise TypeError("parse_outcome must be a ParseOutcome or None.")
        if parsed_representation is not None and not isinstance(
            parsed_representation,
            ParsedRepresentation,
        ):
            raise TypeError(
                "parsed_representation must be a ParsedRepresentation or None."
            )
        if parse_outcome is None:
            if parsed_representation is not None:
                raise StorageConflictError()
            return

        expected_representation = parse_outcome.parsed_record_id is not None
        if expected_representation != (parsed_representation is not None):
            raise StorageConflictError()
        if (
            parse_outcome.document_id != artefact.document_id
            or parse_outcome.document_version_id != artefact.document_version_id
        ):
            raise StorageConflictError()
        if parsed_representation is None:
            return
        if (
            parsed_representation.record_id != parse_outcome.parsed_record_id
            or parsed_representation.document_id != parse_outcome.document_id
            or parsed_representation.document_version_id
            != parse_outcome.document_version_id
            or parsed_representation.record_schema_version
            != parse_outcome.record_schema_version
            or parsed_representation.parser != parse_outcome.parser
            or parsed_representation.normalisation_version
            != parse_outcome.normalisation_version
            or parsed_representation.ocr_producer != parse_outcome.ocr_producer
        ):
            raise StorageConflictError()

    def _connection_or_raise(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            raise RepositoryClosedError()
        return connection

    def _discard_connection_preserving_primary(self) -> None:
        """Make a cleanup-failed connection unreachable without raising."""
        connection = self._connection
        self._connection = None
        close_preserving_primary(connection)

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
