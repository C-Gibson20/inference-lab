"""Internal SQLite persistence for immutable managed source metadata."""

import sqlite3
from collections.abc import Iterator

from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    DocumentId,
    DocumentVersionId,
    HashAlgorithm,
)
from inference_lab.storage._source_blob_store import ManagedSourceBlobStore
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage.source_records import SourceArtefact


def _reconstruct_source_artefact(row: sqlite3.Row) -> SourceArtefact:
    """Reconstruct validated managed-source metadata from one database row."""
    try:
        return SourceArtefact(
            document_id=DocumentId.from_str(row["document_id"]),
            document_version_id=DocumentVersionId.from_str(row["document_version_id"]),
            content_hash=ContentHash(
                algorithm=HashAlgorithm(row["content_hash_algorithm"]),
                digest=row["content_hash_digest"],
            ),
            byte_count=row["byte_count"],
            media_type=row["media_type"],
            storage_key=row["storage_key"],
            record_schema_version=row["record_schema_version"],
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def _source_artefact_for_version(
    connection: sqlite3.Connection,
    document_version_id: DocumentVersionId,
) -> SourceArtefact | None:
    """Return global immutable metadata for replay checks."""
    try:
        row = connection.execute(
            """SELECT document_version_id, document_id, record_schema_version,
                content_hash_algorithm, content_hash_digest, byte_count,
                media_type, storage_key
            FROM source_artefacts
            WHERE document_version_id = ?""",
            (document_version_id.as_str(),),
        ).fetchone()
        if row is None:
            return None
        return _reconstruct_source_artefact(row)
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def get_source_artefact(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
) -> SourceArtefact | None:
    """Return source metadata only through an agreeing corpus membership."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")

    try:
        row = connection.execute(
            """SELECT artefact.document_version_id, artefact.document_id,
                artefact.record_schema_version,
                artefact.content_hash_algorithm,
                artefact.content_hash_digest,
                artefact.byte_count, artefact.media_type, artefact.storage_key,
                membership.document_id AS membership_document_id,
                version.content_hash_algorithm AS version_hash_algorithm,
                version.content_hash_digest AS version_hash_digest
            FROM corpus_document_version_memberships AS membership
            JOIN document_versions AS version
                ON version.document_version_id = membership.document_version_id
                AND version.document_id = membership.document_id
            LEFT JOIN source_artefacts AS artefact
                ON artefact.document_version_id = membership.document_version_id
            WHERE membership.corpus_id = ?
                AND membership.document_version_id = ?""",
            (corpus_id.as_str(), document_version_id.as_str()),
        ).fetchone()
        if row is None or row["document_version_id"] is None:
            return None
        artefact = _reconstruct_source_artefact(row)
        if (
            artefact.document_id.as_str() != row["membership_document_id"]
            or artefact.content_hash.algorithm.value != row["version_hash_algorithm"]
            or artefact.content_hash.digest != row["version_hash_digest"]
        ):
            raise CorruptStorageError()
        return artefact
    except CorruptStorageError:
        raise
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def read_source_bytes(
    connection: sqlite3.Connection,
    blob_store: ManagedSourceBlobStore | None,
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
) -> bytes | None:
    """Return fully verified bytes for corpus-visible source metadata."""
    artefact = get_source_artefact(connection, corpus_id, document_version_id)
    if artefact is None:
        return None
    if blob_store is None:
        raise RepositoryUnavailableError()
    return blob_store.read_bytes(artefact)


def stream_source_bytes(
    connection: sqlite3.Connection,
    blob_store: ManagedSourceBlobStore | None,
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
) -> Iterator[bytes] | None:
    """Return fully verified, detached chunks for corpus-visible metadata."""
    artefact = get_source_artefact(connection, corpus_id, document_version_id)
    if artefact is None:
        return None
    if blob_store is None:
        raise RepositoryUnavailableError()
    return blob_store.stream_bytes(artefact)


def add_source_artefact(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    artefact: SourceArtefact,
) -> None:
    """Add corpus-authorised source metadata or accept exact global replay."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(artefact, SourceArtefact):
        raise TypeError("artefact must be a SourceArtefact.")

    try:
        authorised = connection.execute(
            """SELECT 1
            FROM corpus_document_version_memberships AS membership
            JOIN document_versions AS version
                ON version.document_version_id = membership.document_version_id
                AND version.document_id = membership.document_id
            WHERE membership.corpus_id = ?
                AND membership.document_id = ?
                AND membership.document_version_id = ?
                AND version.content_hash_algorithm = ?
                AND version.content_hash_digest = ?""",
            (
                corpus_id.as_str(),
                artefact.document_id.as_str(),
                artefact.document_version_id.as_str(),
                artefact.content_hash.algorithm.value,
                artefact.content_hash.digest,
            ),
        ).fetchone()
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
    if authorised is None:
        raise StorageConflictError()

    stored = _source_artefact_for_version(
        connection,
        artefact.document_version_id,
    )
    if stored is not None:
        if stored == artefact:
            return
        raise StorageConflictError()

    try:
        connection.execute(
            """INSERT INTO source_artefacts (
                document_version_id,
                document_id,
                record_schema_version,
                content_hash_algorithm,
                content_hash_digest,
                byte_count,
                media_type,
                storage_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artefact.document_version_id.as_str(),
                artefact.document_id.as_str(),
                artefact.record_schema_version,
                artefact.content_hash.algorithm.value,
                artefact.content_hash.digest,
                artefact.byte_count,
                artefact.media_type,
                artefact.storage_key,
            ),
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
