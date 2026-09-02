"""Internal SQLite persistence for immutable document versions."""

import sqlite3
from collections.abc import Sequence

from inference_lab.provenance import (
    ContentHash,
    DocumentId,
    DocumentVersion,
    DocumentVersionId,
    ExternalIdentifier,
    HashAlgorithm,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)


def _external_identifiers_for_version(
    connection: sqlite3.Connection,
    document_version_id: DocumentVersionId,
) -> tuple[ExternalIdentifier, ...]:
    rows = connection.execute(
        """SELECT ordinal, namespace, identifier_value
        FROM external_identifiers
        WHERE document_version_id = ?
        ORDER BY ordinal""",
        (document_version_id.as_str(),),
    ).fetchall()
    ordinals = tuple(row["ordinal"] for row in rows)
    if ordinals != tuple(range(len(rows))):
        raise CorruptStorageError()
    return tuple(
        ExternalIdentifier(
            namespace=row["namespace"],
            value=row["identifier_value"],
        )
        for row in rows
    )


def _reconstruct_document_version(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> DocumentVersion:
    document_version_id = DocumentVersionId.from_str(row["document_version_id"])
    return DocumentVersion(
        document_version_id=document_version_id,
        document_id=DocumentId.from_str(row["document_id"]),
        content_hash=ContentHash(
            algorithm=HashAlgorithm(row["content_hash_algorithm"]),
            digest=row["content_hash_digest"],
        ),
        external_identifiers=_external_identifiers_for_version(
            connection,
            document_version_id,
        ),
    )


def _reconstruct_document_versions(
    connection: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
) -> tuple[DocumentVersion, ...]:
    try:
        return tuple(_reconstruct_document_version(connection, row) for row in rows)
    except CorruptStorageError:
        raise
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def get_document_version(
    connection: sqlite3.Connection,
    document_version_id: DocumentVersionId,
) -> DocumentVersion | None:
    """Reconstruct one immutable document version and its identifiers."""
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")

    try:
        row = connection.execute(
            """SELECT document_version_id, document_id,
                content_hash_algorithm, content_hash_digest
            FROM document_versions
            WHERE document_version_id = ?""",
            (document_version_id.as_str(),),
        ).fetchone()
        if row is None:
            return None
        return _reconstruct_document_versions(connection, (row,))[0]
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_document_versions(
    connection: sqlite3.Connection,
    document_id: DocumentId,
) -> tuple[DocumentVersion, ...]:
    """Return a document's versions in canonical identifier order."""
    if not isinstance(document_id, DocumentId):
        raise TypeError("document_id must be a DocumentId.")

    try:
        rows = connection.execute(
            """SELECT document_version_id, document_id,
                content_hash_algorithm, content_hash_digest
            FROM document_versions
            WHERE document_id = ?
            ORDER BY document_version_id""",
            (document_id.as_str(),),
        ).fetchall()
        return _reconstruct_document_versions(connection, rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def add_document_version(
    connection: sqlite3.Connection,
    document_version: DocumentVersion,
) -> None:
    """Insert one version and its ordered external identifiers."""
    if not isinstance(document_version, DocumentVersion):
        raise TypeError("document_version must be a DocumentVersion.")

    existing = get_document_version(
        connection,
        document_version.document_version_id,
    )
    if existing == document_version:
        return
    if existing is not None:
        raise StorageConflictError()

    try:
        connection.execute(
            """INSERT INTO document_versions (
                document_version_id,
                document_id,
                content_hash_algorithm,
                content_hash_digest
            ) VALUES (?, ?, ?, ?)""",
            (
                document_version.document_version_id.as_str(),
                document_version.document_id.as_str(),
                document_version.content_hash.algorithm.value,
                document_version.content_hash.digest,
            ),
        )
        connection.executemany(
            """INSERT INTO external_identifiers (
                document_version_id,
                ordinal,
                namespace,
                identifier_value
            ) VALUES (?, ?, ?, ?)""",
            (
                (
                    document_version.document_version_id.as_str(),
                    ordinal,
                    identifier.namespace,
                    identifier.value,
                )
                for ordinal, identifier in enumerate(
                    document_version.external_identifiers
                )
            ),
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
