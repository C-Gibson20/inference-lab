"""Internal SQLite persistence for immutable document versions."""

import sqlite3
from collections.abc import Sequence

from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    DocumentId,
    DocumentVersion,
    DocumentVersionId,
    ExternalIdentifier,
    HashAlgorithm,
)
from inference_lab.provenance.values import validate_external_identifiers
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


def find_document_versions_by_content_hash(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    content_hash: ContentHash,
) -> tuple[DocumentVersion, ...]:
    """Return exact-hash candidates in resolution-request order."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(content_hash, ContentHash):
        raise TypeError("content_hash must be a ContentHash.")

    try:
        rows = connection.execute(
            """SELECT versions.document_version_id, versions.document_id,
                versions.content_hash_algorithm, versions.content_hash_digest
            FROM document_versions AS versions
            JOIN corpus_document_version_memberships AS membership
                ON membership.document_version_id = versions.document_version_id
            WHERE membership.corpus_id = ?
                AND content_hash_algorithm = ? AND content_hash_digest = ?
            ORDER BY versions.document_id, versions.document_version_id""",
            (
                corpus_id.as_str(),
                content_hash.algorithm.value,
                content_hash.digest,
            ),
        ).fetchall()
        return _reconstruct_document_versions(connection, rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def find_document_versions_by_external_identifiers(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    external_identifiers: tuple[ExternalIdentifier, ...],
) -> tuple[DocumentVersion, ...]:
    """Return unique versions matching any supplied external identifier."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    validate_external_identifiers(external_identifiers)
    if not external_identifiers:
        return ()

    try:
        matched_ids: set[str] = set()
        for identifier in external_identifiers:
            rows = connection.execute(
                """SELECT identifiers.document_version_id
                FROM external_identifiers AS identifiers
                JOIN corpus_document_version_memberships AS membership
                    ON membership.document_version_id = identifiers.document_version_id
                WHERE membership.corpus_id = ?
                    AND namespace = ? AND identifier_value = ?
                UNION
                SELECT document_version_id
                FROM external_identifier_assertions
                WHERE corpus_id = ?
                    AND namespace = ? AND identifier_value = ?""",
                (
                    corpus_id.as_str(),
                    *identifier.as_tuple(),
                    corpus_id.as_str(),
                    *identifier.as_tuple(),
                ),
            ).fetchall()
            matched_ids.update(row["document_version_id"] for row in rows)

        candidates: list[DocumentVersion] = []
        for stored_id in matched_ids:
            candidate = get_document_version(
                connection,
                DocumentVersionId.from_str(stored_id),
            )
            if candidate is None:
                raise CorruptStorageError()
            candidates.append(candidate)
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.document_id.as_str(),
                    candidate.document_version_id.as_str(),
                ),
            )
        )
    except CorruptStorageError:
        raise
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
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
