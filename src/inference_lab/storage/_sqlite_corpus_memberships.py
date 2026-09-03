"""Internal SQLite persistence for corpus catalogue membership."""

import sqlite3
from collections.abc import Sequence

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)


def _reconstruct_memberships(
    rows: Sequence[sqlite3.Row],
) -> tuple[CorpusDocumentVersionMembership, ...]:
    try:
        return tuple(
            CorpusDocumentVersionMembership(
                corpus_id=CorpusId.from_str(row["corpus_id"]),
                document_id=DocumentId.from_str(row["document_id"]),
                document_version_id=DocumentVersionId.from_str(
                    row["document_version_id"]
                ),
            )
            for row in rows
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def get_corpus_document_version_membership(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
) -> CorpusDocumentVersionMembership | None:
    """Return one catalogue membership, or ``None`` when absent."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")

    try:
        row = connection.execute(
            """SELECT corpus_id, document_id, document_version_id
            FROM corpus_document_version_memberships
            WHERE corpus_id = ? AND document_version_id = ?""",
            (corpus_id.as_str(), document_version_id.as_str()),
        ).fetchone()
        if row is None:
            return None
        return _reconstruct_memberships((row,))[0]
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_corpus_document_version_memberships(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
) -> tuple[CorpusDocumentVersionMembership, ...]:
    """Return a corpus catalogue in canonical document/version order."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")

    try:
        rows = connection.execute(
            """SELECT corpus_id, document_id, document_version_id
            FROM corpus_document_version_memberships
            WHERE corpus_id = ?
            ORDER BY document_id, document_version_id""",
            (corpus_id.as_str(),),
        ).fetchall()
        return _reconstruct_memberships(rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def add_corpus_document_version_membership(
    connection: sqlite3.Connection,
    membership: CorpusDocumentVersionMembership,
) -> None:
    """Add an explicit catalogue membership or accept exact replay."""
    if not isinstance(membership, CorpusDocumentVersionMembership):
        raise TypeError("membership must be a CorpusDocumentVersionMembership.")

    stored = get_corpus_document_version_membership(
        connection,
        membership.corpus_id,
        membership.document_version_id,
    )
    if stored is not None:
        if stored == membership:
            return
        raise StorageConflictError()

    try:
        connection.execute(
            """INSERT INTO corpus_document_version_memberships (
                corpus_id,
                document_id,
                document_version_id
            ) VALUES (?, ?, ?)""",
            (
                membership.corpus_id.as_str(),
                membership.document_id.as_str(),
                membership.document_version_id.as_str(),
            ),
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
