"""Internal SQLite persistence for corpus and logical-document identities."""

import sqlite3

from inference_lab.provenance import CorpusId, DocumentId, DocumentIdentity
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
)


def get_corpus(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
) -> CorpusId | None:
    """Reconstruct one corpus identity without exposing its storage row."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")

    try:
        row = connection.execute(
            "SELECT corpus_id FROM corpora WHERE corpus_id = ?",
            (corpus_id.as_str(),),
        ).fetchone()
        if row is None:
            return None
        return CorpusId.from_str(row["corpus_id"])
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def add_corpus(connection: sqlite3.Connection, corpus_id: CorpusId) -> None:
    """Insert a corpus identity, treating exact replay as a no-op."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if get_corpus(connection, corpus_id) is not None:
        return

    try:
        connection.execute(
            "INSERT INTO corpora (corpus_id) VALUES (?)",
            (corpus_id.as_str(),),
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def get_document(
    connection: sqlite3.Connection,
    document_id: DocumentId,
) -> DocumentIdentity | None:
    """Reconstruct one logical document without exposing its storage row."""
    if not isinstance(document_id, DocumentId):
        raise TypeError("document_id must be a DocumentId.")

    try:
        row = connection.execute(
            "SELECT document_id FROM documents WHERE document_id = ?",
            (document_id.as_str(),),
        ).fetchone()
        if row is None:
            return None
        return DocumentIdentity(DocumentId.from_str(row["document_id"]))
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def add_document(
    connection: sqlite3.Connection,
    document: DocumentIdentity,
) -> None:
    """Insert a logical document, treating exact replay as a no-op."""
    if not isinstance(document, DocumentIdentity):
        raise TypeError("document must be a DocumentIdentity.")
    if get_document(connection, document.document_id) is not None:
        return

    try:
        connection.execute(
            "INSERT INTO documents (document_id) VALUES (?)",
            (document.document_id.as_str(),),
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
