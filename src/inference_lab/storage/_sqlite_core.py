"""Internal SQLite persistence for stable metadata identities."""

import sqlite3

from inference_lab.provenance import (
    CorpusId,
    CorpusMetadata,
    DocumentFamilyId,
    DocumentId,
    DocumentIdentity,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)


def get_corpus(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
) -> CorpusMetadata | None:
    """Reconstruct one named corpus without exposing its storage row."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")

    try:
        row = connection.execute(
            "SELECT corpus_id, display_name FROM corpora WHERE corpus_id = ?",
            (corpus_id.as_str(),),
        ).fetchone()
        if row is None:
            return None
        return CorpusMetadata(
            corpus_id=CorpusId.from_str(row["corpus_id"]),
            display_name=row["display_name"],
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_corpora(connection: sqlite3.Connection) -> tuple[CorpusMetadata, ...]:
    """Return all named corpora in canonical identifier order."""
    try:
        rows = connection.execute(
            "SELECT corpus_id, display_name FROM corpora ORDER BY corpus_id"
        ).fetchall()
        return tuple(
            CorpusMetadata(
                corpus_id=CorpusId.from_str(row["corpus_id"]),
                display_name=row["display_name"],
            )
            for row in rows
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def add_corpus(connection: sqlite3.Connection, corpus: CorpusMetadata) -> None:
    """Insert a named corpus, treating exact replay as a no-op."""
    if not isinstance(corpus, CorpusMetadata):
        raise TypeError("corpus must be a CorpusMetadata.")
    stored = get_corpus(connection, corpus.corpus_id)
    if stored is not None:
        if stored == corpus:
            return
        raise StorageConflictError()

    try:
        connection.execute(
            "INSERT INTO corpora (corpus_id, display_name) VALUES (?, ?)",
            (corpus.corpus_id.as_str(), corpus.display_name),
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def set_corpus_display_name(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    display_name: str,
) -> CorpusMetadata:
    """Replace a corpus display name without changing its stable identity."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    updated = CorpusMetadata(corpus_id=corpus_id, display_name=display_name)
    if get_corpus(connection, corpus_id) is None:
        raise StorageConflictError()

    try:
        connection.execute(
            "UPDATE corpora SET display_name = ? WHERE corpus_id = ?",
            (updated.display_name, updated.corpus_id.as_str()),
        )
        return updated
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


def get_document_family(
    connection: sqlite3.Connection,
    family_id: DocumentFamilyId,
) -> DocumentFamilyId | None:
    """Reconstruct one document-family identity without its storage row."""
    if not isinstance(family_id, DocumentFamilyId):
        raise TypeError("family_id must be a DocumentFamilyId.")

    try:
        row = connection.execute(
            "SELECT family_id FROM document_families WHERE family_id = ?",
            (family_id.as_str(),),
        ).fetchone()
        if row is None:
            return None
        return DocumentFamilyId.from_str(row["family_id"])
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def add_document_family(
    connection: sqlite3.Connection,
    family_id: DocumentFamilyId,
) -> None:
    """Insert a document-family identity, accepting exact replay."""
    if not isinstance(family_id, DocumentFamilyId):
        raise TypeError("family_id must be a DocumentFamilyId.")
    if get_document_family(connection, family_id) is not None:
        return

    try:
        connection.execute(
            "INSERT INTO document_families (family_id) VALUES (?)",
            (family_id.as_str(),),
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
