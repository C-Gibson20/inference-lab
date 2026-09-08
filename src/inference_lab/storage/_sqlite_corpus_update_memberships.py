"""Internal SQLite persistence for corpus-update document memberships."""

import sqlite3

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_corpus_memberships import (
    get_corpus_document_version_membership,
)
from inference_lab.storage._sqlite_corpus_update_reads import (
    _reconstruct_document_memberships,
    get_corpus_update,
)
from inference_lab.storage._sqlite_corpus_update_support import (
    _raise_conflict,
    _require_membership_editable,
    _require_update,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)


def _staged_membership_for_document(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    document_id: DocumentId,
) -> CorpusDocumentVersionMembership | None:
    row = connection.execute(
        """SELECT corpus_id, document_id, document_version_id
        FROM corpus_update_document_memberships
        WHERE corpus_id = ? AND corpus_version = ? AND document_id = ?""",
        (corpus_id.as_str(), corpus_version.as_str(), document_id.as_str()),
    ).fetchone()
    if row is None:
        return None
    return _reconstruct_document_memberships((row,))[0]


def list_corpus_update_document_memberships(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> tuple[CorpusDocumentVersionMembership, ...]:
    """Return deterministic proposed document membership for one update."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")

    try:
        if get_corpus_update(connection, corpus_id, corpus_version) is None:
            _raise_conflict(corpus_id, corpus_version)
        rows = connection.execute(
            """SELECT corpus_id, document_id, document_version_id
            FROM corpus_update_document_memberships
            WHERE corpus_id = ? AND corpus_version = ?
            ORDER BY document_id, document_version_id""",
            (corpus_id.as_str(), corpus_version.as_str()),
        ).fetchall()
        return _reconstruct_document_memberships(rows)
    except (StorageConflictError, CorruptStorageError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _require_catalogue_membership(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    membership: CorpusDocumentVersionMembership,
) -> None:
    stored = get_corpus_document_version_membership(
        connection,
        corpus_id,
        membership.document_version_id,
    )
    if stored != membership:
        _raise_conflict(corpus_id, corpus_version)


def add_corpus_update_document_membership(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    membership: CorpusDocumentVersionMembership,
) -> None:
    """Add one catalogue member to a preparing update or accept replay."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")
    if not isinstance(membership, CorpusDocumentVersionMembership):
        raise TypeError("membership must be a CorpusDocumentVersionMembership.")
    if membership.corpus_id != corpus_id:
        _raise_conflict(corpus_id, corpus_version)

    try:
        update = _require_update(connection, corpus_id, corpus_version)
        _require_catalogue_membership(
            connection,
            corpus_id,
            corpus_version,
            membership,
        )
        staged = _staged_membership_for_document(
            connection,
            corpus_id,
            corpus_version,
            membership.document_id,
        )
        if staged == membership:
            return
        if staged is not None:
            _raise_conflict(corpus_id, corpus_version)
        _require_membership_editable(update)
        connection.execute(
            """INSERT INTO corpus_update_document_memberships (
                corpus_id, corpus_version, document_id, document_version_id
            ) VALUES (?, ?, ?, ?)""",
            (
                corpus_id.as_str(),
                corpus_version.as_str(),
                membership.document_id.as_str(),
                membership.document_version_id.as_str(),
            ),
        )
    except (StorageConflictError, CorruptStorageError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def replace_corpus_update_document_membership(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    expected_membership: CorpusDocumentVersionMembership,
    replacement_membership: CorpusDocumentVersionMembership,
) -> None:
    """Replace one explicitly agreeing logical-document membership."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")
    for field_name, membership in (
        ("expected_membership", expected_membership),
        ("replacement_membership", replacement_membership),
    ):
        if not isinstance(membership, CorpusDocumentVersionMembership):
            raise TypeError(f"{field_name} must be a CorpusDocumentVersionMembership.")
    if (
        expected_membership.corpus_id != corpus_id
        or replacement_membership.corpus_id != corpus_id
        or expected_membership.document_id != replacement_membership.document_id
    ):
        _raise_conflict(corpus_id, corpus_version)

    try:
        update = _require_update(connection, corpus_id, corpus_version)
        _require_catalogue_membership(
            connection,
            corpus_id,
            corpus_version,
            replacement_membership,
        )
        staged = _staged_membership_for_document(
            connection,
            corpus_id,
            corpus_version,
            expected_membership.document_id,
        )
        if staged == replacement_membership:
            return
        if staged != expected_membership:
            _raise_conflict(corpus_id, corpus_version)
        _require_membership_editable(update)
        connection.execute(
            """UPDATE corpus_update_document_memberships
            SET document_version_id = ?
            WHERE corpus_id = ? AND corpus_version = ?
                AND document_id = ? AND document_version_id = ?""",
            (
                replacement_membership.document_version_id.as_str(),
                corpus_id.as_str(),
                corpus_version.as_str(),
                expected_membership.document_id.as_str(),
                expected_membership.document_version_id.as_str(),
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            raise RepositoryUnavailableError()
    except (StorageConflictError, CorruptStorageError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def remove_corpus_update_document_membership(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    membership: CorpusDocumentVersionMembership,
) -> None:
    """Remove one exact staged membership or accept an absent replay."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")
    if not isinstance(membership, CorpusDocumentVersionMembership):
        raise TypeError("membership must be a CorpusDocumentVersionMembership.")
    if membership.corpus_id != corpus_id:
        _raise_conflict(corpus_id, corpus_version)

    try:
        update = _require_update(connection, corpus_id, corpus_version)
        staged = _staged_membership_for_document(
            connection,
            corpus_id,
            corpus_version,
            membership.document_id,
        )
        if staged is None:
            return
        if staged != membership:
            _raise_conflict(corpus_id, corpus_version)
        _require_membership_editable(update)
        connection.execute(
            """DELETE FROM corpus_update_document_memberships
            WHERE corpus_id = ? AND corpus_version = ?
                AND document_id = ? AND document_version_id = ?""",
            (
                corpus_id.as_str(),
                corpus_version.as_str(),
                membership.document_id.as_str(),
                membership.document_version_id.as_str(),
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            raise RepositoryUnavailableError()
    except (StorageConflictError, CorruptStorageError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
