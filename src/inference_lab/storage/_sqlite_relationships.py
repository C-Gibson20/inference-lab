"""Internal SQLite persistence for provenance relationships."""

import sqlite3
from collections.abc import Sequence

from inference_lab.provenance import (
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
)


def _reconstruct_duplicate_relationships(
    rows: Sequence[sqlite3.Row],
) -> tuple[DuplicateRelationship, ...]:
    try:
        return tuple(
            DuplicateRelationship(
                first_version_id=DocumentVersionId.from_str(row["first_version_id"]),
                second_version_id=DocumentVersionId.from_str(row["second_version_id"]),
                duplicate_class=DuplicateClass(row["duplicate_class"]),
            )
            for row in rows
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def list_duplicate_relationships(
    connection: sqlite3.Connection,
    document_version_id: DocumentVersionId,
) -> tuple[DuplicateRelationship, ...]:
    """Return relationships involving a version in canonical key order."""
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")

    try:
        rows = connection.execute(
            """SELECT duplicate_class, first_version_id, second_version_id
            FROM duplicate_relationships
            WHERE first_version_id = ? OR second_version_id = ?
            ORDER BY duplicate_class, first_version_id, second_version_id""",
            (document_version_id.as_str(), document_version_id.as_str()),
        ).fetchall()
        return _reconstruct_duplicate_relationships(rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def add_duplicate_relationship(
    connection: sqlite3.Connection,
    relationship: DuplicateRelationship,
) -> None:
    """Insert a canonical duplicate relationship or accept exact replay."""
    if not isinstance(relationship, DuplicateRelationship):
        raise TypeError("relationship must be a DuplicateRelationship.")

    try:
        row = connection.execute(
            """SELECT duplicate_class, first_version_id, second_version_id
            FROM duplicate_relationships
            WHERE duplicate_class = ?
                AND first_version_id = ?
                AND second_version_id = ?""",
            relationship.key(),
        ).fetchone()
        if row is not None:
            stored = _reconstruct_duplicate_relationships((row,))[0]
            if stored == relationship:
                return
            raise CorruptStorageError()

        connection.execute(
            """INSERT INTO duplicate_relationships (
                duplicate_class,
                first_version_id,
                second_version_id
            ) VALUES (?, ?, ?)""",
            relationship.key(),
        )
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
