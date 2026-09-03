"""Internal SQLite persistence for provenance relationships."""

import sqlite3
from collections.abc import Sequence

from inference_lab.provenance import (
    CorpusId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)

_MEMBERSHIPS_BY_FAMILY = """SELECT family_id, document_id, document_version_id
FROM document_family_memberships AS family_membership
WHERE family_id = ? AND (
    (
        document_version_id IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM corpus_document_version_memberships AS corpus_membership
            WHERE corpus_membership.corpus_id = ?
                AND corpus_membership.document_version_id =
                    family_membership.document_version_id
        )
    ) OR (
        document_version_id IS NULL
        AND EXISTS (
            SELECT 1 FROM corpus_document_version_memberships AS corpus_membership
            WHERE corpus_membership.corpus_id = ?
                AND corpus_membership.document_id = family_membership.document_id
        )
    )
)
ORDER BY family_id, document_id, document_version_id"""
_MEMBERSHIPS_BY_DOCUMENT = """SELECT family_id, document_id, document_version_id
FROM document_family_memberships AS family_membership
WHERE document_id = ? AND (
    (
        document_version_id IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM corpus_document_version_memberships AS corpus_membership
            WHERE corpus_membership.corpus_id = ?
                AND corpus_membership.document_version_id =
                    family_membership.document_version_id
        )
    ) OR (
        document_version_id IS NULL
        AND EXISTS (
            SELECT 1 FROM corpus_document_version_memberships AS corpus_membership
            WHERE corpus_membership.corpus_id = ?
                AND corpus_membership.document_id = family_membership.document_id
        )
    )
)
ORDER BY family_id, document_id, document_version_id"""
_MEMBERSHIPS_BY_VERSION = """SELECT family_id, document_id, document_version_id
FROM document_family_memberships AS family_membership
WHERE document_version_id = ?
    AND EXISTS (
        SELECT 1 FROM corpus_document_version_memberships AS corpus_membership
        WHERE corpus_membership.corpus_id = ?
            AND corpus_membership.document_version_id =
                family_membership.document_version_id
    )
ORDER BY family_id, document_id, document_version_id"""


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
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
) -> tuple[DuplicateRelationship, ...]:
    """Return relationships involving a version in canonical key order."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")

    try:
        rows = connection.execute(
            """SELECT duplicate_class, first_version_id, second_version_id
            FROM duplicate_relationships AS relationship
            WHERE (
                relationship.first_version_id = ?
                OR relationship.second_version_id = ?
            )
                AND EXISTS (
                    SELECT 1 FROM corpus_document_version_memberships AS first
                    WHERE first.corpus_id = ?
                        AND first.document_version_id =
                            relationship.first_version_id
                )
                AND EXISTS (
                    SELECT 1 FROM corpus_document_version_memberships AS second
                    WHERE second.corpus_id = ?
                        AND second.document_version_id =
                            relationship.second_version_id
                )
            ORDER BY relationship.duplicate_class,
                relationship.first_version_id,
                relationship.second_version_id""",
            (
                document_version_id.as_str(),
                document_version_id.as_str(),
                corpus_id.as_str(),
                corpus_id.as_str(),
            ),
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
        endpoints = connection.execute(
            """SELECT
                first.content_hash_algorithm AS first_algorithm,
                first.content_hash_digest AS first_digest,
                second.content_hash_algorithm AS second_algorithm,
                second.content_hash_digest AS second_digest
            FROM document_versions AS first
            JOIN document_versions AS second
                ON second.document_version_id = ?
            WHERE first.document_version_id = ?""",
            (
                relationship.second_version_id.as_str(),
                relationship.first_version_id.as_str(),
            ),
        ).fetchone()
        if endpoints is None:
            raise StorageConflictError()
        if (
            endpoints["first_algorithm"],
            endpoints["first_digest"],
        ) != (
            endpoints["second_algorithm"],
            endpoints["second_digest"],
        ):
            raise StorageConflictError()

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
    except (CorruptStorageError, StorageConflictError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _reconstruct_family_memberships(
    rows: Sequence[sqlite3.Row],
) -> tuple[DocumentFamilyMembership, ...]:
    try:
        return tuple(
            DocumentFamilyMembership(
                family_id=DocumentFamilyId.from_str(row["family_id"]),
                document_id=DocumentId.from_str(row["document_id"]),
                document_version_id=(
                    DocumentVersionId.from_str(row["document_version_id"])
                    if row["document_version_id"] is not None
                    else None
                ),
            )
            for row in rows
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def _list_family_memberships(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    statement: str,
    parameter: str,
    corpus_parameter_count: int,
) -> tuple[DocumentFamilyMembership, ...]:
    try:
        parameters = (parameter,) + (corpus_id.as_str(),) * corpus_parameter_count
        rows = connection.execute(statement, parameters).fetchall()
        return _reconstruct_family_memberships(rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_family_memberships_by_family(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    family_id: DocumentFamilyId,
) -> tuple[DocumentFamilyMembership, ...]:
    """Return a family's memberships in canonical key order."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(family_id, DocumentFamilyId):
        raise TypeError("family_id must be a DocumentFamilyId.")
    return _list_family_memberships(
        connection,
        corpus_id,
        _MEMBERSHIPS_BY_FAMILY,
        family_id.as_str(),
        2,
    )


def list_family_memberships_by_document(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    document_id: DocumentId,
) -> tuple[DocumentFamilyMembership, ...]:
    """Return a document's memberships in canonical key order."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(document_id, DocumentId):
        raise TypeError("document_id must be a DocumentId.")
    return _list_family_memberships(
        connection,
        corpus_id,
        _MEMBERSHIPS_BY_DOCUMENT,
        document_id.as_str(),
        2,
    )


def list_family_memberships_by_document_version(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
) -> tuple[DocumentFamilyMembership, ...]:
    """Return a version's memberships in canonical key order."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")
    return _list_family_memberships(
        connection,
        corpus_id,
        _MEMBERSHIPS_BY_VERSION,
        document_version_id.as_str(),
        1,
    )


def add_family_membership(
    connection: sqlite3.Connection,
    membership: DocumentFamilyMembership,
) -> None:
    """Insert a family membership or accept exact replay."""
    if not isinstance(membership, DocumentFamilyMembership):
        raise TypeError("membership must be a DocumentFamilyMembership.")

    try:
        row = connection.execute(
            """SELECT family_id, document_id, document_version_id
            FROM document_family_memberships
            WHERE family_id = ?
                AND document_id = ?
                AND document_version_id IS ?""",
            membership.key(),
        ).fetchone()
        if row is not None:
            stored = _reconstruct_family_memberships((row,))[0]
            if stored == membership:
                return
            raise CorruptStorageError()

        connection.execute(
            """INSERT INTO document_family_memberships (
                family_id,
                document_id,
                document_version_id
            ) VALUES (?, ?, ?)""",
            membership.key(),
        )
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
