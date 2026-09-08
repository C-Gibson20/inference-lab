"""SQLite reads and row reconstruction for durable corpus updates."""

import sqlite3
from collections.abc import Sequence
from datetime import datetime

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import CorruptStorageError, RepositoryUnavailableError
from inference_lab.storage.versioning import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusUpdate,
    CorpusUpdateState,
)


def _required_component_kinds(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> tuple[ComponentKind, ...]:
    rows = connection.execute(
        """SELECT component_kind
        FROM corpus_update_required_components
        WHERE corpus_id = ? AND corpus_version = ?
        ORDER BY component_kind""",
        (corpus_id.as_str(), corpus_version.as_str()),
    ).fetchall()
    return tuple(ComponentKind.from_str(row["component_kind"]) for row in rows)


def _reconstruct_update(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> CorpusUpdate:
    corpus_id = CorpusId.from_str(row["corpus_id"])
    corpus_version = CorpusVersionId.from_str(row["corpus_version"])
    expected = row["expected_corpus_version"]
    restores = row["restores_corpus_version"]
    return CorpusUpdate(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        expected_corpus_version=(
            None if expected is None else CorpusVersionId.from_str(expected)
        ),
        required_component_kinds=_required_component_kinds(
            connection,
            corpus_id,
            corpus_version,
        ),
        state=CorpusUpdateState(row["state"]),
        manifest_digest=row["manifest_digest"],
        created_at=datetime.fromisoformat(row["created_at"]),
        restores_corpus_version=(
            None if restores is None else CorpusVersionId.from_str(restores)
        ),
    )


def _reconstruct_updates(
    connection: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
) -> tuple[CorpusUpdate, ...]:
    try:
        return tuple(_reconstruct_update(connection, row) for row in rows)
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def _get_update_by_version(
    connection: sqlite3.Connection,
    corpus_version: CorpusVersionId,
) -> CorpusUpdate | None:
    row = connection.execute(
        """SELECT corpus_id, corpus_version, expected_corpus_version, state,
            manifest_digest, created_at, restores_corpus_version
        FROM corpus_updates
        WHERE corpus_version = ?""",
        (corpus_version.as_str(),),
    ).fetchone()
    if row is None:
        return None
    return _reconstruct_updates(connection, (row,))[0]


def get_corpus_update(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> CorpusUpdate | None:
    """Return one owned durable update, or ``None`` when absent."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")

    try:
        stored = _get_update_by_version(connection, corpus_version)
        if stored is None or stored.corpus_id != corpus_id:
            return None
        return stored
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_non_terminal_corpus_updates(
    connection: sqlite3.Connection,
) -> tuple[CorpusUpdate, ...]:
    """Return durable preparing and ready work in deterministic order."""
    try:
        rows = connection.execute(
            """SELECT corpus_id, corpus_version, expected_corpus_version, state,
                manifest_digest, created_at, restores_corpus_version
            FROM corpus_updates
            WHERE state IN ('preparing', 'ready')
            ORDER BY created_at, corpus_id, corpus_version"""
        ).fetchall()
        return _reconstruct_updates(connection, rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _reconstruct_document_memberships(
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


def _reconstruct_component_generations(
    rows: Sequence[sqlite3.Row],
) -> tuple[ComponentGenerationReference, ...]:
    try:
        return tuple(
            ComponentGenerationReference(
                component_kind=ComponentKind.from_str(row["component_kind"]),
                generation_id=row["generation_id"],
                implementation=row["implementation"],
                format_version=row["format_version"],
                contract_version=row["contract_version"],
                configuration_digest=row["configuration_digest"],
                source_manifest_digest=row["source_manifest_digest"],
                corpus_id=CorpusId.from_str(row["corpus_id"]),
                corpus_version=CorpusVersionId.from_str(row["corpus_version"]),
                state=ComponentGenerationState(row["state"]),
            )
            for row in rows
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
