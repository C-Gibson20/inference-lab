"""Internal SQLite persistence for source continuity and identifier assertions."""

import sqlite3
from collections.abc import Sequence

from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_identifier_assertions import (
    _ensure_advance_identifiers_are_consistent,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)


def _reconstruct_states(
    rows: Sequence[sqlite3.Row],
) -> tuple[SourceObservationState, ...]:
    try:
        return tuple(
            SourceObservationState(
                corpus_id=CorpusId.from_str(row["corpus_id"]),
                source_observation_id=SourceObservationId.from_str(
                    row["source_observation_id"]
                ),
                observation_revision=row["observation_revision"],
                document_id=DocumentId.from_str(row["document_id"]),
                document_version_id=DocumentVersionId.from_str(
                    row["document_version_id"]
                ),
            )
            for row in rows
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def list_source_observation_history(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
) -> tuple[SourceObservationState, ...]:
    """Return one source observation's validated append-only history."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(source_observation_id, SourceObservationId):
        raise TypeError("source_observation_id must be a SourceObservationId.")
    try:
        rows = connection.execute(
            """SELECT corpus_id, source_observation_id, observation_revision,
                document_id, document_version_id
            FROM source_observation_history
            WHERE corpus_id = ? AND source_observation_id = ?
            ORDER BY observation_revision""",
            (corpus_id.as_str(), source_observation_id.as_str()),
        ).fetchall()
        states = _reconstruct_states(rows)
        if tuple(state.observation_revision for state in states) != tuple(
            range(1, len(states) + 1)
        ):
            raise CorruptStorageError()
        if len({state.document_id for state in states}) > 1:
            raise CorruptStorageError()
        return states
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def get_source_observation(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
) -> SourceObservationState | None:
    """Return the current validated state for one observed source."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(source_observation_id, SourceObservationId):
        raise TypeError("source_observation_id must be a SourceObservationId.")
    try:
        row = connection.execute(
            """SELECT corpus_id, source_observation_id,
                current_revision AS observation_revision,
                document_id,
                current_document_version_id AS document_version_id
            FROM source_observations
            WHERE corpus_id = ? AND source_observation_id = ?""",
            (corpus_id.as_str(), source_observation_id.as_str()),
        ).fetchone()
        if row is None:
            return None
        current = _reconstruct_states((row,))[0]
        history = list_source_observation_history(
            connection, corpus_id, source_observation_id
        )
        if not history or history[-1] != current:
            raise CorruptStorageError()
        return current
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def add_source_observation(
    connection: sqlite3.Connection,
    state: SourceObservationState,
) -> None:
    """Insert a source observation at revision one or accept exact replay."""
    if not isinstance(state, SourceObservationState):
        raise TypeError("state must be a SourceObservationState.")
    if state.observation_revision != 1:
        raise StorageConflictError()
    existing = get_source_observation(
        connection, state.corpus_id, state.source_observation_id
    )
    if existing == state:
        return
    if existing is not None:
        raise StorageConflictError()

    values = (
        state.corpus_id.as_str(),
        state.source_observation_id.as_str(),
        state.document_id.as_str(),
        state.document_version_id.as_str(),
    )
    try:
        connection.execute(
            """INSERT INTO source_observations (
                corpus_id,
                source_observation_id,
                document_id,
                current_document_version_id,
                current_revision
            ) VALUES (?, ?, ?, ?, 1)""",
            values,
        )
        connection.execute(
            """INSERT INTO source_observation_history (
                corpus_id,
                source_observation_id,
                observation_revision,
                document_id,
                document_version_id
            ) VALUES (?, ?, 1, ?, ?)""",
            values,
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _is_exact_advance_replay(
    history: tuple[SourceObservationState, ...],
    *,
    expected_version_id: DocumentVersionId,
    new_version_id: DocumentVersionId,
) -> bool:
    current = history[-1]
    if current.document_version_id != new_version_id:
        return False
    if expected_version_id == new_version_id:
        return True
    return len(history) >= 2 and history[-2].document_version_id == expected_version_id


def _current_state_for_advance(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
    *,
    expected_version_id: DocumentVersionId,
    new_version_id: DocumentVersionId,
) -> tuple[SourceObservationState, bool]:
    """Return the current state and whether this advance is an exact replay."""
    current = get_source_observation(connection, corpus_id, source_observation_id)
    if current is None:
        raise StorageConflictError()
    history = list_source_observation_history(
        connection, corpus_id, source_observation_id
    )
    if _is_exact_advance_replay(
        history,
        expected_version_id=expected_version_id,
        new_version_id=new_version_id,
    ):
        return current, True
    if current.document_version_id != expected_version_id:
        raise StorageConflictError()
    return current, False


def _write_source_observation_advance(
    connection: sqlite3.Connection,
    current: SourceObservationState,
    next_state: SourceObservationState,
    *,
    expected_version_id: DocumentVersionId,
) -> None:
    """Append history and compare-and-swap the current observation pointer."""
    connection.execute(
        """INSERT INTO source_observation_history (
            corpus_id,
            source_observation_id,
            observation_revision,
            document_id,
            document_version_id
        ) VALUES (?, ?, ?, ?, ?)""",
        (
            next_state.corpus_id.as_str(),
            next_state.source_observation_id.as_str(),
            next_state.observation_revision,
            next_state.document_id.as_str(),
            next_state.document_version_id.as_str(),
        ),
    )
    cursor = connection.execute(
        """UPDATE source_observations
        SET current_document_version_id = ?, current_revision = ?
        WHERE source_observation_id = ?
            AND corpus_id = ?
            AND current_document_version_id = ?
            AND current_revision = ?""",
        (
            next_state.document_version_id.as_str(),
            next_state.observation_revision,
            next_state.source_observation_id.as_str(),
            next_state.corpus_id.as_str(),
            expected_version_id.as_str(),
            current.observation_revision,
        ),
    )
    if cursor.rowcount != 1:
        raise StorageConflictError()


def _advance_source_observation(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
    *,
    expected_version_id: DocumentVersionId,
    new_version_id: DocumentVersionId,
) -> SourceObservationState:
    """Check and persist one source-observation transition."""
    current, is_replay = _current_state_for_advance(
        connection,
        corpus_id,
        source_observation_id,
        expected_version_id=expected_version_id,
        new_version_id=new_version_id,
    )
    if is_replay:
        return current

    _ensure_advance_identifiers_are_consistent(
        connection,
        corpus_id,
        source_observation_id,
        new_version_id,
    )
    next_state = SourceObservationState(
        corpus_id=corpus_id,
        source_observation_id=source_observation_id,
        observation_revision=current.observation_revision + 1,
        document_id=current.document_id,
        document_version_id=new_version_id,
    )
    _write_source_observation_advance(
        connection,
        current,
        next_state,
        expected_version_id=expected_version_id,
    )
    return next_state


def advance_source_observation(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
    *,
    expected_version_id: DocumentVersionId,
    new_version_id: DocumentVersionId,
) -> SourceObservationState:
    """Atomically advance an observed source from its expected version."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(source_observation_id, SourceObservationId):
        raise TypeError("source_observation_id must be a SourceObservationId.")
    if not isinstance(expected_version_id, DocumentVersionId):
        raise TypeError("expected_version_id must be a DocumentVersionId.")
    if not isinstance(new_version_id, DocumentVersionId):
        raise TypeError("new_version_id must be a DocumentVersionId.")

    try:
        return _advance_source_observation(
            connection,
            corpus_id,
            source_observation_id,
            expected_version_id=expected_version_id,
            new_version_id=new_version_id,
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
