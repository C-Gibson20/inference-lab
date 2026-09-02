"""Internal SQLite persistence for source continuity and identifier assertions."""

import sqlite3
from collections.abc import Sequence

from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
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


def _identifier_values_for_source(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
) -> dict[str, set[str]]:
    rows = connection.execute(
        """SELECT identifiers.namespace, identifiers.identifier_value
        FROM source_observation_history AS history
        JOIN external_identifiers AS identifiers
            ON identifiers.document_version_id = history.document_version_id
        WHERE history.corpus_id = ? AND history.source_observation_id = ?
        UNION
        SELECT namespace, identifier_value
        FROM external_identifier_assertions
        WHERE corpus_id = ? AND source_observation_id = ?""",
        (
            corpus_id.as_str(),
            source_observation_id.as_str(),
            corpus_id.as_str(),
            source_observation_id.as_str(),
        ),
    ).fetchall()
    values: dict[str, set[str]] = {}
    for row in rows:
        values.setdefault(row["namespace"], set()).add(row["identifier_value"])
    return values


def _identifier_values_for_version(
    connection: sqlite3.Connection,
    document_version_id: DocumentVersionId,
) -> dict[str, set[str]]:
    rows = connection.execute(
        """SELECT namespace, identifier_value
        FROM external_identifiers
        WHERE document_version_id = ?""",
        (document_version_id.as_str(),),
    ).fetchall()
    values: dict[str, set[str]] = {}
    for row in rows:
        values.setdefault(row["namespace"], set()).add(row["identifier_value"])
    return values


def _ensure_advance_identifiers_are_consistent(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
    new_version_id: DocumentVersionId,
) -> None:
    existing = _identifier_values_for_source(
        connection, corpus_id, source_observation_id
    )
    incoming = _identifier_values_for_version(connection, new_version_id)
    if any(
        existing[namespace] != incoming[namespace]
        for namespace in existing.keys() & incoming.keys()
    ):
        raise StorageConflictError()


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
        return current
    if current.document_version_id != expected_version_id:
        raise StorageConflictError()

    try:
        _ensure_advance_identifiers_are_consistent(
            connection,
            corpus_id,
            source_observation_id,
            new_version_id,
        )
    except StorageConflictError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None

    next_state = SourceObservationState(
        corpus_id=corpus_id,
        source_observation_id=source_observation_id,
        observation_revision=current.observation_revision + 1,
        document_id=current.document_id,
        document_version_id=new_version_id,
    )
    try:
        connection.execute(
            """INSERT INTO source_observation_history (
                corpus_id,
                source_observation_id,
                observation_revision,
                document_id,
                document_version_id
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                corpus_id.as_str(),
                source_observation_id.as_str(),
                next_state.observation_revision,
                current.document_id.as_str(),
                new_version_id.as_str(),
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
                new_version_id.as_str(),
                next_state.observation_revision,
                source_observation_id.as_str(),
                corpus_id.as_str(),
                expected_version_id.as_str(),
                current.observation_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise StorageConflictError()
        return next_state
    except StorageConflictError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _reconstruct_assertions(
    rows: Sequence[sqlite3.Row],
) -> tuple[ExternalIdentifierAssertion, ...]:
    try:
        return tuple(
            ExternalIdentifierAssertion(
                source_observation_id=SourceObservationId.from_str(
                    row["source_observation_id"]
                ),
                document_version_id=DocumentVersionId.from_str(
                    row["document_version_id"]
                ),
                external_identifier=ExternalIdentifier(
                    namespace=row["namespace"],
                    value=row["identifier_value"],
                ),
            )
            for row in rows
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def list_external_identifier_assertions(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
) -> tuple[ExternalIdentifierAssertion, ...]:
    """Return append-only assertions for one source version."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")
    try:
        rows = connection.execute(
            """SELECT source_observation_id, document_version_id,
                namespace, identifier_value
            FROM external_identifier_assertions
            WHERE corpus_id = ? AND document_version_id = ?
            ORDER BY document_version_id, source_observation_id,
                namespace, identifier_value""",
            (corpus_id.as_str(), document_version_id.as_str()),
        ).fetchall()
        return _reconstruct_assertions(rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _validate_assertion_batch(
    assertions: object,
) -> tuple[ExternalIdentifierAssertion, ...]:
    if not isinstance(assertions, tuple):
        raise TypeError("assertions must be a tuple.")
    if any(
        not isinstance(assertion, ExternalIdentifierAssertion)
        for assertion in assertions
    ):
        raise TypeError(
            "assertions must contain only ExternalIdentifierAssertion values."
        )
    keys = tuple(assertion.key() for assertion in assertions)
    if len(keys) != len(set(keys)):
        raise ValueError("assertions must not contain duplicates.")
    if keys != tuple(sorted(keys)):
        raise ValueError("assertions must be in canonical order.")
    associations = {
        (assertion.source_observation_id, assertion.document_version_id)
        for assertion in assertions
    }
    if len(associations) > 1:
        raise ValueError("assertions must describe one source-version association.")
    return assertions


def add_external_identifier_assertions(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    assertions: tuple[ExternalIdentifierAssertion, ...],
) -> None:
    """Append one complete identifier-evidence batch or accept exact replay."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    _validate_assertion_batch(assertions)
    if not assertions:
        return
    first = assertions[0]
    source_id = first.source_observation_id.as_str()
    version_id = first.document_version_id.as_str()
    try:
        association = connection.execute(
            """SELECT document_id
            FROM source_observation_history
            WHERE corpus_id = ? AND source_observation_id = ?
                AND document_version_id = ?""",
            (corpus_id.as_str(), source_id, version_id),
        ).fetchone()
        if association is None:
            raise StorageConflictError()

        incoming: dict[str, set[str]] = {}
        for assertion in assertions:
            identifier = assertion.external_identifier
            incoming.setdefault(identifier.namespace, set()).add(identifier.value)

        existing = _identifier_values_for_source(
            connection,
            corpus_id,
            first.source_observation_id,
        )
        if any(
            existing[namespace] != incoming[namespace]
            for namespace in existing.keys() & incoming.keys()
        ):
            raise StorageConflictError()

        target_base = _identifier_values_for_version(
            connection,
            first.document_version_id,
        )
        stored_rows = connection.execute(
            """SELECT namespace, identifier_value
            FROM external_identifier_assertions
            WHERE corpus_id = ? AND source_observation_id = ?
                AND document_version_id = ?""",
            (corpus_id.as_str(), source_id, version_id),
        ).fetchall()
        stored = {(row["namespace"], row["identifier_value"]) for row in stored_rows}
        connection.executemany(
            """INSERT INTO external_identifier_assertions (
                corpus_id,
                source_observation_id,
                document_version_id,
                document_id,
                namespace,
                identifier_value
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            tuple(
                (
                    corpus_id.as_str(),
                    source_id,
                    version_id,
                    association["document_id"],
                    assertion.external_identifier.namespace,
                    assertion.external_identifier.value,
                )
                for assertion in assertions
                if (
                    assertion.external_identifier.namespace,
                    assertion.external_identifier.value,
                )
                not in stored
                and target_base.get(assertion.external_identifier.namespace)
                != incoming[assertion.external_identifier.namespace]
            ),
        )
    except StorageConflictError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
