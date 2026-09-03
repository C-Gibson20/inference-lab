"""Internal SQLite persistence for external-identifier assertions."""

import sqlite3
from collections.abc import Sequence

from inference_lab.provenance import (
    CorpusId,
    DocumentVersionId,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    SourceObservationId,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)


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


def _identifier_values_from_assertions(
    assertions: tuple[ExternalIdentifierAssertion, ...],
) -> dict[str, set[str]]:
    """Group an assertion batch by identifier namespace."""
    values: dict[str, set[str]] = {}
    for assertion in assertions:
        identifier = assertion.external_identifier
        values.setdefault(identifier.namespace, set()).add(identifier.value)
    return values


def _ensure_identifier_values_are_consistent(
    existing: dict[str, set[str]],
    incoming: dict[str, set[str]],
) -> None:
    """Reject different values asserted under a shared namespace."""
    if any(
        existing[namespace] != incoming[namespace]
        for namespace in existing.keys() & incoming.keys()
    ):
        raise StorageConflictError()


def _ensure_advance_identifiers_are_consistent(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
    new_version_id: DocumentVersionId,
) -> None:
    """Reject a source advance that contradicts prior identifier evidence."""
    existing = _identifier_values_for_source(
        connection, corpus_id, source_observation_id
    )
    incoming = _identifier_values_for_version(connection, new_version_id)
    _ensure_identifier_values_are_consistent(existing, incoming)


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


def _validate_assertion_batch(assertions: object) -> None:
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


def _assertion_document_id(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    assertion: ExternalIdentifierAssertion,
) -> str:
    """Return the document associated with an observed source version."""
    row = connection.execute(
        """SELECT document_id
        FROM source_observation_history
        WHERE corpus_id = ? AND source_observation_id = ?
            AND document_version_id = ?""",
        (
            corpus_id.as_str(),
            assertion.source_observation_id.as_str(),
            assertion.document_version_id.as_str(),
        ),
    ).fetchone()
    if row is None:
        raise StorageConflictError()
    return row["document_id"]


def _stored_assertion_keys(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    assertion: ExternalIdentifierAssertion,
) -> set[tuple[str, str]]:
    """Return identifier keys already stored for one observed version."""
    rows = connection.execute(
        """SELECT namespace, identifier_value
        FROM external_identifier_assertions
        WHERE corpus_id = ? AND source_observation_id = ?
            AND document_version_id = ?""",
        (
            corpus_id.as_str(),
            assertion.source_observation_id.as_str(),
            assertion.document_version_id.as_str(),
        ),
    ).fetchall()
    return {(row["namespace"], row["identifier_value"]) for row in rows}


def _new_assertion_rows(
    corpus_id: CorpusId,
    document_id: str,
    assertions: tuple[ExternalIdentifierAssertion, ...],
    incoming: dict[str, set[str]],
    target_base: dict[str, set[str]],
    stored: set[tuple[str, str]],
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    """Return rows not already represented by base or appended evidence."""
    return tuple(
        (
            corpus_id.as_str(),
            assertion.source_observation_id.as_str(),
            assertion.document_version_id.as_str(),
            document_id,
            assertion.external_identifier.namespace,
            assertion.external_identifier.value,
        )
        for assertion in assertions
        if assertion.external_identifier.as_tuple() not in stored
        and target_base.get(assertion.external_identifier.namespace)
        != incoming[assertion.external_identifier.namespace]
    )


def _write_external_identifier_assertions(
    connection: sqlite3.Connection,
    rows: tuple[tuple[str, str, str, str, str, str], ...],
) -> None:
    """Append identifier assertion rows within the caller's transaction."""
    connection.executemany(
        """INSERT INTO external_identifier_assertions (
            corpus_id,
            source_observation_id,
            document_version_id,
            document_id,
            namespace,
            identifier_value
        ) VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )


def _append_external_identifier_assertions(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    assertions: tuple[ExternalIdentifierAssertion, ...],
) -> None:
    """Check and append one non-empty identifier assertion batch."""
    first = assertions[0]
    document_id = _assertion_document_id(connection, corpus_id, first)
    incoming = _identifier_values_from_assertions(assertions)
    existing = _identifier_values_for_source(
        connection,
        corpus_id,
        first.source_observation_id,
    )
    _ensure_identifier_values_are_consistent(existing, incoming)
    target_base = _identifier_values_for_version(
        connection,
        first.document_version_id,
    )
    stored = _stored_assertion_keys(connection, corpus_id, first)
    rows = _new_assertion_rows(
        corpus_id,
        document_id,
        assertions,
        incoming,
        target_base,
        stored,
    )
    _write_external_identifier_assertions(connection, rows)


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
    try:
        _append_external_identifier_assertions(connection, corpus_id, assertions)
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
