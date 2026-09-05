"""Internal SQLite persistence for durable corpus update identities."""

import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import NoReturn

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_corpus_memberships import (
    get_corpus_document_version_membership,
)
from inference_lab.storage._sqlite_logging import safe_log_failure
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage.outcomes import (
    BeginCorpusUpdateOutcome,
    CorpusVersionConflictKind,
)
from inference_lab.storage.versioning import (
    ComponentKind,
    CorpusUpdate,
    CorpusUpdateState,
    canonicalise_component_kinds,
    validate_corpus_update_identity,
)

CreatedAtFactory = Callable[[], datetime]


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


def _raise_conflict(
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    kind: CorpusVersionConflictKind | None = None,
) -> NoReturn:
    error = StorageConflictError()
    fields: dict[str, object] = {
        "corpus_id": corpus_id.as_str(),
        "corpus_version": corpus_version.as_str(),
    }
    if kind is not None:
        fields["conflict_kind"] = kind.value
    safe_log_failure("corpus_version_conflict", error, **fields)
    raise error


def _require_update(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> CorpusUpdate:
    update = get_corpus_update(connection, corpus_id, corpus_version)
    if update is None:
        _raise_conflict(corpus_id, corpus_version)
    return update


def _require_preparing(update: CorpusUpdate) -> None:
    if update.state is not CorpusUpdateState.PREPARING:
        _raise_conflict(update.corpus_id, update.corpus_version)


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
        _require_preparing(update)
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
        _require_preparing(update)
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
        _require_preparing(update)
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


def _active_corpus_version(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
) -> CorpusVersionId | None:
    row = connection.execute(
        """SELECT corpora.corpus_id AS owned_corpus_id,
            active.corpus_id AS pointer_corpus_id,
            active.corpus_version
        FROM corpora
        LEFT JOIN corpus_active_versions AS active
            ON active.corpus_id = corpora.corpus_id
        WHERE corpora.corpus_id = ?""",
        (corpus_id.as_str(),),
    ).fetchone()
    if row is None:
        _raise_conflict(corpus_id, proposed_corpus_version)
    if row["pointer_corpus_id"] is None:
        raise CorruptStorageError()
    value = row["corpus_version"]
    if value is None:
        return None
    try:
        return CorpusVersionId.from_str(value)
    except (TypeError, ValueError):
        raise CorruptStorageError() from None


def _matches_begin_payload(
    stored: CorpusUpdate,
    *,
    corpus_id: CorpusId,
    expected_corpus_version: CorpusVersionId | None,
    required_component_kinds: tuple[ComponentKind, ...],
    restores_corpus_version: CorpusVersionId | None,
) -> bool:
    return (
        stored.corpus_id == corpus_id
        and stored.expected_corpus_version == expected_corpus_version
        and stored.required_component_kinds == required_component_kinds
        and stored.restores_corpus_version == restores_corpus_version
    )


def _created_at(factory: CreatedAtFactory) -> datetime:
    try:
        value = factory()
    except Exception:
        raise RepositoryUnavailableError() from None
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise RepositoryUnavailableError()
    return value


def begin_corpus_update(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    *,
    expected_corpus_version: CorpusVersionId | None,
    required_component_kinds: tuple[ComponentKind, ...],
    restores_corpus_version: CorpusVersionId | None,
    created_at_factory: CreatedAtFactory,
) -> BeginCorpusUpdateOutcome:
    """Create one preparing update, seed its parent, or accept exact replay."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")
    if expected_corpus_version is not None and not isinstance(
        expected_corpus_version,
        CorpusVersionId,
    ):
        raise TypeError("expected_corpus_version must be a CorpusVersionId or None.")
    if restores_corpus_version is not None and not isinstance(
        restores_corpus_version,
        CorpusVersionId,
    ):
        raise TypeError("restores_corpus_version must be a CorpusVersionId or None.")
    required = canonicalise_component_kinds(required_component_kinds)
    validate_corpus_update_identity(
        corpus_version,
        expected_corpus_version,
        restores_corpus_version,
    )

    try:
        stored = _get_update_by_version(connection, corpus_version)
        if stored is not None:
            if _matches_begin_payload(
                stored,
                corpus_id=corpus_id,
                expected_corpus_version=expected_corpus_version,
                required_component_kinds=required,
                restores_corpus_version=restores_corpus_version,
            ):
                return BeginCorpusUpdateOutcome(update=stored, replayed=True)
            kind = (
                CorpusVersionConflictKind.UPDATE_PAYLOAD_MISMATCH
                if stored.corpus_id == corpus_id
                else CorpusVersionConflictKind.VERSION_ID_REUSED
            )
            _raise_conflict(corpus_id, corpus_version, kind)

        active = _active_corpus_version(connection, corpus_id, corpus_version)
        if active != expected_corpus_version:
            _raise_conflict(
                corpus_id,
                corpus_version,
                CorpusVersionConflictKind.EXPECTED_ACTIVE_MISMATCH,
            )

        update = CorpusUpdate(
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            expected_corpus_version=expected_corpus_version,
            required_component_kinds=required,
            state=CorpusUpdateState.PREPARING,
            created_at=_created_at(created_at_factory),
            restores_corpus_version=restores_corpus_version,
        )
        connection.execute(
            """INSERT INTO corpus_updates (
                corpus_id,
                corpus_version,
                expected_corpus_version,
                state,
                manifest_digest,
                created_at,
                restores_corpus_version
            ) VALUES (?, ?, ?, 'preparing', NULL, ?, ?)""",
            (
                update.corpus_id.as_str(),
                update.corpus_version.as_str(),
                (
                    None
                    if update.expected_corpus_version is None
                    else update.expected_corpus_version.as_str()
                ),
                update.created_at.astimezone(UTC).isoformat(timespec="microseconds"),
                (
                    None
                    if update.restores_corpus_version is None
                    else update.restores_corpus_version.as_str()
                ),
            ),
        )
        connection.executemany(
            """INSERT INTO corpus_update_required_components (
                corpus_id, corpus_version, component_kind
            ) VALUES (?, ?, ?)""",
            (
                (
                    update.corpus_id.as_str(),
                    update.corpus_version.as_str(),
                    kind.as_str(),
                )
                for kind in update.required_component_kinds
            ),
        )
        if update.expected_corpus_version is not None:
            connection.execute(
                """INSERT INTO corpus_update_document_memberships (
                    corpus_id,
                    corpus_version,
                    document_id,
                    document_version_id
                )
                SELECT corpus_id, ?, document_id, document_version_id
                FROM corpus_version_document_memberships
                WHERE corpus_id = ? AND corpus_version = ?""",
                (
                    update.corpus_version.as_str(),
                    update.corpus_id.as_str(),
                    update.expected_corpus_version.as_str(),
                ),
            )
        return BeginCorpusUpdateOutcome(update=update)
    except StorageConflictError:
        raise
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
