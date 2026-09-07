"""Internal SQLite persistence for private source-locator continuity."""

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    HashAlgorithm,
    SourceObservationId,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage.locator_continuity import (
    SourceFileIdentityHint,
    SourceFingerprint,
    SourceLocatorRevision,
    _validate_source_locator_path,
)

_LOCATOR_COLUMNS = """revision.corpus_id,
    revision.source_observation_id,
    revision.locator_revision,
    revision.locator_path,
    revision.device_id,
    revision.file_id,
    revision.content_hash_algorithm,
    revision.content_hash_digest,
    revision.byte_count,
    revision.modified_time_ns"""


def _reconstruct_revisions(
    rows: Sequence[sqlite3.Row],
) -> tuple[SourceLocatorRevision, ...]:
    """Reconstruct private locator rows without exposing malformed values."""
    try:
        revisions = []
        for row in rows:
            device_id = row["device_id"]
            file_id = row["file_id"]
            if (device_id is None) != (file_id is None):
                raise CorruptStorageError()
            identity_hint = None
            if device_id is not None and file_id is not None:
                identity_hint = SourceFileIdentityHint(
                    device_id=device_id,
                    file_id=file_id,
                )
            revisions.append(
                SourceLocatorRevision(
                    corpus_id=CorpusId.from_str(row["corpus_id"]),
                    source_observation_id=SourceObservationId.from_str(
                        row["source_observation_id"]
                    ),
                    locator_revision=row["locator_revision"],
                    path=Path(row["locator_path"]),
                    identity_hint=identity_hint,
                    fingerprint=SourceFingerprint(
                        content_hash=ContentHash(
                            HashAlgorithm(row["content_hash_algorithm"]),
                            row["content_hash_digest"],
                        ),
                        byte_count=row["byte_count"],
                        modified_time_ns=row["modified_time_ns"],
                    ),
                )
            )
        return tuple(revisions)
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def _validated_path(path: Path) -> str:
    """Validate a lookup path using the durable locator contract."""
    _validate_source_locator_path(path)
    return str(path)


def _query_revisions(
    connection: sqlite3.Connection,
    statement: str,
    parameters: tuple[object, ...],
) -> tuple[SourceLocatorRevision, ...]:
    """Execute and reconstruct one private locator query."""
    try:
        rows = connection.execute(statement, parameters).fetchall()
        return _reconstruct_revisions(rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_source_locator_history(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
) -> tuple[SourceLocatorRevision, ...]:
    """Return one source observation's validated locator history."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(source_observation_id, SourceObservationId):
        raise TypeError("source_observation_id must be a SourceObservationId.")
    revisions = _query_revisions(
        connection,
        f"""SELECT {_LOCATOR_COLUMNS}
        FROM source_locator_revisions AS revision
        WHERE revision.corpus_id = ? AND revision.source_observation_id = ?
        ORDER BY revision.locator_revision""",
        (corpus_id.as_str(), source_observation_id.as_str()),
    )
    if tuple(item.locator_revision for item in revisions) != tuple(
        range(1, len(revisions) + 1)
    ):
        raise CorruptStorageError()
    return revisions


def get_source_locator(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_observation_id: SourceObservationId,
) -> SourceLocatorRevision | None:
    """Return the validated current locator for one observed source."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(source_observation_id, SourceObservationId):
        raise TypeError("source_observation_id must be a SourceObservationId.")
    current = _query_revisions(
        connection,
        f"""SELECT {_LOCATOR_COLUMNS}
        FROM current_source_locators AS current
        JOIN source_locator_revisions AS revision
            ON revision.corpus_id = current.corpus_id
            AND revision.source_observation_id = current.source_observation_id
            AND revision.locator_revision = current.locator_revision
            AND revision.locator_path = current.locator_path
        WHERE current.corpus_id = ? AND current.source_observation_id = ?""",
        (corpus_id.as_str(), source_observation_id.as_str()),
    )
    if not current:
        if list_source_locator_history(
            connection,
            corpus_id,
            source_observation_id,
        ):
            raise CorruptStorageError()
        return None
    if len(current) != 1:
        raise CorruptStorageError()
    history = list_source_locator_history(
        connection,
        corpus_id,
        source_observation_id,
    )
    if not history or history[-1] != current[0]:
        raise CorruptStorageError()
    return current[0]


def find_source_locator_by_path(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    path: Path,
) -> SourceLocatorRevision | None:
    """Return the current exact-locator match within one corpus."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    locator_path = _validated_path(path)
    matches = _query_revisions(
        connection,
        f"""SELECT {_LOCATOR_COLUMNS}
        FROM current_source_locators AS current
        JOIN source_locator_revisions AS revision
            ON revision.corpus_id = current.corpus_id
            AND revision.source_observation_id = current.source_observation_id
            AND revision.locator_revision = current.locator_revision
            AND revision.locator_path = current.locator_path
        WHERE current.corpus_id = ? AND current.locator_path = ?""",
        (corpus_id.as_str(), locator_path),
    )
    if len(matches) > 1:
        raise CorruptStorageError()
    if not matches:
        return None
    match = matches[0]
    if (
        get_source_locator(
            connection,
            corpus_id,
            match.source_observation_id,
        )
        != match
    ):
        raise CorruptStorageError()
    return match


def find_source_locators_by_identity_hint(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    identity_hint: SourceFileIdentityHint,
) -> tuple[SourceLocatorRevision, ...]:
    """Return all current filesystem-hint matches within one corpus."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(identity_hint, SourceFileIdentityHint):
        raise TypeError("identity_hint must be a SourceFileIdentityHint.")
    matches = _query_revisions(
        connection,
        f"""SELECT {_LOCATOR_COLUMNS}
        FROM current_source_locators AS current
        JOIN source_locator_revisions AS revision
            ON revision.corpus_id = current.corpus_id
            AND revision.source_observation_id = current.source_observation_id
            AND revision.locator_revision = current.locator_revision
            AND revision.locator_path = current.locator_path
        WHERE current.corpus_id = ?
            AND revision.device_id = ?
            AND revision.file_id = ?
        ORDER BY revision.source_observation_id""",
        (
            corpus_id.as_str(),
            identity_hint.device_id,
            identity_hint.file_id,
        ),
    )
    for match in matches:
        if (
            get_source_locator(
                connection,
                corpus_id,
                match.source_observation_id,
            )
            != match
        ):
            raise CorruptStorageError()
    return matches


def _revision_values(revision: SourceLocatorRevision) -> tuple[object, ...]:
    """Return private persistence values for one validated revision."""
    identity_hint = revision.identity_hint
    return (
        revision.corpus_id.as_str(),
        revision.source_observation_id.as_str(),
        revision.locator_revision,
        str(revision.path),
        identity_hint.device_id if identity_hint is not None else None,
        identity_hint.file_id if identity_hint is not None else None,
        revision.fingerprint.content_hash.algorithm.value,
        revision.fingerprint.content_hash.digest,
        revision.fingerprint.byte_count,
        revision.fingerprint.modified_time_ns,
    )


def _insert_revision(
    connection: sqlite3.Connection,
    revision: SourceLocatorRevision,
) -> None:
    """Append one locator revision."""
    connection.execute(
        """INSERT INTO source_locator_revisions (
            corpus_id,
            source_observation_id,
            locator_revision,
            locator_path,
            device_id,
            file_id,
            content_hash_algorithm,
            content_hash_digest,
            byte_count,
            modified_time_ns
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        _revision_values(revision),
    )


def add_source_locator(
    connection: sqlite3.Connection,
    revision: SourceLocatorRevision,
) -> None:
    """Create a locator at revision one or accept exact replay."""
    if not isinstance(revision, SourceLocatorRevision):
        raise TypeError("revision must be a SourceLocatorRevision.")
    if revision.locator_revision != 1:
        raise StorageConflictError()
    existing = get_source_locator(
        connection,
        revision.corpus_id,
        revision.source_observation_id,
    )
    if existing == revision:
        return
    if existing is not None:
        raise StorageConflictError()
    try:
        _insert_revision(connection, revision)
        connection.execute(
            """INSERT INTO current_source_locators (
                corpus_id,
                source_observation_id,
                locator_revision,
                locator_path
            ) VALUES (?, ?, ?, ?)""",
            (
                revision.corpus_id.as_str(),
                revision.source_observation_id.as_str(),
                revision.locator_revision,
                str(revision.path),
            ),
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def advance_source_locator(
    connection: sqlite3.Connection,
    revision: SourceLocatorRevision,
) -> SourceLocatorRevision:
    """Append and select the next locator revision or accept exact replay."""
    if not isinstance(revision, SourceLocatorRevision):
        raise TypeError("revision must be a SourceLocatorRevision.")
    current = get_source_locator(
        connection,
        revision.corpus_id,
        revision.source_observation_id,
    )
    if current == revision:
        return revision
    if current is None or revision.locator_revision != current.locator_revision + 1:
        raise StorageConflictError()
    try:
        _insert_revision(connection, revision)
        cursor = connection.execute(
            """UPDATE current_source_locators
            SET locator_revision = ?, locator_path = ?
            WHERE corpus_id = ?
                AND source_observation_id = ?
                AND locator_revision = ?
                AND locator_path = ?""",
            (
                revision.locator_revision,
                str(revision.path),
                revision.corpus_id.as_str(),
                revision.source_observation_id.as_str(),
                current.locator_revision,
                str(current.path),
            ),
        )
        if cursor.rowcount != 1:
            raise StorageConflictError()
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
    return revision
