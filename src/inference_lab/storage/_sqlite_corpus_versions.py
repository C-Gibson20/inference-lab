"""Internal SQLite reads for immutable committed corpus versions."""

import sqlite3
from collections.abc import Sequence
from datetime import datetime

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import CorruptStorageError, RepositoryUnavailableError
from inference_lab.storage.versioning import (
    CorpusUpdateState,
    CorpusVersion,
    InvalidCorpusVersionContractError,
)


def _reconstruct_corpus_versions(
    rows: Sequence[sqlite3.Row],
) -> tuple[CorpusVersion, ...]:
    try:
        versions = tuple(
            CorpusVersion(
                corpus_id=CorpusId.from_str(row["corpus_id"]),
                corpus_version=CorpusVersionId.from_str(row["corpus_version"]),
                sequence=row["sequence"],
                parent_corpus_version=(
                    None
                    if row["parent_corpus_version"] is None
                    else CorpusVersionId.from_str(row["parent_corpus_version"])
                ),
                manifest_digest=row["manifest_digest"],
                created_at=datetime.fromisoformat(row["created_at"]),
                restores_corpus_version=(
                    None
                    if row["restores_corpus_version"] is None
                    else CorpusVersionId.from_str(row["restores_corpus_version"])
                ),
            )
            for row in rows
            if row["update_state"] == CorpusUpdateState.COMMITTED.value
        )
    except (
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        InvalidCorpusVersionContractError,
    ):
        raise CorruptStorageError() from None
    if len(versions) != len(rows):
        raise CorruptStorageError()
    return versions


def _get_committed_corpus_version(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> CorpusVersion:
    row = connection.execute(
        """SELECT version.corpus_id, version.corpus_version, version.sequence,
            version.parent_corpus_version, version.manifest_digest,
            version.created_at, version.restores_corpus_version,
            update_record.state AS update_state
        FROM corpus_versions AS version
        LEFT JOIN corpus_updates AS update_record
            ON update_record.corpus_id = version.corpus_id
            AND update_record.corpus_version = version.corpus_version
        WHERE version.corpus_id = ? AND version.corpus_version = ?""",
        (corpus_id.as_str(), corpus_version.as_str()),
    ).fetchone()
    if row is None:
        raise CorruptStorageError()
    return _reconstruct_corpus_versions((row,))[0]


def get_corpus_version(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> CorpusVersion | None:
    """Return one committed version owned by a corpus, or ``None``."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")

    try:
        row = connection.execute(
            """SELECT version.corpus_id, version.corpus_version, version.sequence,
                version.parent_corpus_version, version.manifest_digest,
                version.created_at, version.restores_corpus_version,
                update_record.state AS update_state
            FROM corpus_versions AS version
            LEFT JOIN corpus_updates AS update_record
                ON update_record.corpus_id = version.corpus_id
                AND update_record.corpus_version = version.corpus_version
            WHERE version.corpus_version = ?""",
            (corpus_version.as_str(),),
        ).fetchone()
        if row is None or row["corpus_id"] != corpus_id.as_str():
            return None
        return _reconstruct_corpus_versions((row,))[0]
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def get_active_corpus_version(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
) -> CorpusVersion | None:
    """Return a corpus's active committed version, or ``None``."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")

    try:
        row = connection.execute(
            """SELECT corpus.corpus_id AS known_corpus_id,
                active.corpus_id AS pointer_corpus_id,
                active.corpus_version AS active_corpus_version
            FROM corpora AS corpus
            LEFT JOIN corpus_active_versions AS active
                ON active.corpus_id = corpus.corpus_id
            WHERE corpus.corpus_id = ?""",
            (corpus_id.as_str(),),
        ).fetchone()
        if row is None:
            return None
        if row["pointer_corpus_id"] is None:
            raise CorruptStorageError()
        active = row["active_corpus_version"]
        if active is None:
            return None
        try:
            corpus_version = CorpusVersionId.from_str(active)
        except (TypeError, ValueError):
            raise CorruptStorageError() from None
        return _get_committed_corpus_version(connection, corpus_id, corpus_version)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_corpus_versions(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
) -> tuple[CorpusVersion, ...]:
    """Return committed history in ascending contiguous sequence order."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")

    try:
        rows = connection.execute(
            """SELECT version.corpus_id, version.corpus_version, version.sequence,
                version.parent_corpus_version, version.manifest_digest,
                version.created_at, version.restores_corpus_version,
                update_record.state AS update_state
            FROM corpus_versions AS version
            LEFT JOIN corpus_updates AS update_record
                ON update_record.corpus_id = version.corpus_id
                AND update_record.corpus_version = version.corpus_version
            WHERE version.corpus_id = ?
            ORDER BY version.sequence""",
            (corpus_id.as_str(),),
        ).fetchall()
        versions = _reconstruct_corpus_versions(rows)
        if tuple(version.sequence for version in versions) != tuple(
            range(1, len(versions) + 1)
        ):
            raise CorruptStorageError()
        if any(
            version.parent_corpus_version != previous.corpus_version
            for previous, version in zip(versions, versions[1:], strict=False)
        ):
            raise CorruptStorageError()
        return versions
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
