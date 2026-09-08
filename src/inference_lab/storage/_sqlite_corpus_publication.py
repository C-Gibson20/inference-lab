"""Internal SQLite atomic corpus-version publication operations."""

import sqlite3
from dataclasses import replace

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_corpus_update_components import (
    list_corpus_update_component_generations,
)
from inference_lab.storage._sqlite_corpus_update_reads import (
    _reconstruct_document_memberships,
)
from inference_lab.storage._sqlite_corpus_update_support import (
    _CorpusVersionConflictKind,
    _raise_conflict,
    _require_update,
)
from inference_lab.storage._sqlite_corpus_updates import (
    CreatedAtFactory,
    _active_corpus_version,
    _created_at,
    _proposed_manifest,
)
from inference_lab.storage._sqlite_corpus_versions import (
    _get_committed_corpus_version,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage.outcomes import (
    CommitCorpusUpdateOutcome,
    RollbackCorpusVersionOutcome,
)
from inference_lab.storage.versioning import (
    CorpusUpdate,
    CorpusUpdateState,
    CorpusVersion,
    CorpusVersionManifest,
    InvalidCorpusVersionContractError,
)


def _published_manifest(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
) -> CorpusVersionManifest:
    rows = connection.execute(
        """SELECT corpus_id, document_id, document_version_id
        FROM corpus_version_document_memberships
        WHERE corpus_id = ? AND corpus_version = ?
        ORDER BY document_id, document_version_id""",
        (update.corpus_id.as_str(), update.corpus_version.as_str()),
    ).fetchall()
    memberships = _reconstruct_document_memberships(rows)
    components = list_corpus_update_component_generations(
        connection,
        update.corpus_id,
        update.corpus_version,
    )
    try:
        manifest = CorpusVersionManifest(
            corpus_id=update.corpus_id,
            corpus_version=update.corpus_version,
            document_memberships=memberships,
            component_generations=components,
        )
        complete = manifest.has_required_components(update.required_component_kinds)
    except (TypeError, InvalidCorpusVersionContractError):
        raise CorruptStorageError() from None
    if not complete or manifest.manifest_digest != update.manifest_digest:
        raise CorruptStorageError()
    return manifest


def _committed_replay(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
) -> CommitCorpusUpdateOutcome:
    version = _get_committed_corpus_version(
        connection,
        update.corpus_id,
        update.corpus_version,
    )
    manifest = _published_manifest(connection, update)
    try:
        return CommitCorpusUpdateOutcome(
            update=update,
            version=version,
            manifest=manifest,
            replayed=True,
        )
    except InvalidCorpusVersionContractError:
        raise CorruptStorageError() from None


def _next_sequence(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
) -> int:
    expected = update.expected_corpus_version
    if expected is None:
        return 1
    return (
        _get_committed_corpus_version(connection, update.corpus_id, expected).sequence
        + 1
    )


def _insert_version(
    connection: sqlite3.Connection,
    version: CorpusVersion,
) -> None:
    connection.execute(
        """INSERT INTO corpus_versions (
            corpus_id, corpus_version, sequence, parent_corpus_version,
            manifest_digest, created_at, restores_corpus_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            version.corpus_id.as_str(),
            version.corpus_version.as_str(),
            version.sequence,
            (
                None
                if version.parent_corpus_version is None
                else version.parent_corpus_version.as_str()
            ),
            version.manifest_digest,
            version.created_at.isoformat(timespec="microseconds"),
            (
                None
                if version.restores_corpus_version is None
                else version.restores_corpus_version.as_str()
            ),
        ),
    )


def _materialise_document_membership(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
) -> None:
    connection.execute(
        """INSERT INTO corpus_version_document_memberships (
            corpus_id, corpus_version, document_id, document_version_id
        )
        SELECT corpus_id, corpus_version, document_id, document_version_id
        FROM corpus_update_document_memberships
        WHERE corpus_id = ? AND corpus_version = ?""",
        (update.corpus_id.as_str(), update.corpus_version.as_str()),
    )


def _advance_active_version(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
) -> None:
    expected = update.expected_corpus_version
    connection.execute(
        """UPDATE corpus_active_versions
        SET corpus_version = ?
        WHERE corpus_id = ? AND corpus_version IS ?""",
        (
            update.corpus_version.as_str(),
            update.corpus_id.as_str(),
            None if expected is None else expected.as_str(),
        ),
    )
    if connection.execute("SELECT changes()").fetchone()[0] != 1:
        _raise_conflict(
            update.corpus_id,
            update.corpus_version,
            _CorpusVersionConflictKind.EXPECTED_ACTIVE_MISMATCH,
        )


def _mark_committed(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
) -> CorpusUpdate:
    connection.execute(
        """UPDATE corpus_updates SET state = 'committed'
        WHERE corpus_id = ? AND corpus_version = ? AND state = 'ready'""",
        (update.corpus_id.as_str(), update.corpus_version.as_str()),
    )
    if connection.execute("SELECT changes()").fetchone()[0] != 1:
        raise RepositoryUnavailableError()
    return replace(update, state=CorpusUpdateState.COMMITTED)


def _commit_corpus_update(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    *,
    created_at_factory: CreatedAtFactory,
    rollback: bool,
) -> CommitCorpusUpdateOutcome:
    """Publish one ready ordinary update or rollback with explicit intent."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")

    try:
        update = _require_update(connection, corpus_id, corpus_version)
        if (update.restores_corpus_version is not None) is not rollback:
            _raise_conflict(corpus_id, corpus_version)
        if update.state is CorpusUpdateState.COMMITTED:
            return _committed_replay(connection, update)
        if update.state is not CorpusUpdateState.READY:
            _raise_conflict(corpus_id, corpus_version)

        manifest = _proposed_manifest(
            connection,
            update,
            persisted_seal=True,
        )
        if manifest.manifest_digest != update.manifest_digest:
            raise CorruptStorageError()
        active = _active_corpus_version(connection, corpus_id, corpus_version)
        if active != update.expected_corpus_version:
            _raise_conflict(
                corpus_id,
                corpus_version,
                _CorpusVersionConflictKind.EXPECTED_ACTIVE_MISMATCH,
            )
        try:
            version = CorpusVersion(
                corpus_id=corpus_id,
                corpus_version=corpus_version,
                sequence=_next_sequence(connection, update),
                parent_corpus_version=update.expected_corpus_version,
                manifest_digest=manifest.manifest_digest,
                created_at=_created_at(created_at_factory),
                restores_corpus_version=update.restores_corpus_version,
            )
        except InvalidCorpusVersionContractError:
            raise CorruptStorageError() from None

        _insert_version(connection, version)
        _materialise_document_membership(connection, update)
        _advance_active_version(connection, update)
        committed = _mark_committed(connection, update)
        return CommitCorpusUpdateOutcome(
            update=committed,
            version=version,
            manifest=manifest,
        )
    except (
        StorageConflictError,
        CorruptStorageError,
        RepositoryUnavailableError,
    ):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def commit_corpus_update(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    *,
    created_at_factory: CreatedAtFactory,
) -> CommitCorpusUpdateOutcome:
    """Atomically publish one ready non-rollback corpus update."""
    return _commit_corpus_update(
        connection,
        corpus_id,
        corpus_version,
        created_at_factory=created_at_factory,
        rollback=False,
    )


def commit_corpus_rollback(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    *,
    created_at_factory: CreatedAtFactory,
) -> RollbackCorpusVersionOutcome:
    """Atomically publish one ready rollback as a new corpus version."""
    publication = _commit_corpus_update(
        connection,
        corpus_id,
        corpus_version,
        created_at_factory=created_at_factory,
        rollback=True,
    )
    try:
        return RollbackCorpusVersionOutcome(publication=publication)
    except InvalidCorpusVersionContractError:
        raise CorruptStorageError() from None
