"""SQLite persistence for baseline-ingest manifests and checkpoints."""

import sqlite3
from dataclasses import replace
from typing import Final

from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
    RecordId,
    SourceObservationId,
    SourceSpanId,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.baseline_checkpoint_records import (
    BaselineCheckpointIdentityRecord,
    BaselineCheckpointRecord,
    BaselineCheckpointRecordFailureCategory,
    BaselineCheckpointRecordIdentityKind,
    BaselineCheckpointRecordStage,
    BaselineCheckpointRecordStatus,
    BaselineManifestRecord,
    BaselineManifestRecordState,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage.outcomes import (
    BeginBaselineManifestOutcome,
    TransitionBaselineManifestOutcome,
    WriteBaselineCheckpointOutcome,
)
from inference_lab.storage.versioning import ComponentKind


def _required_component_kinds(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
) -> tuple[ComponentKind, ...]:
    rows = connection.execute(
        """SELECT component_kind
        FROM baseline_manifest_required_components
        WHERE corpus_id = ? AND proposed_corpus_version = ?
        ORDER BY component_kind""",
        (corpus_id.as_str(), proposed_corpus_version.as_str()),
    ).fetchall()
    return tuple(ComponentKind.from_str(row["component_kind"]) for row in rows)


def _reconstruct_manifest(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> BaselineManifestRecord:
    try:
        corpus_id = CorpusId.from_str(row["corpus_id"])
        proposed = CorpusVersionId.from_str(row["proposed_corpus_version"])
        expected = row["expected_corpus_version"]
        return BaselineManifestRecord(
            corpus_id=corpus_id,
            proposed_corpus_version=proposed,
            expected_corpus_version=(
                None if expected is None else CorpusVersionId.from_str(expected)
            ),
            configuration_identity=ConfigurationIdentity(
                algorithm=DigestAlgorithm(row["configuration_algorithm"]),
                digest=row["configuration_digest"],
            ),
            required_component_kinds=_required_component_kinds(
                connection,
                corpus_id,
                proposed,
            ),
            discovery_summary_identity=OutputDigest(
                algorithm=DigestAlgorithm(row["discovery_summary_algorithm"]),
                digest=row["discovery_summary_digest"],
            ),
            state=BaselineManifestRecordState(row["state"]),
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def _get_manifest(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
) -> BaselineManifestRecord | None:
    row = connection.execute(
        """SELECT corpus_id, proposed_corpus_version,
            expected_corpus_version, configuration_algorithm,
            configuration_digest, discovery_summary_algorithm,
            discovery_summary_digest, state
        FROM baseline_manifests
        WHERE corpus_id = ? AND proposed_corpus_version = ?""",
        (corpus_id.as_str(), proposed_corpus_version.as_str()),
    ).fetchone()
    return None if row is None else _reconstruct_manifest(connection, row)


def get_baseline_manifest(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
) -> BaselineManifestRecord | None:
    """Return one owned baseline manifest, or ``None`` when absent."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(proposed_corpus_version, CorpusVersionId):
        raise TypeError("proposed_corpus_version must be a CorpusVersionId.")

    try:
        return _get_manifest(connection, corpus_id, proposed_corpus_version)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _identity_from_row(
    row: sqlite3.Row,
) -> BaselineCheckpointIdentityRecord:
    try:
        kind = BaselineCheckpointRecordIdentityKind(row["identity_kind"])
        value = row["identity_value"]
        if kind is BaselineCheckpointRecordIdentityKind.DOCUMENT:
            DocumentId.from_str(value)
        elif kind is BaselineCheckpointRecordIdentityKind.DOCUMENT_VERSION:
            DocumentVersionId.from_str(value)
        elif kind is BaselineCheckpointRecordIdentityKind.RECORD:
            RecordId.from_str(value)
        elif kind is BaselineCheckpointRecordIdentityKind.SOURCE_OBSERVATION:
            SourceObservationId.from_str(value)
        else:
            SourceSpanId.from_str(value)
        return BaselineCheckpointIdentityRecord(kind=kind, value=value)
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def _reconstruct_checkpoint(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> BaselineCheckpointRecord:
    try:
        checkpoint_storage_id = row["storage_id"]
        identity_rows = connection.execute(
            """SELECT identity_role, ordinal, identity_kind, identity_value
            FROM baseline_checkpoint_identities
            WHERE checkpoint_storage_id = ?
            ORDER BY identity_role, ordinal""",
            (checkpoint_storage_id,),
        ).fetchall()
        if any(
            identity_row["identity_role"] not in {"input", "output"}
            for identity_row in identity_rows
        ):
            raise CorruptStorageError()
        input_identities = tuple(
            _identity_from_row(identity_row)
            for identity_row in identity_rows
            if identity_row["identity_role"] == "input"
        )
        output_identities = tuple(
            _identity_from_row(identity_row)
            for identity_row in identity_rows
            if identity_row["identity_role"] == "output"
        )
        return BaselineCheckpointRecord(
            corpus_id=CorpusId.from_str(row["corpus_id"]),
            proposed_corpus_version=CorpusVersionId.from_str(
                row["proposed_corpus_version"]
            ),
            document_version_id=DocumentVersionId.from_str(row["document_version_id"]),
            document_id=DocumentId.from_str(row["document_id"]),
            stage=BaselineCheckpointRecordStage(row["stage"]),
            configuration_identity=ConfigurationIdentity(
                algorithm=DigestAlgorithm(row["configuration_algorithm"]),
                digest=row["configuration_digest"],
            ),
            input_identities=input_identities,
            output_identities=output_identities,
            status=BaselineCheckpointRecordStatus(row["status"]),
            failure_category=(
                None
                if row["failure_category"] is None
                else BaselineCheckpointRecordFailureCategory(row["failure_category"])
            ),
        )
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def _checkpoint_query(
    connection: sqlite3.Connection,
    *,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
    document_version_id: DocumentVersionId,
    stage: BaselineCheckpointRecordStage,
    configuration_identity: ConfigurationIdentity,
) -> sqlite3.Row | None:
    return connection.execute(
        """SELECT storage_id, corpus_id, proposed_corpus_version,
            document_version_id, document_id, stage,
            configuration_algorithm, configuration_digest,
            status, failure_category
        FROM baseline_checkpoints
        WHERE corpus_id = ? AND proposed_corpus_version = ?
            AND document_version_id = ? AND stage = ?
            AND configuration_algorithm = ? AND configuration_digest = ?""",
        (
            corpus_id.as_str(),
            proposed_corpus_version.as_str(),
            document_version_id.as_str(),
            stage.value,
            configuration_identity.algorithm.value,
            configuration_identity.digest,
        ),
    ).fetchone()


def get_baseline_checkpoint(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
    document_version_id: DocumentVersionId,
    stage: BaselineCheckpointRecordStage,
    configuration_identity: ConfigurationIdentity,
) -> BaselineCheckpointRecord | None:
    """Return one exact baseline checkpoint, or ``None`` when absent."""
    _validate_checkpoint_key(
        corpus_id,
        proposed_corpus_version,
        document_version_id,
        stage,
        configuration_identity,
    )
    try:
        row = _checkpoint_query(
            connection,
            corpus_id=corpus_id,
            proposed_corpus_version=proposed_corpus_version,
            document_version_id=document_version_id,
            stage=stage,
            configuration_identity=configuration_identity,
        )
        return None if row is None else _reconstruct_checkpoint(connection, row)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_baseline_checkpoints(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
) -> tuple[BaselineCheckpointRecord, ...]:
    """Return all checkpoints for one proposal in stable work-key order."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(proposed_corpus_version, CorpusVersionId):
        raise TypeError("proposed_corpus_version must be a CorpusVersionId.")
    try:
        rows = connection.execute(
            """SELECT storage_id, corpus_id, proposed_corpus_version,
                document_version_id, document_id, stage,
                configuration_algorithm, configuration_digest,
                status, failure_category
            FROM baseline_checkpoints
            WHERE corpus_id = ? AND proposed_corpus_version = ?
            ORDER BY document_version_id, stage,
                configuration_algorithm, configuration_digest""",
            (corpus_id.as_str(), proposed_corpus_version.as_str()),
        ).fetchall()
        return tuple(_reconstruct_checkpoint(connection, row) for row in rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _validate_checkpoint_key(
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
    document_version_id: DocumentVersionId,
    stage: BaselineCheckpointRecordStage,
    configuration_identity: ConfigurationIdentity,
) -> None:
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(proposed_corpus_version, CorpusVersionId):
        raise TypeError("proposed_corpus_version must be a CorpusVersionId.")
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")
    if not isinstance(stage, BaselineCheckpointRecordStage):
        raise TypeError("stage must be a BaselineCheckpointRecordStage.")
    if not isinstance(configuration_identity, ConfigurationIdentity):
        raise TypeError("configuration_identity must be a ConfigurationIdentity.")


def _insert_manifest(
    connection: sqlite3.Connection,
    manifest: BaselineManifestRecord,
) -> None:
    connection.execute(
        """INSERT INTO baseline_manifests (
            corpus_id, proposed_corpus_version, expected_corpus_version,
            configuration_algorithm, configuration_digest,
            discovery_summary_algorithm, discovery_summary_digest, state
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            manifest.corpus_id.as_str(),
            manifest.proposed_corpus_version.as_str(),
            (
                None
                if manifest.expected_corpus_version is None
                else manifest.expected_corpus_version.as_str()
            ),
            manifest.configuration_identity.algorithm.value,
            manifest.configuration_identity.digest,
            manifest.discovery_summary_identity.algorithm.value,
            manifest.discovery_summary_identity.digest,
            manifest.state.value,
        ),
    )
    connection.executemany(
        """INSERT INTO baseline_manifest_required_components (
            corpus_id, proposed_corpus_version, component_kind
        ) VALUES (?, ?, ?)""",
        (
            (
                manifest.corpus_id.as_str(),
                manifest.proposed_corpus_version.as_str(),
                component_kind.as_str(),
            )
            for component_kind in manifest.required_component_kinds
        ),
    )


def begin_baseline_manifest(
    connection: sqlite3.Connection,
    manifest: BaselineManifestRecord,
) -> BeginBaselineManifestOutcome:
    """Insert one preparing manifest or return an exact replay."""
    if not isinstance(manifest, BaselineManifestRecord):
        raise TypeError("manifest must be a BaselineManifestRecord.")
    if manifest.state is not BaselineManifestRecordState.PREPARING:
        raise ValueError("a new baseline manifest must be preparing.")

    try:
        stored = _get_manifest(
            connection,
            manifest.corpus_id,
            manifest.proposed_corpus_version,
        )
        if stored is not None:
            if stored == manifest:
                return BeginBaselineManifestOutcome(manifest=stored, replayed=True)
            raise StorageConflictError()
        _insert_manifest(connection, manifest)
        return BeginBaselineManifestOutcome(manifest=manifest)
    except (StorageConflictError, CorruptStorageError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


_MANIFEST_TRANSITIONS: Final = {
    (BaselineManifestRecordState.PREPARING, BaselineManifestRecordState.READY),
    (BaselineManifestRecordState.PREPARING, BaselineManifestRecordState.ABORTED),
    (BaselineManifestRecordState.READY, BaselineManifestRecordState.COMMITTED),
    (BaselineManifestRecordState.READY, BaselineManifestRecordState.ABORTED),
}


def transition_baseline_manifest(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
    *,
    expected_state: BaselineManifestRecordState,
    next_state: BaselineManifestRecordState,
) -> TransitionBaselineManifestOutcome:
    """Compare-and-set one manifest state or return an exact replay."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(proposed_corpus_version, CorpusVersionId):
        raise TypeError("proposed_corpus_version must be a CorpusVersionId.")
    if not isinstance(expected_state, BaselineManifestRecordState):
        raise TypeError("expected_state must be a BaselineManifestRecordState.")
    if not isinstance(next_state, BaselineManifestRecordState):
        raise TypeError("next_state must be a BaselineManifestRecordState.")

    try:
        stored = _get_manifest(connection, corpus_id, proposed_corpus_version)
        if stored is None:
            raise StorageConflictError()
        if stored.state is next_state:
            return TransitionBaselineManifestOutcome(manifest=stored, replayed=True)
        if (
            stored.state is not expected_state
            or (expected_state, next_state) not in _MANIFEST_TRANSITIONS
        ):
            raise StorageConflictError()

        connection.execute(
            """UPDATE baseline_manifests
            SET state = ?
            WHERE corpus_id = ? AND proposed_corpus_version = ?
                AND state = ?""",
            (
                next_state.value,
                corpus_id.as_str(),
                proposed_corpus_version.as_str(),
                expected_state.value,
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            raise RepositoryUnavailableError()
        return TransitionBaselineManifestOutcome(
            manifest=replace(stored, state=next_state),
        )
    except (StorageConflictError, CorruptStorageError, RepositoryUnavailableError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _insert_checkpoint_identities(
    connection: sqlite3.Connection,
    storage_id: int,
    checkpoint: BaselineCheckpointRecord,
) -> None:
    values = tuple(
        (
            storage_id,
            "input",
            ordinal,
            identity.kind.value,
            identity.value,
        )
        for ordinal, identity in enumerate(checkpoint.input_identities)
    ) + tuple(
        (
            storage_id,
            "output",
            ordinal,
            identity.kind.value,
            identity.value,
        )
        for ordinal, identity in enumerate(checkpoint.output_identities)
    )
    connection.executemany(
        """INSERT INTO baseline_checkpoint_identities (
            checkpoint_storage_id, identity_role, ordinal,
            identity_kind, identity_value
        ) VALUES (?, ?, ?, ?, ?)""",
        values,
    )


def _insert_checkpoint(
    connection: sqlite3.Connection,
    checkpoint: BaselineCheckpointRecord,
) -> None:
    cursor = connection.execute(
        """INSERT INTO baseline_checkpoints (
            corpus_id, proposed_corpus_version, document_version_id,
            document_id, stage, configuration_algorithm, configuration_digest,
            status, failure_category
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            checkpoint.corpus_id.as_str(),
            checkpoint.proposed_corpus_version.as_str(),
            checkpoint.document_version_id.as_str(),
            checkpoint.document_id.as_str(),
            checkpoint.stage.value,
            checkpoint.configuration_identity.algorithm.value,
            checkpoint.configuration_identity.digest,
            checkpoint.status.value,
            (
                None
                if checkpoint.failure_category is None
                else checkpoint.failure_category.value
            ),
        ),
    )
    storage_id = cursor.lastrowid
    if not isinstance(storage_id, int):
        raise RepositoryUnavailableError()
    _insert_checkpoint_identities(connection, storage_id, checkpoint)


def _validate_checkpoint_manifest_compatibility(
    connection: sqlite3.Connection,
    checkpoint: BaselineCheckpointRecord,
) -> None:
    """Require a checkpoint to resume the manifest's exact request identity."""
    manifest = _get_manifest(
        connection,
        checkpoint.corpus_id,
        checkpoint.proposed_corpus_version,
    )
    if manifest is None or (
        manifest.configuration_identity != checkpoint.configuration_identity
    ):
        raise StorageConflictError()


def write_baseline_checkpoint(
    connection: sqlite3.Connection,
    checkpoint: BaselineCheckpointRecord,
    *,
    expected_status: BaselineCheckpointRecordStatus | None,
) -> WriteBaselineCheckpointOutcome:
    """Insert or compare-and-set one checkpoint with exact replay."""
    if not isinstance(checkpoint, BaselineCheckpointRecord):
        raise TypeError("checkpoint must be a BaselineCheckpointRecord.")
    if expected_status is not None and not isinstance(
        expected_status,
        BaselineCheckpointRecordStatus,
    ):
        raise TypeError(
            "expected_status must be a BaselineCheckpointRecordStatus or None."
        )

    try:
        _validate_checkpoint_manifest_compatibility(connection, checkpoint)
        row = _checkpoint_query(
            connection,
            corpus_id=checkpoint.corpus_id,
            proposed_corpus_version=checkpoint.proposed_corpus_version,
            document_version_id=checkpoint.document_version_id,
            stage=checkpoint.stage,
            configuration_identity=checkpoint.configuration_identity,
        )
        if row is None:
            if expected_status is not None:
                raise StorageConflictError()
            _insert_checkpoint(connection, checkpoint)
            return WriteBaselineCheckpointOutcome(checkpoint=checkpoint)

        stored = _reconstruct_checkpoint(connection, row)
        if stored == checkpoint:
            return WriteBaselineCheckpointOutcome(
                checkpoint=stored,
                replayed=True,
            )
        if expected_status is None or stored.status is not expected_status:
            raise StorageConflictError()

        connection.execute(
            """UPDATE baseline_checkpoints
            SET document_id = ?, status = ?, failure_category = ?
            WHERE storage_id = ? AND status = ?""",
            (
                checkpoint.document_id.as_str(),
                checkpoint.status.value,
                (
                    None
                    if checkpoint.failure_category is None
                    else checkpoint.failure_category.value
                ),
                row["storage_id"],
                expected_status.value,
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            raise RepositoryUnavailableError()
        connection.execute(
            "DELETE FROM baseline_checkpoint_identities WHERE checkpoint_storage_id = ?",
            (row["storage_id"],),
        )
        _insert_checkpoint_identities(connection, row["storage_id"], checkpoint)
        return WriteBaselineCheckpointOutcome(checkpoint=checkpoint)
    except (StorageConflictError, CorruptStorageError, RepositoryUnavailableError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
