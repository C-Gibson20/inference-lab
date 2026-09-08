"""Internal SQLite persistence for durable corpus update identities."""

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_corpus_update_memberships import (
    list_corpus_update_document_memberships,
)
from inference_lab.storage._sqlite_corpus_update_reads import (
    _get_update_by_version,
    _reconstruct_component_generations,
    _reconstruct_document_memberships,
    get_corpus_update,
)
from inference_lab.storage._sqlite_corpus_update_support import (
    _CorpusVersionConflictKind,
    _raise_conflict,
    _require_abortable,
    _require_preparing,
    _require_update,
)
from inference_lab.storage._sqlite_corpus_versions import (
    _get_committed_corpus_version,
    get_corpus_version,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage.outcomes import (
    AbortCorpusUpdateOutcome,
    BeginCorpusUpdateOutcome,
    NoCorpusChangeOutcome,
    SealCorpusUpdateOutcome,
)
from inference_lab.storage.protocols import ComponentGenerationValidator
from inference_lab.storage.versioning import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusUpdate,
    CorpusUpdateState,
    CorpusVersion,
    CorpusVersionManifest,
    InvalidCorpusVersionContractError,
    canonicalise_component_kinds,
    validate_corpus_update_identity,
)

CreatedAtFactory = Callable[[], datetime]


def list_corpus_update_component_generations(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> tuple[ComponentGenerationReference, ...]:
    """Return deterministic ready generations staged for one update."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")

    try:
        if get_corpus_update(connection, corpus_id, corpus_version) is None:
            _raise_conflict(corpus_id, corpus_version)
        rows = connection.execute(
            """SELECT component_kind, generation_id, implementation,
                format_version, contract_version, configuration_digest,
                source_manifest_digest, corpus_id, corpus_version, state
            FROM corpus_update_component_generations
            WHERE corpus_id = ? AND corpus_version = ?
            ORDER BY component_kind, generation_id""",
            (corpus_id.as_str(), corpus_version.as_str()),
        ).fetchall()
        return _reconstruct_component_generations(rows)
    except (StorageConflictError, CorruptStorageError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def abort_corpus_update(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> AbortCorpusUpdateOutcome:
    """Abort one preparing or ready update or accept an exact replay."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")

    try:
        update = _require_update(connection, corpus_id, corpus_version)
        if update.state is CorpusUpdateState.ABORTED:
            return AbortCorpusUpdateOutcome(update=update, replayed=True)
        _require_abortable(update)
        connection.execute(
            """UPDATE corpus_updates
            SET state = 'aborted'
            WHERE corpus_id = ? AND corpus_version = ?
                AND state IN ('preparing', 'ready')""",
            (corpus_id.as_str(), corpus_version.as_str()),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            raise RepositoryUnavailableError()
        return AbortCorpusUpdateOutcome(
            update=replace(update, state=CorpusUpdateState.ABORTED),
        )
    except (StorageConflictError, CorruptStorageError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _staged_component_generation(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    component_kind: ComponentKind,
) -> ComponentGenerationReference | None:
    row = connection.execute(
        """SELECT component_kind, generation_id, implementation,
            format_version, contract_version, configuration_digest,
            source_manifest_digest, corpus_id, corpus_version, state
        FROM corpus_update_component_generations
        WHERE corpus_id = ? AND corpus_version = ? AND component_kind = ?""",
        (
            corpus_id.as_str(),
            corpus_version.as_str(),
            component_kind.as_str(),
        ),
    ).fetchone()
    if row is None:
        return None
    return _reconstruct_component_generations((row,))[0]


def _require_generation_identity(
    update: CorpusUpdate,
    generation: ComponentGenerationReference,
) -> None:
    if (
        generation.corpus_id != update.corpus_id
        or generation.corpus_version != update.corpus_version
        or generation.component_kind not in update.required_component_kinds
    ):
        _raise_conflict(update.corpus_id, update.corpus_version)


def _ready_generation(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
    generation: ComponentGenerationReference,
    validator: ComponentGenerationValidator,
) -> ComponentGenerationReference:
    if generation.state is not ComponentGenerationState.PREPARING:
        _raise_conflict(update.corpus_id, update.corpus_version)
    memberships = list_corpus_update_document_memberships(
        connection,
        update.corpus_id,
        update.corpus_version,
    )
    try:
        valid = validator.validate_component_generation(generation, memberships)
    except Exception:
        raise RepositoryUnavailableError() from None
    if type(valid) is not bool:
        raise RepositoryUnavailableError()
    if not valid:
        _raise_conflict(update.corpus_id, update.corpus_version)
    return replace(generation, state=ComponentGenerationState.READY)


def _generation_parameters(
    generation: ComponentGenerationReference,
) -> tuple[str, ...]:
    return (
        generation.corpus_id.as_str(),
        generation.corpus_version.as_str(),
        generation.component_kind.as_str(),
        generation.generation_id,
        generation.implementation,
        generation.format_version,
        generation.contract_version,
        generation.configuration_digest,
        generation.source_manifest_digest,
        generation.state.value,
    )


def register_corpus_update_component_generation(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    generation: ComponentGenerationReference,
    *,
    validator: ComponentGenerationValidator,
) -> ComponentGenerationReference:
    """Validate and register one required immutable component generation."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")
    if not isinstance(generation, ComponentGenerationReference):
        raise TypeError("generation must be a ComponentGenerationReference.")
    if not isinstance(validator, ComponentGenerationValidator):
        raise TypeError("validator must be a ComponentGenerationValidator.")

    try:
        update = _require_update(connection, corpus_id, corpus_version)
        _require_generation_identity(update, generation)
        desired = replace(generation, state=ComponentGenerationState.READY)
        stored = _staged_component_generation(
            connection,
            corpus_id,
            corpus_version,
            generation.component_kind,
        )
        if stored == desired:
            return desired
        if stored is not None:
            _raise_conflict(corpus_id, corpus_version)
        _require_preparing(update)
        ready = _ready_generation(connection, update, generation, validator)
        connection.execute(
            """INSERT INTO corpus_update_component_generations (
                corpus_id, corpus_version, component_kind, generation_id,
                implementation, format_version, contract_version,
                configuration_digest, source_manifest_digest, state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            _generation_parameters(ready),
        )
        return ready
    except (StorageConflictError, CorruptStorageError, RepositoryUnavailableError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def replace_corpus_update_component_generation(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    expected_generation: ComponentGenerationReference,
    replacement_generation: ComponentGenerationReference,
    *,
    validator: ComponentGenerationValidator,
) -> ComponentGenerationReference:
    """Validate and replace one explicitly expected component generation."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")
    for field_name, generation in (
        ("expected_generation", expected_generation),
        ("replacement_generation", replacement_generation),
    ):
        if not isinstance(generation, ComponentGenerationReference):
            raise TypeError(f"{field_name} must be a ComponentGenerationReference.")
    if not isinstance(validator, ComponentGenerationValidator):
        raise TypeError("validator must be a ComponentGenerationValidator.")

    try:
        update = _require_update(connection, corpus_id, corpus_version)
        _require_generation_identity(update, expected_generation)
        _require_generation_identity(update, replacement_generation)
        if expected_generation.component_kind != replacement_generation.component_kind:
            _raise_conflict(corpus_id, corpus_version)
        desired = replace(
            replacement_generation,
            state=ComponentGenerationState.READY,
        )
        stored = _staged_component_generation(
            connection,
            corpus_id,
            corpus_version,
            expected_generation.component_kind,
        )
        if stored == desired:
            return desired
        if stored != expected_generation:
            _raise_conflict(corpus_id, corpus_version)
        _require_preparing(update)
        ready = _ready_generation(
            connection,
            update,
            replacement_generation,
            validator,
        )
        parameters = _generation_parameters(ready)
        connection.execute(
            """UPDATE corpus_update_component_generations
            SET corpus_id = ?, corpus_version = ?, component_kind = ?,
                generation_id = ?, implementation = ?, format_version = ?,
                contract_version = ?, configuration_digest = ?,
                source_manifest_digest = ?, state = ?
            WHERE corpus_id = ? AND corpus_version = ? AND component_kind = ?""",
            (
                *parameters,
                corpus_id.as_str(),
                corpus_version.as_str(),
                expected_generation.component_kind.as_str(),
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            raise RepositoryUnavailableError()
        return ready
    except (StorageConflictError, CorruptStorageError, RepositoryUnavailableError):
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _proposed_manifest(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
    *,
    persisted_seal: bool,
) -> CorpusVersionManifest:
    memberships = list_corpus_update_document_memberships(
        connection,
        update.corpus_id,
        update.corpus_version,
    )
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
        if persisted_seal:
            raise CorruptStorageError() from None
        _raise_conflict(update.corpus_id, update.corpus_version)
    if not complete:
        if persisted_seal:
            raise CorruptStorageError()
        _raise_conflict(update.corpus_id, update.corpus_version)
    return manifest


def _committed_version_manifest(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> tuple[CorpusVersion, CorpusUpdate, CorpusVersionManifest]:
    version = _get_committed_corpus_version(connection, corpus_id, corpus_version)
    committed_update = _require_update(connection, corpus_id, corpus_version)
    if committed_update.state is not CorpusUpdateState.COMMITTED:
        raise CorruptStorageError()
    document_rows = connection.execute(
        """SELECT corpus_id, document_id, document_version_id
        FROM corpus_version_document_memberships
        WHERE corpus_id = ? AND corpus_version = ?
        ORDER BY document_id, document_version_id""",
        (corpus_id.as_str(), corpus_version.as_str()),
    ).fetchall()
    memberships = _reconstruct_document_memberships(document_rows)
    components = list_corpus_update_component_generations(
        connection,
        corpus_id,
        corpus_version,
    )
    try:
        manifest = CorpusVersionManifest(
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            document_memberships=memberships,
            component_generations=components,
        )
        complete = manifest.has_required_components(
            committed_update.required_component_kinds
        )
    except (TypeError, InvalidCorpusVersionContractError):
        raise CorruptStorageError() from None
    if not complete:
        raise CorruptStorageError()
    if manifest.manifest_digest != version.manifest_digest:
        raise CorruptStorageError()
    return version, committed_update, manifest


def _expected_version(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
) -> tuple[CorpusVersion, CorpusVersionManifest] | None:
    expected = update.expected_corpus_version
    if expected is None:
        return None
    version, _, manifest = _committed_version_manifest(
        connection,
        update.corpus_id,
        expected,
    )
    return version, manifest


def _store_update_state(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
    state: CorpusUpdateState,
    manifest_digest: str,
) -> CorpusUpdate:
    connection.execute(
        """UPDATE corpus_updates
        SET state = ?, manifest_digest = ?
        WHERE corpus_id = ? AND corpus_version = ? AND state = 'preparing'""",
        (
            state.value,
            manifest_digest,
            update.corpus_id.as_str(),
            update.corpus_version.as_str(),
        ),
    )
    if connection.execute("SELECT changes()").fetchone()[0] != 1:
        raise RepositoryUnavailableError()
    return replace(update, state=state, manifest_digest=manifest_digest)


def _require_rollback_manifest_equivalence(
    connection: sqlite3.Connection,
    update: CorpusUpdate,
    proposed: CorpusVersionManifest,
) -> None:
    target = update.restores_corpus_version
    if target is None:
        return
    _, target_update, target_manifest = _committed_version_manifest(
        connection,
        update.corpus_id,
        target,
    )
    if (
        update.required_component_kinds != target_update.required_component_kinds
        or not proposed.has_same_semantic_state(target_manifest)
    ):
        _raise_conflict(update.corpus_id, update.corpus_version)


def seal_corpus_update(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> SealCorpusUpdateOutcome | NoCorpusChangeOutcome:
    """Seal a complete manifest or terminate an unchanged proposal."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")

    try:
        update = _require_update(connection, corpus_id, corpus_version)
        persisted_seal = update.manifest_digest is not None
        if update.state in {CorpusUpdateState.READY, CorpusUpdateState.COMMITTED}:
            manifest = _proposed_manifest(
                connection,
                update,
                persisted_seal=True,
            )
            if manifest.manifest_digest != update.manifest_digest:
                raise CorruptStorageError()
            return SealCorpusUpdateOutcome(
                update=update,
                manifest=manifest,
                replayed=True,
            )
        if update.state is CorpusUpdateState.ABORTED and not persisted_seal:
            _raise_conflict(corpus_id, corpus_version)

        manifest = _proposed_manifest(
            connection,
            update,
            persisted_seal=persisted_seal,
        )
        if persisted_seal and manifest.manifest_digest != update.manifest_digest:
            raise CorruptStorageError()
        _require_rollback_manifest_equivalence(connection, update, manifest)
        expected = _expected_version(connection, update)
        if (
            update.restores_corpus_version is None
            and expected is not None
            and manifest.has_same_semantic_state(expected[1])
        ):
            if update.state is CorpusUpdateState.PREPARING:
                update = _store_update_state(
                    connection,
                    update,
                    CorpusUpdateState.ABORTED,
                    manifest.manifest_digest,
                )
            elif update.state is not CorpusUpdateState.ABORTED:
                raise CorruptStorageError()
            return NoCorpusChangeOutcome(
                update=update,
                active_version=expected[0],
                proposed_manifest=manifest,
                active_manifest=expected[1],
                replayed=persisted_seal,
            )
        if update.state is not CorpusUpdateState.PREPARING:
            _raise_conflict(corpus_id, corpus_version)
        ready = _store_update_state(
            connection,
            update,
            CorpusUpdateState.READY,
            manifest.manifest_digest,
        )
        return SealCorpusUpdateOutcome(update=ready, manifest=manifest)
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


def _rollback_target(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
    restores_corpus_version: CorpusVersionId,
) -> tuple[CorpusUpdate, CorpusVersionManifest]:
    if get_corpus_version(connection, corpus_id, restores_corpus_version) is None:
        _raise_conflict(corpus_id, proposed_corpus_version)
    _, target_update, target_manifest = _committed_version_manifest(
        connection,
        corpus_id,
        restores_corpus_version,
    )
    return target_update, target_manifest


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
    """Create and seed an update, deriving rollback requirements from its target."""
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
        rollback_manifest: CorpusVersionManifest | None = None
        if restores_corpus_version is not None:
            target_update, rollback_manifest = _rollback_target(
                connection,
                corpus_id,
                corpus_version,
                restores_corpus_version,
            )
            target_required = target_update.required_component_kinds
            if required and required != target_required:
                _raise_conflict(corpus_id, corpus_version)
            required = target_required

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
                _CorpusVersionConflictKind.UPDATE_PAYLOAD_MISMATCH
                if stored.corpus_id == corpus_id
                else _CorpusVersionConflictKind.VERSION_ID_REUSED
            )
            _raise_conflict(corpus_id, corpus_version, kind)

        active = _active_corpus_version(connection, corpus_id, corpus_version)
        if active != expected_corpus_version:
            _raise_conflict(
                corpus_id,
                corpus_version,
                _CorpusVersionConflictKind.EXPECTED_ACTIVE_MISMATCH,
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
        seed_corpus_version = (
            update.restores_corpus_version
            if rollback_manifest is not None
            else update.expected_corpus_version
        )
        if seed_corpus_version is not None:
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
                    seed_corpus_version.as_str(),
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
