"""Internal SQLite persistence for corpus-update component generations."""

import sqlite3
from dataclasses import replace

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_corpus_update_memberships import (
    list_corpus_update_document_memberships,
)
from inference_lab.storage._sqlite_corpus_update_reads import (
    _reconstruct_component_generations,
    get_corpus_update,
)
from inference_lab.storage._sqlite_corpus_update_support import (
    _raise_conflict,
    _require_preparing,
    _require_update,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage.protocols import ComponentGenerationValidator
from inference_lab.storage.versioning import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusUpdate,
)


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
