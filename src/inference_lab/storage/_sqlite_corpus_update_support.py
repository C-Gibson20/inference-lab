"""Shared SQLite corpus-update conflict and state preconditions."""

import sqlite3
from enum import StrEnum
from typing import NoReturn

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage._sqlite_corpus_update_reads import get_corpus_update
from inference_lab.storage._sqlite_logging import safe_log_failure
from inference_lab.storage.errors import StorageConflictError
from inference_lab.storage.versioning import CorpusUpdate, CorpusUpdateState


class _CorpusVersionConflictKind(StrEnum):
    """Bounded operational reasons why a corpus proposal cannot advance."""

    EXPECTED_ACTIVE_MISMATCH = "expected_active_mismatch"
    VERSION_ID_REUSED = "version_id_reused"
    UPDATE_PAYLOAD_MISMATCH = "update_payload_mismatch"


def _raise_conflict(
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    kind: _CorpusVersionConflictKind | None = None,
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


def _require_membership_editable(update: CorpusUpdate) -> None:
    _require_preparing(update)
    if update.restores_corpus_version is not None:
        _raise_conflict(update.corpus_id, update.corpus_version)


def _require_abortable(update: CorpusUpdate) -> None:
    if update.state not in {CorpusUpdateState.PREPARING, CorpusUpdateState.READY}:
        _raise_conflict(update.corpus_id, update.corpus_version)
