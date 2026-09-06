"""Tests for typed corpus-version operation outcomes."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.storage import (
    AbortCorpusUpdateOutcome,
    BeginCorpusUpdateOutcome,
    CommitCorpusUpdateOutcome,
    ComponentKind,
    CorpusUpdate,
    CorpusUpdateState,
    CorpusVersion,
    CorpusVersionConflictKind,
    CorpusVersionConflictOutcome,
    CorpusVersionManifest,
    InvalidCorpusVersionContractError,
    NoCorpusChangeOutcome,
    RollbackCorpusVersionOutcome,
    SealCorpusUpdateOutcome,
)

CREATED_AT = datetime(2026, 9, 5, 15, 0, tzinfo=UTC)
CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
OTHER_CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000002")
PROPOSED_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000003")
ACTIVE_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000004")
OTHER_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000005")
RESTORE_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000006")
MEMBERSHIP = CorpusDocumentVersionMembership(
    CORPUS_ID,
    DocumentId.from_str("00000000-0000-4000-8000-000000000007"),
    DocumentVersionId.from_str("00000000-0000-4000-8000-000000000008"),
)


def _manifest(
    *,
    corpus_id: CorpusId = CORPUS_ID,
    corpus_version: CorpusVersionId = PROPOSED_ID,
    memberships: tuple[CorpusDocumentVersionMembership, ...] = (),
) -> CorpusVersionManifest:
    return CorpusVersionManifest(corpus_id, corpus_version, memberships)


def _update(
    *,
    state: CorpusUpdateState = CorpusUpdateState.READY,
    manifest: CorpusVersionManifest | None = None,
    expected: CorpusVersionId | None = None,
    restores: CorpusVersionId | None = None,
) -> CorpusUpdate:
    selected_manifest = manifest or _manifest()
    return CorpusUpdate(
        corpus_id=CORPUS_ID,
        corpus_version=PROPOSED_ID,
        expected_corpus_version=expected,
        required_component_kinds=(),
        state=state,
        created_at=CREATED_AT,
        manifest_digest=(
            None
            if state is CorpusUpdateState.PREPARING
            else selected_manifest.manifest_digest
        ),
        restores_corpus_version=restores,
    )


def _version(
    manifest: CorpusVersionManifest,
    *,
    corpus_id: CorpusId = CORPUS_ID,
    corpus_version: CorpusVersionId = PROPOSED_ID,
    sequence: int = 1,
    parent: CorpusVersionId | None = None,
    restores: CorpusVersionId | None = None,
) -> CorpusVersion:
    return CorpusVersion(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        sequence=sequence,
        parent_corpus_version=parent,
        manifest_digest=manifest.manifest_digest,
        created_at=CREATED_AT,
        restores_corpus_version=restores,
    )


def _commit(
    *,
    expected: CorpusVersionId | None = None,
    restores: CorpusVersionId | None = None,
    replayed: bool = False,
) -> CommitCorpusUpdateOutcome:
    manifest = _manifest()
    update = _update(
        state=CorpusUpdateState.COMMITTED,
        manifest=manifest,
        expected=expected,
        restores=restores,
    )
    return CommitCorpusUpdateOutcome(
        update=update,
        version=_version(
            manifest,
            sequence=1 if expected is None else 3,
            parent=expected,
            restores=restores,
        ),
        manifest=manifest,
        replayed=replayed,
    )


def test_begin_outcome_supports_new_begin_and_later_exact_replay() -> None:
    preparing = _update(state=CorpusUpdateState.PREPARING)
    committed = _update(state=CorpusUpdateState.COMMITTED)

    outcome = BeginCorpusUpdateOutcome(preparing)

    assert outcome.update == preparing
    assert not outcome.replayed
    assert BeginCorpusUpdateOutcome(committed, replayed=True).replayed
    with pytest.raises(FrozenInstanceError):
        outcome.replayed = True  # type: ignore[misc]
    with pytest.raises(InvalidCorpusVersionContractError, match="new begin"):
        BeginCorpusUpdateOutcome(committed)


@pytest.mark.parametrize(
    ("constructor", "values", "message"),
    [
        (BeginCorpusUpdateOutcome, {"update": None}, "update"),
        (
            BeginCorpusUpdateOutcome,
            {"update": _update(state=CorpusUpdateState.PREPARING), "replayed": 1},
            "replayed",
        ),
        (SealCorpusUpdateOutcome, {"update": None, "manifest": _manifest()}, "update"),
        (
            SealCorpusUpdateOutcome,
            {"update": _update(), "manifest": None},
            "manifest",
        ),
        (
            CommitCorpusUpdateOutcome,
            {"update": None, "version": None, "manifest": None},
            "update",
        ),
        (AbortCorpusUpdateOutcome, {"update": None}, "update"),
    ],
)
def test_basic_outcomes_reject_wrong_types(
    constructor: type[object],
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        constructor(**values)


def test_seal_outcome_accepts_ready_and_committed_exact_replay() -> None:
    manifest = _manifest()
    ready = SealCorpusUpdateOutcome(_update(manifest=manifest), manifest)
    replay = SealCorpusUpdateOutcome(
        _update(state=CorpusUpdateState.COMMITTED, manifest=manifest),
        manifest,
        replayed=True,
    )

    assert ready.manifest == manifest
    assert replay.replayed


@pytest.mark.parametrize(
    ("state", "replayed"),
    [
        (CorpusUpdateState.PREPARING, False),
        (CorpusUpdateState.COMMITTED, False),
        (CorpusUpdateState.ABORTED, True),
    ],
)
def test_seal_outcome_rejects_invalid_lifecycle(
    state: CorpusUpdateState,
    replayed: bool,
) -> None:
    manifest = _manifest()

    with pytest.raises(InvalidCorpusVersionContractError, match="seal outcome"):
        SealCorpusUpdateOutcome(
            _update(state=state, manifest=manifest),
            manifest,
            replayed,
        )


@pytest.mark.parametrize(
    "manifest",
    [
        _manifest(corpus_id=OTHER_CORPUS_ID),
        _manifest(corpus_version=OTHER_VERSION_ID),
    ],
)
def test_seal_outcome_rejects_foreign_manifest(
    manifest: CorpusVersionManifest,
) -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="outcome manifest"):
        SealCorpusUpdateOutcome(_update(), manifest)


def test_seal_outcome_rejects_digest_or_required_component_mismatch() -> None:
    manifest = _manifest()
    mismatched_digest = replace(_update(), manifest_digest="f" * 64)
    missing_component = replace(
        _update(),
        required_component_kinds=(ComponentKind("metadata"),),
    )

    with pytest.raises(InvalidCorpusVersionContractError, match="digest"):
        SealCorpusUpdateOutcome(mismatched_digest, manifest)
    with pytest.raises(InvalidCorpusVersionContractError, match="required components"):
        SealCorpusUpdateOutcome(missing_component, manifest)


def test_commit_outcome_validates_first_publication_and_exact_replay() -> None:
    first = _commit()
    replay = _commit(replayed=True)

    assert first.version.sequence == 1
    assert first.version.parent_corpus_version is None
    assert replay.replayed


@pytest.mark.parametrize("field", ["version", "manifest"])
def test_commit_outcome_rejects_wrong_result_type(field: str) -> None:
    outcome = _commit()

    with pytest.raises(TypeError, match=field):
        replace(outcome, **{field: None})


def test_commit_outcome_rejects_lifecycle_and_version_mismatches() -> None:
    outcome = _commit()
    other_manifest = _manifest(corpus_id=OTHER_CORPUS_ID)

    with pytest.raises(InvalidCorpusVersionContractError, match="committed update"):
        replace(outcome, update=_update())
    with pytest.raises(InvalidCorpusVersionContractError, match="committed version"):
        replace(
            outcome,
            version=_version(
                other_manifest,
                corpus_id=OTHER_CORPUS_ID,
            ),
        )


def test_commit_outcome_requires_parent_and_restore_intent_alignment() -> None:
    first = _commit()

    with pytest.raises(InvalidCorpusVersionContractError, match="parent"):
        replace(
            first,
            update=replace(first.update, expected_corpus_version=ACTIVE_ID),
        )
    with pytest.raises(InvalidCorpusVersionContractError, match="restore target"):
        replace(
            _commit(expected=ACTIVE_ID),
            update=replace(
                _commit(expected=ACTIVE_ID).update,
                restores_corpus_version=RESTORE_ID,
            ),
        )


def test_abort_outcome_requires_terminal_aborted_update_and_strict_replay() -> None:
    aborted = _update(state=CorpusUpdateState.ABORTED)

    assert AbortCorpusUpdateOutcome(aborted, replayed=True).replayed
    with pytest.raises(InvalidCorpusVersionContractError, match="aborted update"):
        AbortCorpusUpdateOutcome(_update())
    with pytest.raises(TypeError, match="replayed"):
        AbortCorpusUpdateOutcome(aborted, replayed=1)  # type: ignore[arg-type]


def test_no_change_outcome_requires_equal_terminal_proposal() -> None:
    manifest = _manifest()
    update = _update(
        state=CorpusUpdateState.ABORTED,
        manifest=manifest,
        expected=ACTIVE_ID,
    )
    active_manifest = _manifest(corpus_version=ACTIVE_ID)
    active = _version(
        active_manifest,
        corpus_version=ACTIVE_ID,
        sequence=2,
        parent=OTHER_VERSION_ID,
    )

    outcome = NoCorpusChangeOutcome(update, active, active_manifest, manifest)

    assert outcome.active_version == active
    assert outcome.active_manifest == active_manifest
    assert outcome.proposed_manifest.manifest_digest == active.manifest_digest


@pytest.mark.parametrize(
    "field",
    ["update", "active_version", "proposed_manifest", "active_manifest"],
)
def test_no_change_outcome_rejects_wrong_types(field: str) -> None:
    manifest = _manifest()
    update = _update(
        state=CorpusUpdateState.ABORTED,
        manifest=manifest,
        expected=ACTIVE_ID,
    )
    active_manifest = _manifest(corpus_version=ACTIVE_ID)
    outcome = NoCorpusChangeOutcome(
        update,
        _version(
            active_manifest,
            corpus_version=ACTIVE_ID,
            sequence=2,
            parent=OTHER_VERSION_ID,
        ),
        active_manifest,
        manifest,
    )

    with pytest.raises(TypeError, match=field):
        replace(outcome, **{field: None})


def test_no_change_outcome_rejects_nonterminal_or_mismatched_state() -> None:
    manifest = _manifest()
    active_manifest = _manifest(corpus_version=ACTIVE_ID)
    active = _version(
        active_manifest,
        corpus_version=ACTIVE_ID,
        sequence=2,
        parent=OTHER_VERSION_ID,
    )
    update = _update(
        state=CorpusUpdateState.ABORTED,
        manifest=manifest,
        expected=ACTIVE_ID,
    )

    with pytest.raises(InvalidCorpusVersionContractError, match="terminal"):
        NoCorpusChangeOutcome(
            replace(update, state=CorpusUpdateState.READY),
            active,
            active_manifest,
            manifest,
        )
    with pytest.raises(InvalidCorpusVersionContractError, match="expectation"):
        NoCorpusChangeOutcome(
            replace(update, expected_corpus_version=OTHER_VERSION_ID),
            active,
            active_manifest,
            manifest,
        )
    with pytest.raises(InvalidCorpusVersionContractError, match="update corpus"):
        NoCorpusChangeOutcome(
            update,
            _version(
                _manifest(corpus_id=OTHER_CORPUS_ID, corpus_version=ACTIVE_ID),
                corpus_id=OTHER_CORPUS_ID,
                corpus_version=ACTIVE_ID,
                sequence=2,
                parent=OTHER_VERSION_ID,
            ),
            active_manifest,
            manifest,
        )


def test_no_change_outcome_rejects_active_manifest_mismatch() -> None:
    manifest = _manifest()
    update = _update(
        state=CorpusUpdateState.ABORTED,
        manifest=manifest,
        expected=ACTIVE_ID,
    )
    active = _version(
        _manifest(corpus_version=ACTIVE_ID),
        corpus_version=ACTIVE_ID,
        sequence=2,
        parent=OTHER_VERSION_ID,
    )

    with pytest.raises(InvalidCorpusVersionContractError, match="active manifest"):
        NoCorpusChangeOutcome(
            update,
            replace(active, manifest_digest="f" * 64),
            _manifest(corpus_version=ACTIVE_ID),
            manifest,
        )


def test_no_change_outcome_rejects_different_semantic_state() -> None:
    manifest = _manifest()
    update = _update(
        state=CorpusUpdateState.ABORTED,
        manifest=manifest,
        expected=ACTIVE_ID,
    )
    active_manifest = _manifest(
        corpus_version=ACTIVE_ID,
        memberships=(MEMBERSHIP,),
    )
    active = _version(
        active_manifest,
        corpus_version=ACTIVE_ID,
        sequence=2,
        parent=OTHER_VERSION_ID,
    )

    with pytest.raises(InvalidCorpusVersionContractError, match="semantics"):
        NoCorpusChangeOutcome(update, active, active_manifest, manifest)


def test_conflict_outcome_records_bounded_fail_closed_context() -> None:
    conflict = CorpusVersionConflictOutcome(
        corpus_id=CORPUS_ID,
        corpus_version=PROPOSED_ID,
        kind=CorpusVersionConflictKind.EXPECTED_ACTIVE_MISMATCH,
        expected_corpus_version=None,
        observed_active_corpus_version=ACTIVE_ID,
    )

    assert conflict.kind is CorpusVersionConflictKind.EXPECTED_ACTIVE_MISMATCH
    assert conflict.observed_active_corpus_version == ACTIVE_ID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus_id", PROPOSED_ID),
        ("corpus_version", CORPUS_ID),
        ("kind", "expected_active_mismatch"),
        ("expected_corpus_version", CORPUS_ID),
        ("observed_active_corpus_version", CORPUS_ID),
    ],
)
def test_conflict_outcome_rejects_wrong_types(field: str, value: object) -> None:
    conflict = CorpusVersionConflictOutcome(
        CORPUS_ID,
        PROPOSED_ID,
        CorpusVersionConflictKind.VERSION_ID_REUSED,
    )

    with pytest.raises(TypeError, match=field):
        replace(conflict, **{field: value})


def test_active_mismatch_conflict_requires_different_versions() -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="mismatched"):
        CorpusVersionConflictOutcome(
            CORPUS_ID,
            PROPOSED_ID,
            CorpusVersionConflictKind.EXPECTED_ACTIVE_MISMATCH,
            expected_corpus_version=ACTIVE_ID,
            observed_active_corpus_version=ACTIVE_ID,
        )


def test_rollback_outcome_wraps_audited_commit() -> None:
    publication = _commit(expected=ACTIVE_ID, restores=RESTORE_ID)

    assert RollbackCorpusVersionOutcome(publication).publication == publication
    with pytest.raises(TypeError, match="publication"):
        RollbackCorpusVersionOutcome(None)  # type: ignore[arg-type]
    with pytest.raises(InvalidCorpusVersionContractError, match="restore target"):
        RollbackCorpusVersionOutcome(_commit())
