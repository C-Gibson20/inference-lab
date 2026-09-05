"""Tests for immutable corpus update and committed-version records."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage import (
    ComponentKind,
    CorpusUpdate,
    CorpusUpdateState,
    CorpusVersion,
    CorpusVersionManifest,
    InvalidCorpusVersionContractError,
)

CREATED_AT = datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
OTHER_CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000002")
VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000003")
PARENT_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000004")
RESTORED_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000005")
DIGEST = "a" * 64


def _update(
    *,
    expected: CorpusVersionId | None = None,
    kinds: tuple[ComponentKind, ...] = (),
    state: CorpusUpdateState = CorpusUpdateState.PREPARING,
    created_at: datetime = CREATED_AT,
    digest: str | None = None,
    restores: CorpusVersionId | None = None,
) -> CorpusUpdate:
    return CorpusUpdate(
        corpus_id=CORPUS_ID,
        corpus_version=VERSION_ID,
        expected_corpus_version=expected,
        required_component_kinds=kinds,
        state=state,
        created_at=created_at,
        manifest_digest=digest,
        restores_corpus_version=restores,
    )


def _version(
    *,
    corpus_id: CorpusId = CORPUS_ID,
    corpus_version: CorpusVersionId = VERSION_ID,
    sequence: int = 1,
    parent: CorpusVersionId | None = None,
    digest: str = DIGEST,
    created_at: datetime = CREATED_AT,
    restores: CorpusVersionId | None = None,
) -> CorpusVersion:
    return CorpusVersion(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        sequence=sequence,
        parent_corpus_version=parent,
        manifest_digest=digest,
        created_at=created_at,
        restores_corpus_version=restores,
    )


def test_preparing_update_canonicalises_required_components_and_is_immutable() -> None:
    update = _update(
        kinds=(ComponentKind("metadata"), ComponentKind("dense-index")),
    )

    assert tuple(kind.as_str() for kind in update.required_component_kinds) == (
        "dense-index",
        "metadata",
    )
    assert update.manifest_digest is None
    with pytest.raises(FrozenInstanceError):
        update.state = CorpusUpdateState.READY  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus_id", VERSION_ID),
        ("corpus_version", CORPUS_ID),
        ("expected_corpus_version", CORPUS_ID),
        ("required_component_kinds", []),
        ("required_component_kinds", ("metadata",)),
        ("state", "preparing"),
        ("created_at", "2026-09-05T12:30:00Z"),
        ("manifest_digest", 1),
        ("restores_corpus_version", CORPUS_ID),
    ],
)
def test_update_rejects_wrong_field_types(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        replace(_update(), **{field: value})


def test_update_rejects_duplicate_required_component_kind() -> None:
    kind = ComponentKind("lexical-index")

    with pytest.raises(InvalidCorpusVersionContractError, match="duplicates"):
        _update(kinds=(kind, kind))


@pytest.mark.parametrize(
    ("state", "digest", "message"),
    [
        (CorpusUpdateState.PREPARING, DIGEST, "preparing"),
        (CorpusUpdateState.READY, None, "ready and committed"),
        (CorpusUpdateState.COMMITTED, None, "ready and committed"),
    ],
)
def test_update_enforces_sealed_digest_lifecycle(
    state: CorpusUpdateState,
    digest: str | None,
    message: str,
) -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match=message):
        _update(state=state, digest=digest)


def test_ready_and_committed_updates_require_canonical_manifest_digest() -> None:
    assert (
        _update(state=CorpusUpdateState.READY, digest=DIGEST).manifest_digest == DIGEST
    )
    assert (
        _update(state=CorpusUpdateState.COMMITTED, digest=DIGEST).manifest_digest
        == DIGEST
    )
    with pytest.raises(InvalidCorpusVersionContractError, match="manifest_digest"):
        _update(state=CorpusUpdateState.READY, digest="A" * 64)


def test_aborted_update_preserves_optional_sealed_digest() -> None:
    unsealed = _update(state=CorpusUpdateState.ABORTED)
    sealed = replace(unsealed, manifest_digest=DIGEST)

    assert unsealed.manifest_digest is None
    assert sealed.manifest_digest == DIGEST


def test_update_rejects_proposed_version_as_expected_parent() -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="expected active"):
        _update(expected=VERSION_ID)


def test_rollback_update_requires_distinct_proposal_and_active_parent() -> None:
    update = _update(
        expected=PARENT_VERSION_ID,
        restores=RESTORED_VERSION_ID,
    )

    assert update.restores_corpus_version == RESTORED_VERSION_ID
    with pytest.raises(InvalidCorpusVersionContractError, match="expected active"):
        replace(update, expected_corpus_version=None)
    with pytest.raises(InvalidCorpusVersionContractError, match="own proposed"):
        replace(update, restores_corpus_version=VERSION_ID)


@pytest.mark.parametrize(
    "created_at",
    [
        datetime(2026, 9, 5, 12, 30),
        datetime(
            2026,
            9,
            5,
            12,
            30,
            tzinfo=timezone(timedelta(hours=1)),
        ),
    ],
)
def test_update_requires_timezone_aware_utc_creation_time(
    created_at: datetime,
) -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="created_at"):
        _update(created_at=created_at)


def test_first_version_matches_complete_manifest_and_is_immutable() -> None:
    manifest = CorpusVersionManifest(CORPUS_ID, VERSION_ID)
    version = _version(digest=manifest.manifest_digest)

    assert version.matches_manifest(manifest)
    assert not version.matches_manifest(
        CorpusVersionManifest(CORPUS_ID, PARENT_VERSION_ID)
    )
    assert not version.matches_manifest(
        CorpusVersionManifest(OTHER_CORPUS_ID, VERSION_ID)
    )
    with pytest.raises(FrozenInstanceError):
        version.sequence = 2  # type: ignore[misc]


def test_later_version_requires_parent_and_supports_rollback_audit() -> None:
    version = _version(
        sequence=3,
        parent=PARENT_VERSION_ID,
        restores=RESTORED_VERSION_ID,
    )

    assert version.sequence == 3
    assert version.parent_corpus_version == PARENT_VERSION_ID
    assert version.restores_corpus_version == RESTORED_VERSION_ID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus_id", VERSION_ID),
        ("corpus_version", CORPUS_ID),
        ("sequence", True),
        ("parent_corpus_version", CORPUS_ID),
        ("manifest_digest", 1),
        ("created_at", "2026-09-05T12:30:00Z"),
        ("restores_corpus_version", CORPUS_ID),
    ],
)
def test_version_rejects_wrong_field_types(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        replace(_version(), **{field: value})


@pytest.mark.parametrize(
    ("sequence", "parent", "message"),
    [
        (0, None, "at least one"),
        (1, PARENT_VERSION_ID, "first corpus version"),
        (2, None, "requires a parent"),
        (2, VERSION_ID, "own parent"),
    ],
)
def test_version_rejects_invalid_sequence_parent_shape(
    sequence: int,
    parent: CorpusVersionId | None,
    message: str,
) -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match=message):
        _version(sequence=sequence, parent=parent)


def test_version_rejects_invalid_digest_and_creation_time() -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="manifest_digest"):
        _version(digest="not-a-digest")
    with pytest.raises(InvalidCorpusVersionContractError, match="created_at"):
        _version(created_at=datetime(2026, 9, 5))


def test_version_rejects_invalid_rollback_shape() -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="first corpus"):
        _version(restores=RESTORED_VERSION_ID)
    with pytest.raises(InvalidCorpusVersionContractError, match="restore itself"):
        _version(sequence=2, parent=PARENT_VERSION_ID, restores=VERSION_ID)


def test_matches_manifest_requires_manifest_contract() -> None:
    with pytest.raises(TypeError, match="manifest"):
        _version().matches_manifest(None)  # type: ignore[arg-type]
