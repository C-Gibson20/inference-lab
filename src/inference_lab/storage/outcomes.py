"""Typed outcomes for corpus update and publication operations."""

from dataclasses import dataclass
from enum import StrEnum

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage.versioning import (
    CorpusUpdate,
    CorpusUpdateState,
    CorpusVersion,
    CorpusVersionManifest,
    InvalidCorpusVersionContractError,
)


def _validate_replayed(value: object) -> None:
    """Require an explicit strict boolean replay indicator."""
    if type(value) is not bool:
        raise TypeError("replayed must be a boolean.")


def _validate_manifest_alignment(
    update: CorpusUpdate,
    manifest: CorpusVersionManifest,
) -> None:
    """Require an update and manifest to identify the same sealed proposal."""
    if manifest.corpus_id != update.corpus_id:
        raise InvalidCorpusVersionContractError(
            "the outcome manifest must belong to the update corpus."
        )
    if manifest.corpus_version != update.corpus_version:
        raise InvalidCorpusVersionContractError(
            "the outcome manifest must belong to the proposed corpus version."
        )
    if update.manifest_digest != manifest.manifest_digest:
        raise InvalidCorpusVersionContractError(
            "the outcome manifest digest must match the update."
        )
    if not manifest.has_required_components(update.required_component_kinds):
        raise InvalidCorpusVersionContractError(
            "the outcome manifest must contain exactly the required components."
        )


@dataclass(frozen=True, slots=True)
class BeginCorpusUpdateOutcome:
    """Return a newly preparing update or an exact replay of its begin call."""

    update: CorpusUpdate
    replayed: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the begin outcome is self-consistent."""
        if not isinstance(self.update, CorpusUpdate):
            raise TypeError("update must be a CorpusUpdate.")
        _validate_replayed(self.replayed)
        if not self.replayed and self.update.state is not CorpusUpdateState.PREPARING:
            raise InvalidCorpusVersionContractError(
                "a new begin outcome requires a preparing update."
            )


@dataclass(frozen=True, slots=True)
class SealCorpusUpdateOutcome:
    """Return an immutable ready update and its complete sealed manifest."""

    update: CorpusUpdate
    manifest: CorpusVersionManifest
    replayed: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the seal outcome is self-consistent."""
        if not isinstance(self.update, CorpusUpdate):
            raise TypeError("update must be a CorpusUpdate.")
        if not isinstance(self.manifest, CorpusVersionManifest):
            raise TypeError("manifest must be a CorpusVersionManifest.")
        _validate_replayed(self.replayed)
        permitted_states = {CorpusUpdateState.READY}
        if self.replayed:
            permitted_states.add(CorpusUpdateState.COMMITTED)
        if self.update.state not in permitted_states:
            raise InvalidCorpusVersionContractError(
                "a seal outcome requires a ready update or committed replay."
            )
        _validate_manifest_alignment(self.update, self.manifest)


@dataclass(frozen=True, slots=True)
class CommitCorpusUpdateOutcome:
    """Return one committed version, including an exact commit replay."""

    update: CorpusUpdate
    version: CorpusVersion
    manifest: CorpusVersionManifest
    replayed: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the commit outcome is self-consistent."""
        if not isinstance(self.update, CorpusUpdate):
            raise TypeError("update must be a CorpusUpdate.")
        if not isinstance(self.version, CorpusVersion):
            raise TypeError("version must be a CorpusVersion.")
        if not isinstance(self.manifest, CorpusVersionManifest):
            raise TypeError("manifest must be a CorpusVersionManifest.")
        _validate_replayed(self.replayed)
        if self.update.state is not CorpusUpdateState.COMMITTED:
            raise InvalidCorpusVersionContractError(
                "a commit outcome requires a committed update."
            )
        _validate_manifest_alignment(self.update, self.manifest)
        if not self.version.matches_manifest(self.manifest):
            raise InvalidCorpusVersionContractError(
                "the committed version must match the outcome manifest."
            )
        if self.version.parent_corpus_version != self.update.expected_corpus_version:
            raise InvalidCorpusVersionContractError(
                "the committed version parent must match the update expectation."
            )
        if self.version.restores_corpus_version != self.update.restores_corpus_version:
            raise InvalidCorpusVersionContractError(
                "the committed version restore target must match the update."
            )


@dataclass(frozen=True, slots=True)
class AbortCorpusUpdateOutcome:
    """Return an aborted proposal, including an exact abort replay."""

    update: CorpusUpdate
    replayed: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the abort outcome is self-consistent."""
        if not isinstance(self.update, CorpusUpdate):
            raise TypeError("update must be a CorpusUpdate.")
        _validate_replayed(self.replayed)
        if self.update.state is not CorpusUpdateState.ABORTED:
            raise InvalidCorpusVersionContractError(
                "an abort outcome requires an aborted update."
            )


@dataclass(frozen=True, slots=True)
class NoCorpusChangeOutcome:
    """Return a redundant proposal that published no new corpus version."""

    update: CorpusUpdate
    active_version: CorpusVersion
    active_manifest: CorpusVersionManifest
    proposed_manifest: CorpusVersionManifest
    replayed: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the no-change outcome is self-consistent."""
        if not isinstance(self.update, CorpusUpdate):
            raise TypeError("update must be a CorpusUpdate.")
        if not isinstance(self.active_version, CorpusVersion):
            raise TypeError("active_version must be a CorpusVersion.")
        if not isinstance(self.active_manifest, CorpusVersionManifest):
            raise TypeError("active_manifest must be a CorpusVersionManifest.")
        if not isinstance(self.proposed_manifest, CorpusVersionManifest):
            raise TypeError("proposed_manifest must be a CorpusVersionManifest.")
        _validate_replayed(self.replayed)
        if self.update.state is not CorpusUpdateState.ABORTED:
            raise InvalidCorpusVersionContractError(
                "a no-change outcome requires a terminal aborted proposal."
            )
        _validate_manifest_alignment(self.update, self.proposed_manifest)
        if self.update.expected_corpus_version != self.active_version.corpus_version:
            raise InvalidCorpusVersionContractError(
                "the no-change active version must match the update expectation."
            )
        if self.active_version.corpus_id != self.update.corpus_id:
            raise InvalidCorpusVersionContractError(
                "the no-change active version must belong to the update corpus."
            )
        if not self.active_version.matches_manifest(self.active_manifest):
            raise InvalidCorpusVersionContractError(
                "the no-change active manifest must match the active version."
            )
        if not self.proposed_manifest.has_same_semantic_state(self.active_manifest):
            raise InvalidCorpusVersionContractError(
                "a no-change proposal must match the active manifest semantics."
            )


class CorpusVersionConflictKind(StrEnum):
    """Bounded reasons why a corpus proposal cannot advance."""

    EXPECTED_ACTIVE_MISMATCH = "expected_active_mismatch"
    VERSION_ID_REUSED = "version_id_reused"
    UPDATE_PAYLOAD_MISMATCH = "update_payload_mismatch"


@dataclass(frozen=True, slots=True)
class CorpusVersionConflictOutcome:
    """Describe a fail-closed corpus-version conflict without payload data."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    kind: CorpusVersionConflictKind
    expected_corpus_version: CorpusVersionId | None = None
    observed_active_corpus_version: CorpusVersionId | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the conflict outcome is self-consistent."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.kind, CorpusVersionConflictKind):
            raise TypeError("kind must be a CorpusVersionConflictKind.")
        if self.expected_corpus_version is not None and not isinstance(
            self.expected_corpus_version,
            CorpusVersionId,
        ):
            raise TypeError(
                "expected_corpus_version must be a CorpusVersionId or None."
            )
        if self.observed_active_corpus_version is not None and not isinstance(
            self.observed_active_corpus_version,
            CorpusVersionId,
        ):
            raise TypeError(
                "observed_active_corpus_version must be a CorpusVersionId or None."
            )
        if (
            self.kind is CorpusVersionConflictKind.EXPECTED_ACTIVE_MISMATCH
            and self.observed_active_corpus_version == self.expected_corpus_version
        ):
            raise InvalidCorpusVersionContractError(
                "an active-version conflict requires a mismatched observation."
            )


@dataclass(frozen=True, slots=True)
class RollbackCorpusVersionOutcome:
    """Return a newly published version that restores historical corpus state."""

    publication: CommitCorpusUpdateOutcome

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the rollback outcome is self-consistent."""
        if not isinstance(self.publication, CommitCorpusUpdateOutcome):
            raise TypeError("publication must be a CommitCorpusUpdateOutcome.")
        if self.publication.version.restores_corpus_version is None:
            raise InvalidCorpusVersionContractError(
                "a rollback outcome requires a historical restore target."
            )
