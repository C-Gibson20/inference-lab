"""Immutable contracts for durable baseline-ingest checkpoints."""

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from inference_lab.ingest.baseline_contracts import (
    BaselineFailureCategory,
    BaselineStage,
)
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
    RecordId,
    SourceObservationId,
    SourceSpanId,
)
from inference_lab.storage.versioning import (
    ComponentKind,
    canonicalise_component_kinds,
)


class BaselineManifestState(StrEnum):
    """Lifecycle of one proposed-version baseline manifest."""

    PREPARING = "preparing"
    READY = "ready"
    COMMITTED = "committed"
    ABORTED = "aborted"


class BaselineCheckpointStatus(StrEnum):
    """Status of one document-stage checkpoint."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineIngestManifest:
    """Describe the durable identity and lifecycle of one baseline proposal."""

    corpus_id: CorpusId
    proposed_corpus_version: CorpusVersionId
    expected_corpus_version: CorpusVersionId | None
    configuration_identity: ConfigurationIdentity
    required_component_kinds: tuple[ComponentKind, ...]
    discovery_summary_identity: OutputDigest
    state: BaselineManifestState

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate proposal identity, configuration, and lifecycle fields."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(
            self.proposed_corpus_version,
            CorpusVersionId,
        ):
            raise TypeError("proposed_corpus_version must be a CorpusVersionId.")
        if self.expected_corpus_version is not None and not isinstance(
            self.expected_corpus_version,
            CorpusVersionId,
        ):
            raise TypeError(
                "expected_corpus_version must be a CorpusVersionId or None."
            )
        if self.proposed_corpus_version == self.expected_corpus_version:
            raise ValueError(
                "proposed_corpus_version must differ from expected_corpus_version."
            )
        if not isinstance(
            self.configuration_identity,
            ConfigurationIdentity,
        ):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")
        canonical_kinds = canonicalise_component_kinds(self.required_component_kinds)
        object.__setattr__(
            self,
            "required_component_kinds",
            canonical_kinds,
        )
        if not isinstance(
            self.discovery_summary_identity,
            OutputDigest,
        ):
            raise TypeError("discovery_summary_identity must be an OutputDigest.")
        if not isinstance(self.state, BaselineManifestState):
            raise TypeError("state must be a BaselineManifestState.")

    def key(self) -> tuple[str, str]:
        """Return the durable proposal key."""
        return (
            self.corpus_id.as_str(),
            self.proposed_corpus_version.as_str(),
        )


type BaselineCheckpointIdentity = (
    DocumentId | DocumentVersionId | RecordId | SourceObservationId | SourceSpanId
)


@dataclass(frozen=True, slots=True)
class BaselineWorkKey:
    """Identify one invalidation-sensitive document-stage attempt."""

    proposed_corpus_version: CorpusVersionId
    document_version_id: DocumentVersionId
    stage: BaselineStage
    configuration_identity: ConfigurationIdentity

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate every component of the stable work identity."""
        if not isinstance(
            self.proposed_corpus_version,
            CorpusVersionId,
        ):
            raise TypeError("proposed_corpus_version must be a CorpusVersionId.")
        if not isinstance(
            self.document_version_id,
            DocumentVersionId,
        ):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.stage, BaselineStage):
            raise TypeError("stage must be a BaselineStage.")
        if not isinstance(
            self.configuration_identity,
            ConfigurationIdentity,
        ):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")

    def key(self) -> tuple[str, str, str, str]:
        """Return the canonical comparison and persistence key."""
        return (
            self.proposed_corpus_version.as_str(),
            self.document_version_id.as_str(),
            self.stage.value,
            self.configuration_identity.as_str(),
        )


def _validate_checkpoint_identities(
    values: object,
    *,
    field_name: str,
) -> tuple[BaselineCheckpointIdentity, ...]:
    """Validate a canonical tuple of opaque checkpoint identities."""
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple.")

    identity_types = (
        DocumentId,
        DocumentVersionId,
        RecordId,
        SourceObservationId,
        SourceSpanId,
    )

    if any(not isinstance(value, identity_types) for value in values):
        raise TypeError(f"{field_name} must contain only typed provenance identities.")

    typed_values = cast(
        tuple[BaselineCheckpointIdentity, ...],
        values,
    )
    keys = tuple((type(value).__name__, value.as_str()) for value in typed_values)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must not contain duplicates.")

    return typed_values


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineCheckpoint:
    """Describe one document-stage checkpoint."""

    work_key: BaselineWorkKey
    document_id: DocumentId
    input_identities: tuple[BaselineCheckpointIdentity, ...]
    output_identities: tuple[BaselineCheckpointIdentity, ...]
    status: BaselineCheckpointStatus
    failure_category: BaselineFailureCategory | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate checkpoint state and its bounded references."""
        if not isinstance(self.work_key, BaselineWorkKey):
            raise TypeError("work_key must be a BaselineWorkKey.")
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")

        input_identities = _validate_checkpoint_identities(
            self.input_identities,
            field_name="input_identities",
        )
        output_identities = _validate_checkpoint_identities(
            self.output_identities,
            field_name="output_identities",
        )

        object.__setattr__(self, "input_identities", input_identities)
        object.__setattr__(self, "output_identities", output_identities)

        if not isinstance(self.status, BaselineCheckpointStatus):
            raise TypeError("status must be a BaselineCheckpointStatus.")
        if self.failure_category is not None and not isinstance(
            self.failure_category,
            BaselineFailureCategory,
        ):
            raise TypeError(
                "failure_category must be a BaselineFailureCategory or None."
            )

        if self.status is BaselineCheckpointStatus.STARTED:
            if self.failure_category is not None:
                raise ValueError(
                    "started checkpoints must not have a failure category."
                )
            if self.output_identities:
                raise ValueError("started checkpoints must not have output identities.")

        elif self.status is BaselineCheckpointStatus.COMPLETED:
            if self.failure_category is not None:
                raise ValueError(
                    "completed checkpoints must not have a failure category."
                )

        elif self.status is BaselineCheckpointStatus.FAILED:
            if self.failure_category is None:
                raise ValueError("failed checkpoints require a failure category.")
            if self.output_identities:
                raise ValueError("failed checkpoints must not have output identities.")

    def key(self) -> tuple[str, str, str, str]:
        """Return the stable work key for this checkpoint."""
        return self.work_key.key()
