"""Storage-owned records for durable baseline-ingest checkpoints."""

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

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


class BaselineManifestRecordState(StrEnum):
    """Persisted lifecycle values for one baseline manifest."""

    PREPARING = "preparing"
    READY = "ready"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineManifestRecord:
    """Persist one baseline manifest without ingest implementation types."""

    corpus_id: CorpusId
    proposed_corpus_version: CorpusVersionId
    expected_corpus_version: CorpusVersionId | None
    configuration_identity: ConfigurationIdentity
    required_component_kinds: tuple[ComponentKind, ...]
    discovery_summary_identity: OutputDigest
    state: BaselineManifestRecordState

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the storage representation."""
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
        if not isinstance(
            self.state,
            BaselineManifestRecordState,
        ):
            raise TypeError("state must be a BaselineManifestRecordState.")

    def key(self) -> tuple[str, str]:
        """Return the corpus-scoped durable proposal key."""
        return (
            self.corpus_id.as_str(),
            self.proposed_corpus_version.as_str(),
        )


class BaselineCheckpointRecordStage(StrEnum):
    """Persisted baseline pipeline stages."""

    DISCOVERY = "discovery"
    SOURCE_ACQUISITION = "source_acquisition"
    PARSE = "parse"
    PARSE_PERSISTENCE = "parse_persistence"
    CHUNK = "chunk"
    CHUNK_PERSISTENCE = "chunk_persistence"
    MEMBERSHIP = "membership"
    PUBLICATION = "publication"


class BaselineCheckpointRecordStatus(StrEnum):
    """Persisted document-stage checkpoint states."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class BaselineCheckpointRecordIdentityKind(StrEnum):
    """Stable tags for typed checkpoint identities."""

    DOCUMENT = "document"
    DOCUMENT_VERSION = "document_version"
    RECORD = "record"
    SOURCE_OBSERVATION = "source_observation"
    SOURCE_SPAN = "source_span"


class BaselineCheckpointRecordFailureCategory(StrEnum):
    """Persisted bounded baseline failure categories."""

    INVALID_REQUEST = "invalid_request"
    INCOMPATIBLE_RESUME = "incompatible_resume"
    DISCOVERY_FAILED = "discovery_failed"
    SOURCE_ACQUISITION_FAILED = "source_acquisition_failed"
    PARSER_UNAVAILABLE = "parser_unavailable"
    PARSE_FAILED = "parse_failed"
    PARSE_PERSISTENCE_FAILED = "parse_persistence_failed"
    CHUNK_FAILED = "chunk_failed"
    CHUNK_PERSISTENCE_FAILED = "chunk_persistence_failed"
    MEMBERSHIP_CONFLICT = "membership_conflict"
    GENERATION_BUILD_FAILED = "generation_build_failed"
    GENERATION_VALIDATION_FAILED = "generation_validation_failed"
    STORAGE_UNAVAILABLE = "storage_unavailable"
    PUBLICATION_CONFLICT = "publication_conflict"
    PUBLICATION_CONCURRENCY = "publication_concurrency"
    CORRUPT_STATE = "corrupt_state"


def _validate_identity_value(
    kind: BaselineCheckpointRecordIdentityKind,
    value: object,
) -> str:
    """Validate and return one canonical typed identity value."""
    if not isinstance(kind, BaselineCheckpointRecordIdentityKind):
        raise TypeError("kind must be a BaselineCheckpointRecordIdentityKind.")
    if not isinstance(value, str):
        raise TypeError("value must be a string.")

    identifier_types = {
        BaselineCheckpointRecordIdentityKind.DOCUMENT: DocumentId,
        BaselineCheckpointRecordIdentityKind.DOCUMENT_VERSION: DocumentVersionId,
        BaselineCheckpointRecordIdentityKind.RECORD: RecordId,
        BaselineCheckpointRecordIdentityKind.SOURCE_OBSERVATION: (SourceObservationId),
        BaselineCheckpointRecordIdentityKind.SOURCE_SPAN: SourceSpanId,
    }

    identifier_types[kind].from_str(value)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineCheckpointIdentityRecord:
    """Persist one typed opaque identity used by a checkpoint."""

    kind: BaselineCheckpointRecordIdentityKind
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            _validate_identity_value(self.kind, self.value),
        )

    def key(self) -> tuple[str, str]:
        """Return the canonical persistence key."""
        return self.kind.value, self.value


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineCheckpointRecord:
    """Persist one corpus-scoped document-stage checkpoint."""

    corpus_id: CorpusId
    proposed_corpus_version: CorpusVersionId
    document_version_id: DocumentVersionId
    stage: BaselineCheckpointRecordStage
    configuration_identity: ConfigurationIdentity
    document_id: DocumentId
    input_identities: tuple[BaselineCheckpointIdentityRecord, ...]
    output_identities: tuple[BaselineCheckpointIdentityRecord, ...]
    status: BaselineCheckpointRecordStatus
    failure_category: BaselineCheckpointRecordFailureCategory | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate checkpoint identity, references, and lifecycle state."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
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
        if not isinstance(
            self.stage,
            BaselineCheckpointRecordStage,
        ):
            raise TypeError("stage must be a BaselineCheckpointRecordStage.")
        if not isinstance(
            self.configuration_identity,
            ConfigurationIdentity,
        ):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")

        input_identities = self._validate_identity_tuple(
            self.input_identities,
            field_name="input_identities",
        )
        output_identities = self._validate_identity_tuple(
            self.output_identities,
            field_name="output_identities",
        )
        object.__setattr__(self, "input_identities", input_identities)
        object.__setattr__(self, "output_identities", output_identities)

        if not isinstance(
            self.status,
            BaselineCheckpointRecordStatus,
        ):
            raise TypeError("status must be a BaselineCheckpointRecordStatus.")
        if self.failure_category is not None and not isinstance(
            self.failure_category,
            BaselineCheckpointRecordFailureCategory,
        ):
            raise TypeError(
                "failure_category must be a "
                "BaselineCheckpointRecordFailureCategory or None."
            )

        if self.status is BaselineCheckpointRecordStatus.STARTED:
            if self.failure_category is not None:
                raise ValueError(
                    "started checkpoints must not have a failure category."
                )
            if self.output_identities:
                raise ValueError("started checkpoints must not have output identities.")

        elif self.status is BaselineCheckpointRecordStatus.COMPLETED:
            if self.failure_category is not None:
                raise ValueError(
                    "completed checkpoints must not have a failure category."
                )

        elif self.status is BaselineCheckpointRecordStatus.FAILED:
            if self.failure_category is None:
                raise ValueError("failed checkpoints require a failure category.")
            if self.output_identities:
                raise ValueError("failed checkpoints must not have output identities.")

    @staticmethod
    def _validate_identity_tuple(
        values: object,
        *,
        field_name: str,
    ) -> tuple[BaselineCheckpointIdentityRecord, ...]:
        """Validate and canonicalise storage identity records."""
        if not isinstance(values, tuple):
            raise TypeError(f"{field_name} must be a tuple.")
        if any(
            not isinstance(value, BaselineCheckpointIdentityRecord) for value in values
        ):
            raise TypeError(
                f"{field_name} must contain only "
                "BaselineCheckpointIdentityRecord values."
            )

        keys = tuple(value.key() for value in values)
        if len(keys) != len(set(keys)):
            raise ValueError(f"{field_name} must not contain duplicates.")

        typed_values = cast(
            tuple[BaselineCheckpointIdentityRecord, ...],
            values,
        )
        return tuple(sorted(typed_values, key=BaselineCheckpointIdentityRecord.key))

    def key(self) -> tuple[str, str, str, str, str]:
        """Return the corpus-scoped compare-and-set key."""
        return (
            self.corpus_id.as_str(),
            self.proposed_corpus_version.as_str(),
            self.document_version_id.as_str(),
            self.stage.value,
            self.configuration_identity.as_str(),
        )
