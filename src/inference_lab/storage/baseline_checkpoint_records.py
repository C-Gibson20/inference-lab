"""Storage-owned records for durable baseline-ingest checkpoints."""

from dataclasses import dataclass
from enum import StrEnum

from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    OutputDigest,
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
