"""Build and validate the immutable baseline chunk generation manifest."""

import hashlib
from dataclasses import dataclass, field

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.ingest.baseline_contracts import BaselineFailureCategory
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
)
from inference_lab.storage import (
    ChunkSetReader,
    ChunkSetSelectionRecord,
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentGenerationValidator,
)
from inference_lab.storage.versioning import ComponentKind

BASELINE_CHUNK_COMPONENT_KIND = ComponentKind("chunks")
BASELINE_CHUNK_IMPLEMENTATION = "baseline-chunks"
BASELINE_CHUNK_FORMAT_VERSION = "1.0.0"
BASELINE_CHUNK_CONTRACT_VERSION = "1.0.0"


class BaselineGenerationValidationError(ValueError):
    """Represent a payload-free generation completeness or compatibility error."""

    def __init__(self, category: BaselineFailureCategory) -> None:
        """Create a bounded generation validation error."""
        if category not in {
            BaselineFailureCategory.GENERATION_VALIDATION_FAILED,
            BaselineFailureCategory.CORRUPT_STATE,
        }:
            raise ValueError(
                "generation errors require a validation or state category."
            )
        super().__init__(category.value)
        self.category = category


@dataclass(frozen=True, slots=True)
class BaselineChunkGenerationEntry:
    """Map one staged document membership to one exact chunk-set selection."""

    membership: CorpusDocumentVersionMembership
    selection: ChunkSetSelectionRecord

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate selection identity against its staged membership."""
        if not isinstance(self.membership, CorpusDocumentVersionMembership):
            raise TypeError("membership must be a CorpusDocumentVersionMembership.")
        if not isinstance(self.selection, ChunkSetSelectionRecord):
            raise TypeError("selection must be a ChunkSetSelectionRecord.")
        if (
            self.selection.document_id != self.membership.document_id
            or self.selection.document_version_id != self.membership.document_version_id
        ):
            raise BaselineGenerationValidationError(
                BaselineFailureCategory.GENERATION_VALIDATION_FAILED,
            )

    def key(self) -> tuple[str, str]:
        """Return the canonical entry identity."""
        return (
            self.membership.document_id.as_str(),
            self.membership.document_version_id.as_str(),
        )


@dataclass(frozen=True, slots=True)
class BaselineChunkGenerationManifest:
    """Describe a complete immutable chunks generation for one proposal."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    configuration_identity: ConfigurationIdentity
    generation_id: str
    entries: tuple[BaselineChunkGenerationEntry, ...]
    source_manifest_digest: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(
            self,
            "source_manifest_digest",
            self._build_source_manifest_digest(),
        )

    def _validate(self) -> None:
        """Validate complete ownership, ordering, and configuration identity."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.configuration_identity, ConfigurationIdentity):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")
        if self.configuration_identity.algorithm is not DigestAlgorithm.SHA256:
            raise BaselineGenerationValidationError(
                BaselineFailureCategory.GENERATION_VALIDATION_FAILED,
            )
        if not isinstance(self.generation_id, str):
            raise TypeError("generation_id must be a string.")
        if (
            not self.generation_id
            or "/" in self.generation_id
            or "\\" in self.generation_id
        ):
            raise BaselineGenerationValidationError(
                BaselineFailureCategory.GENERATION_VALIDATION_FAILED,
            )
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be a tuple.")
        if any(
            not isinstance(entry, BaselineChunkGenerationEntry)
            for entry in self.entries
        ):
            raise TypeError("entries must contain BaselineChunkGenerationEntry values.")
        if any(
            entry.membership.corpus_id != self.corpus_id
            or entry.selection.configuration_identity != self.configuration_identity
            for entry in self.entries
        ):
            raise BaselineGenerationValidationError(
                BaselineFailureCategory.GENERATION_VALIDATION_FAILED,
            )
        keys = tuple(entry.key() for entry in self.entries)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise BaselineGenerationValidationError(
                BaselineFailureCategory.GENERATION_VALIDATION_FAILED,
            )

    def _build_source_manifest_digest(self) -> str:
        """Digest all staged memberships and selected immutable chunk identities."""
        payload: JsonObject = {
            "entries": [
                {
                    "document_id": entry.membership.document_id.as_str(),
                    "document_version_id": entry.membership.document_version_id.as_str(),
                    "payload_digest": entry.selection.payload_digest.as_str(),
                }
                for entry in self.entries
            ],
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def component_generation(self) -> ComponentGenerationReference:
        """Return the preparing storage-generation reference for registration."""
        return ComponentGenerationReference(
            component_kind=BASELINE_CHUNK_COMPONENT_KIND,
            generation_id=self.generation_id,
            implementation=BASELINE_CHUNK_IMPLEMENTATION,
            format_version=BASELINE_CHUNK_FORMAT_VERSION,
            contract_version=BASELINE_CHUNK_CONTRACT_VERSION,
            configuration_digest=self.configuration_identity.digest,
            source_manifest_digest=self.source_manifest_digest,
            corpus_id=self.corpus_id,
            corpus_version=self.corpus_version,
            state=ComponentGenerationState.PREPARING,
        )

    def build_validator(
        self,
        reader: ChunkSetReader,
    ) -> ComponentGenerationValidator:
        """Build the validator used by the generic publication seam."""
        return BaselineChunkGenerationValidator(reader, self)

    def validate_memberships(
        self,
        memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> None:
        """Require a one-to-one canonical mapping for proposed membership."""
        if not isinstance(memberships, tuple):
            raise TypeError("memberships must be a tuple.")
        expected = tuple(entry.membership for entry in self.entries)
        if memberships != expected:
            raise BaselineGenerationValidationError(
                BaselineFailureCategory.GENERATION_VALIDATION_FAILED,
            )


class BaselineChunkGenerationValidator:
    """Validate exact durable chunk selections before component registration."""

    def __init__(
        self,
        reader: ChunkSetReader,
        manifest: BaselineChunkGenerationManifest,
    ) -> None:
        """Bind a corpus-authorised reader to one immutable manifest."""
        if not isinstance(manifest, BaselineChunkGenerationManifest):
            raise TypeError("manifest must be a BaselineChunkGenerationManifest.")
        self._reader = reader
        self._manifest = manifest

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        """Return whether every manifest entry is durable and exactly compatible."""
        if not isinstance(generation, ComponentGenerationReference):
            raise TypeError("generation must be a ComponentGenerationReference.")
        self._manifest.validate_memberships(document_memberships)
        if generation != self._manifest.component_generation():
            return False

        for entry in self._manifest.entries:
            selection = self._reader.get_chunk_set_selection(
                self._manifest.corpus_id,
                entry.membership.document_version_id,
                self._manifest.configuration_identity,
            )
            if selection != entry.selection:
                return False
        return True
