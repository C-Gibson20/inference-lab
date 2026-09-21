"""Public contracts for the local serving ingest facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from inference_lab.ingest import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_PARSER_REGISTRY,
    CURRENT_PARSER_VERSION,
    ChunkerVersion,
    ChunkingPolicy,
    ChunkTokeniser,
    DiscoveryRequest,
)
from inference_lab.ingest._chunk_identity import _configuration_identity_from_values
from inference_lab.ingest.classification import (
    CURRENT_SOURCE_CLASSIFIER,
    SourceClassifierRegistry,
)
from inference_lab.ingest.hashing import StableReadFileSystem
from inference_lab.ingest.identity_resolution import (
    DiscoveryIdentityIdAllocator,
    UuidDiscoveryIdentityIdAllocator,
)
from inference_lab.ingest.parser_selection import ParserRegistry
from inference_lab.ingest.traversal import TraversalFileSystem
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
)
from inference_lab.retrieval import LexicalIndexConfiguration
from inference_lab.storage import (
    BaselineCheckpointReader,
    BaselineCheckpointWriter,
    ComponentKind,
    MetadataRepository,
    MetadataTransaction,
)


class _ServingTransaction(MetadataTransaction, BaselineCheckpointWriter):
    """Combined transaction surface used by the serving ingest facade."""


class _ServingRepository(MetadataRepository, BaselineCheckpointReader):
    """Combined repository surface used by the serving ingest facade."""

    def transaction(self) -> _ServingTransaction:
        """Return a metadata transaction with checkpoint writes."""
        ...


_BASELINE_REQUIRED_COMPONENT_KINDS = (
    ComponentKind("chunks"),
    ComponentKind("lexical-index"),
)


class ServingIngestFailureCategory(StrEnum):
    """Bounded failures raised by the serving ingest facade."""

    INVALID_REQUEST = "invalid_request"
    CORPUS_NOT_FOUND = "corpus_not_found"
    INCOMPATIBLE_RESUME = "incompatible_resume"


class ServingIngestError(ValueError):
    """Expose a bounded, payload-free operator-facing ingest error."""

    def __init__(self, category: ServingIngestFailureCategory) -> None:
        if not isinstance(category, ServingIngestFailureCategory):
            raise TypeError("category must be a ServingIngestFailureCategory.")
        super().__init__(category.value)
        self.category = category


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ServingIngestRequest:
    """Describe one explicit local ingest attempt for one selected corpus."""

    corpus_id: CorpusId
    discovery_request: DiscoveryRequest
    proposed_corpus_version: CorpusVersionId
    expected_corpus_version: CorpusVersionId | None
    configuration_identity: ConfigurationIdentity
    chunking_policy: ChunkingPolicy
    tokeniser: ChunkTokeniser = field(repr=False)
    parser_version: ParserVersion = CURRENT_PARSER_VERSION
    chunker_version: ChunkerVersion = CURRENT_CHUNKER_VERSION
    excluded_paths: tuple[Path, ...] = ()
    lexical_configuration: LexicalIndexConfiguration = field(
        default_factory=LexicalIndexConfiguration,
        repr=False,
    )
    parser_registry: ParserRegistry = field(
        default=CURRENT_PARSER_REGISTRY,
        repr=False,
    )
    classifier_registry: SourceClassifierRegistry = field(
        default=CURRENT_SOURCE_CLASSIFIER,
        repr=False,
    )
    traversal_filesystem: TraversalFileSystem | None = field(
        default=None,
        repr=False,
    )
    source_filesystem: StableReadFileSystem | None = field(
        default=None,
        repr=False,
    )
    identity_allocator: DiscoveryIdentityIdAllocator = field(
        default_factory=UuidDiscoveryIdentityIdAllocator,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the serving request before any local state is changed."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.discovery_request, DiscoveryRequest):
            raise TypeError("discovery_request must be a DiscoveryRequest.")
        if self.discovery_request.corpus_id != self.corpus_id:
            raise ValueError("discovery_request must target the request corpus.")
        if not isinstance(self.proposed_corpus_version, CorpusVersionId):
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
        if not isinstance(self.configuration_identity, ConfigurationIdentity):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")
        if not isinstance(self.chunking_policy, ChunkingPolicy):
            raise TypeError("chunking_policy must be a ChunkingPolicy.")
        if not isinstance(self.tokeniser, ChunkTokeniser):
            raise TypeError("tokeniser must implement ChunkTokeniser.")
        if not isinstance(self.parser_version, type(CURRENT_PARSER_VERSION)):
            raise TypeError("parser_version must be a ParserVersion.")
        if not isinstance(self.chunker_version, ChunkerVersion):
            raise TypeError("chunker_version must be a ChunkerVersion.")
        if self.chunker_version != CURRENT_CHUNKER_VERSION:
            raise ValueError("unsupported chunker version.")
        expected_configuration_identity = _configuration_identity_from_values(
            parser_version=self.parser_version,
            chunker_version=self.chunker_version,
            tokeniser_identity=self.tokeniser.identity,
            policy=self.chunking_policy,
        )
        if self.configuration_identity != expected_configuration_identity:
            raise ValueError("configuration_identity does not match chunk inputs.")
        if not isinstance(self.excluded_paths, tuple):
            raise TypeError("excluded_paths must be a tuple.")
        if any(not isinstance(path, Path) for path in self.excluded_paths):
            raise TypeError("excluded_paths must contain Path values.")
        if not isinstance(self.lexical_configuration, LexicalIndexConfiguration):
            raise TypeError(
                "lexical_configuration must be a LexicalIndexConfiguration."
            )
        if not isinstance(self.parser_registry, ParserRegistry):
            raise TypeError("parser_registry must be a ParserRegistry.")
        if not isinstance(self.classifier_registry, SourceClassifierRegistry):
            raise TypeError("classifier_registry must be a SourceClassifierRegistry.")
        if self.traversal_filesystem is not None and not isinstance(
            self.traversal_filesystem,
            TraversalFileSystem,
        ):
            raise TypeError(
                "traversal_filesystem must implement TraversalFileSystem or be None."
            )
        if self.source_filesystem is not None and not isinstance(
            self.source_filesystem,
            StableReadFileSystem,
        ):
            raise TypeError(
                "source_filesystem must implement StableReadFileSystem or be None."
            )
        for name in (
            "new_source_observation_id",
            "new_document_id",
            "new_document_version_id",
        ):
            if not callable(getattr(self.identity_allocator, name, None)):
                raise TypeError(f"identity_allocator must provide {name}().")
