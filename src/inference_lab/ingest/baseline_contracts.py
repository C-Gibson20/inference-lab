"""Immutable contracts for incremental baseline ingest orchestration."""

from dataclasses import dataclass, field
from enum import StrEnum

from inference_lab._validation import is_strict_integer
from inference_lab.ingest.chunking import (
    ChunkerVersion,
    ChunkingPolicy,
    TokeniserIdentity,
)
from inference_lab.ingest.discovery import DiscoveryRequest
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)


class BaselineStage(StrEnum):
    """Stable stages exposed to orchestration and profiling."""

    DISCOVERY = "discovery"
    SOURCE_ACQUISITION = "source_acquisition"
    PARSE = "parse"
    PARSE_PERSISTENCE = "parse_persistence"
    CHUNK = "chunk"
    CHUNK_PERSISTENCE = "chunk_persistence"
    MEMBERSHIP = "membership"
    PUBLICATION = "publication"


class BaselineExecutionStatus(StrEnum):
    """Overall execution status independent of publication status."""

    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    FAILED = "failed"
    CONFLICT = "conflict"


class BaselinePublicationStatus(StrEnum):
    """Publication status independent of execution status."""

    COMMITTED = "committed"
    NO_CHANGE = "no_change"
    NOT_PUBLISHED = "not_published"


class BaselineDocumentOutcomeKind(StrEnum):
    """Bounded outcomes for one discovered document."""

    UNCHANGED_REUSED = "unchanged_reused"
    NEW_DOCUMENT = "new_document"
    CHANGED_VERSION = "changed_version"
    BYTE_IDENTICAL_DUPLICATE = "byte_identical_duplicate"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class BaselineStageDisposition(StrEnum):
    """Result of a document-level parse or chunk stage."""

    NOT_ATTEMPTED = "not_attempted"
    REUSED = "reused"
    COMPLETED = "completed"
    FAILED = "failed"


class BaselineFailureScope(StrEnum):
    """Scope of work invalidated by a bounded failure."""

    DOCUMENT = "document"
    REQUEST = "request"


class BaselineFailureSeverity(StrEnum):
    """Severity of a bounded failure for execution decisions."""

    RECOVERABLE = "recoverable"
    FATAL = "fatal"


class BaselineFailureCategory(StrEnum):
    """Closed categories for baseline-ingest failures."""

    INVALID_REQUEST = "invalid_request"
    CANCELLED = "cancelled"
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


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineIngestRequest:
    """Describe one corpus-scoped baseline ingest attempt."""

    corpus_id: CorpusId
    discovery_request: DiscoveryRequest = field(repr=False)
    proposed_corpus_version: CorpusVersionId
    expected_corpus_version: CorpusVersionId | None
    parser_version: ParserVersion
    chunker_version: ChunkerVersion
    tokeniser_identity: TokeniserIdentity
    chunking_policy: ChunkingPolicy
    max_concurrent_documents: int
    configuration_identity: ConfigurationIdentity

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate identities, corpus ownership, and execution bounds."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId")
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

        if not isinstance(self.parser_version, ParserVersion):
            raise TypeError("parser_version must be a ParserVersion.")
        if not isinstance(self.chunker_version, ChunkerVersion):
            raise TypeError("chunker_version must be a ChunkerVersion.")
        if not isinstance(self.tokeniser_identity, TokeniserIdentity):
            raise TypeError("tokeniser_identity must be a TokeniserIdentity.")
        if not isinstance(self.chunking_policy, ChunkingPolicy):
            raise TypeError("chunking_policy must be a ChunkingPolicy.")
        if not isinstance(
            self.configuration_identity,
            ConfigurationIdentity,
        ):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")

        if not is_strict_integer(self.max_concurrent_documents):
            raise TypeError("max_concurrent_documents must be an integer.")
        if self.max_concurrent_documents <= 0:
            raise ValueError("max_concurrent_documents must be positive.")


@dataclass(frozen=True, slots=True)
class BaselineFailure:
    """Describe one payload-free baseline-ingest failure."""

    stage: BaselineStage
    category: BaselineFailureCategory
    scope: BaselineFailureScope
    severity: BaselineFailureSeverity
    retryable: bool

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the closed failure dimensions."""
        if not isinstance(self.stage, BaselineStage):
            raise TypeError("stage must be a BaselineStage.")
        if not isinstance(self.category, BaselineFailureCategory):
            raise TypeError("category must be a BaselineFailureCategory.")
        if not isinstance(self.scope, BaselineFailureScope):
            raise TypeError("scope must be a BaselineFailureScope.")
        if not isinstance(self.severity, BaselineFailureSeverity):
            raise TypeError("severity must be a BaselineFailureSeverity.")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be a boolean.")

    def key(self) -> tuple[str, str, str, str, int]:
        """Return the deterministic ordering key for this failure."""
        return (
            self.stage.value,
            self.category.value,
            self.scope.value,
            self.severity.value,
            int(self.retryable),
        )


@dataclass(frozen=True, slots=True)
class BaselineDocumentOutcome:
    """Summarise one document's baseline-ingest result."""

    ordinal: int
    kind: BaselineDocumentOutcomeKind
    document_id: DocumentId | None = None
    document_version_id: DocumentVersionId | None = None
    parse_disposition: BaselineStageDisposition = BaselineStageDisposition.NOT_ATTEMPTED
    chunk_disposition: BaselineStageDisposition = BaselineStageDisposition.NOT_ATTEMPTED
    failures: tuple[BaselineFailure, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate identity, stage, failure, and ordering invariants."""
        if not is_strict_integer(self.ordinal):
            raise TypeError("ordinal must be an integer.")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative.")

        if not isinstance(self.kind, BaselineDocumentOutcomeKind):
            raise TypeError("kind must be a BaselineDocumentOutcomeKind.")
        if self.document_id is not None and not isinstance(
            self.document_id,
            DocumentId,
        ):
            raise TypeError("document_id must be a DocumentId or None.")
        if self.document_version_id is not None and not isinstance(
            self.document_version_id,
            DocumentVersionId,
        ):
            raise TypeError("document_version_id must be a DocumentVersionId or None.")
        if self.document_version_id is not None and self.document_id is None:
            raise ValueError("document_version_id requires document_id.")

        if not isinstance(
            self.parse_disposition,
            BaselineStageDisposition,
        ):
            raise TypeError("parse_disposition must be a BaselineStageDisposition.")
        if not isinstance(
            self.chunk_disposition,
            BaselineStageDisposition,
        ):
            raise TypeError("chunk_disposition must be a BaselineStageDisposition.")

        if not isinstance(self.failures, tuple):
            raise TypeError("failures must be a tuple.")
        if any(not isinstance(failure, BaselineFailure) for failure in self.failures):
            raise TypeError("failures must contain BaselineFailure values.")
        if any(
            failure.scope is not BaselineFailureScope.DOCUMENT
            for failure in self.failures
        ):
            raise ValueError("document outcomes may contain only document failures.")
        failure_keys = tuple(failure.key() for failure in self.failures)
        if failure_keys != tuple(sorted(failure_keys)):
            raise ValueError("failures must be in canonical order.")

        if self.kind is BaselineDocumentOutcomeKind.FAILED:
            if not self.failures:
                raise ValueError("failed outcomes must contain at least one failure.")
        elif self.failures:
            raise ValueError("non-failed outcomes must not contain failures.")

        if self.kind is BaselineDocumentOutcomeKind.UNCHANGED_REUSED:
            if (
                self.parse_disposition is not BaselineStageDisposition.REUSED
                or self.chunk_disposition is not BaselineStageDisposition.REUSED
            ):
                raise ValueError(
                    "unchanged reuse requires reused parse and chunk stages."
                )

        if (
            self.parse_disposition is BaselineStageDisposition.FAILED
            or self.chunk_disposition is BaselineStageDisposition.FAILED
        ) and self.kind is not BaselineDocumentOutcomeKind.FAILED:
            raise ValueError("a failed stage requires a failed document outcome.")

        if self.kind in {
            BaselineDocumentOutcomeKind.AMBIGUOUS,
            BaselineDocumentOutcomeKind.CONFLICTING,
            BaselineDocumentOutcomeKind.UNSUPPORTED,
        } and (
            self.parse_disposition is not BaselineStageDisposition.NOT_ATTEMPTED
            or self.chunk_disposition is not BaselineStageDisposition.NOT_ATTEMPTED
        ):
            raise ValueError(
                "ambiguous, conflicting, and unsupported sources "
                "must not enter parse or chunk stages."
            )

    def key(self) -> tuple[int, str, str, str]:
        """Return the deterministic result-ordering key."""
        return (
            self.ordinal,
            self.kind.value,
            self.document_id.as_str() if self.document_id is not None else "",
            (
                self.document_version_id.as_str()
                if self.document_version_id is not None
                else ""
            ),
        )


_BASELINE_STAGE_ORDER = {stage: index for index, stage in enumerate(BaselineStage)}


def _validate_non_negative_integer(
    value: object,
    *,
    field_name: str,
) -> None:
    """Validate a strict non-negative integer field."""
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class BaselineStageCounts:
    """Count reuse, work, and failure for one baseline stage."""

    stage: BaselineStage
    reused: int = 0
    worked: int = 0
    failed: int = 0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the stage and its bounded counters."""
        if not isinstance(self.stage, BaselineStage):
            raise TypeError("stage must be a BaselineStage.")
        _validate_non_negative_integer(self.reused, field_name="reused")
        _validate_non_negative_integer(self.worked, field_name="worked")
        _validate_non_negative_integer(self.failed, field_name="failed")

    def key(self) -> int:
        """Return the canonical stage-ordering key."""
        return _BASELINE_STAGE_ORDER[self.stage]


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineIngestSummary:
    """Deterministic, payload-free result of one baseline ingest."""

    corpus_id: CorpusId
    proposed_corpus_version: CorpusVersionId
    expected_corpus_version: CorpusVersionId | None
    execution_status: BaselineExecutionStatus
    publication_status: BaselinePublicationStatus
    document_outcomes: tuple[BaselineDocumentOutcome, ...] = ()
    stage_counts: tuple[BaselineStageCounts, ...] = ()
    failures: tuple[BaselineFailure, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate ownership, ordering, and status-axis invariants."""
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
        if not isinstance(
            self.execution_status,
            BaselineExecutionStatus,
        ):
            raise TypeError("execution_status must be a BaselineExecutionStatus.")
        if not isinstance(
            self.publication_status,
            BaselinePublicationStatus,
        ):
            raise TypeError("publication_status must be a BaselinePublicationStatus.")

        self._validate_document_outcomes()
        self._validate_stage_counts()
        self._validate_request_failures()
        self._validate_status_axes()

    def _validate_document_outcomes(self) -> None:
        """Require typed, unique, canonically ordered document outcomes."""
        if not isinstance(self.document_outcomes, tuple):
            raise TypeError("document_outcomes must be a tuple.")
        if any(
            not isinstance(outcome, BaselineDocumentOutcome)
            for outcome in self.document_outcomes
        ):
            raise TypeError(
                "document_outcomes must contain BaselineDocumentOutcome values."
            )

        keys = tuple(outcome.key() for outcome in self.document_outcomes)
        if keys != tuple(sorted(keys)):
            raise ValueError("document_outcomes must be in canonical order.")

        ordinals = tuple(outcome.ordinal for outcome in self.document_outcomes)
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("document_outcomes must not contain duplicate ordinals.")

    def _validate_stage_counts(self) -> None:
        """Require one ordered count record per represented stage."""
        if not isinstance(self.stage_counts, tuple):
            raise TypeError("stage_counts must be a tuple.")
        if any(
            not isinstance(counts, BaselineStageCounts) for counts in self.stage_counts
        ):
            raise TypeError("stage_counts must contain BaselineStageCounts values.")

        keys = tuple(counts.key() for counts in self.stage_counts)
        if keys != tuple(sorted(keys)):
            raise ValueError("stage_counts must be in canonical order.")

        if len(keys) != len(set(keys)):
            raise ValueError("stage_counts must contain each stage at most once.")

    def _validate_request_failures(self) -> None:
        """Keep request failures separate from document outcomes."""
        if not isinstance(self.failures, tuple):
            raise TypeError("failures must be a tuple.")
        if any(not isinstance(failure, BaselineFailure) for failure in self.failures):
            raise TypeError("failures must contain BaselineFailure values.")
        if any(
            failure.scope is not BaselineFailureScope.REQUEST
            for failure in self.failures
        ):
            raise ValueError("summary failures must be request-scoped.")

        keys = tuple(failure.key() for failure in self.failures)
        if keys != tuple(sorted(keys)):
            raise ValueError("failures must be in canonical order.")

    def _validate_status_axes(self) -> None:
        """Prevent contradictory execution and publication combinations."""
        failed_documents = any(
            outcome.kind is BaselineDocumentOutcomeKind.FAILED
            for outcome in self.document_outcomes
        )

        if (
            self.execution_status is BaselineExecutionStatus.COMPLETED
            and failed_documents
        ):
            raise ValueError("completed execution must not contain failed documents.")

        if (
            self.execution_status is BaselineExecutionStatus.COMPLETED_WITH_FAILURES
            and not failed_documents
        ):
            raise ValueError("completed_with_failures requires a failed document.")

        if (
            self.execution_status
            in {
                BaselineExecutionStatus.FAILED,
                BaselineExecutionStatus.CONFLICT,
            }
            and not self.failures
        ):
            raise ValueError(
                "failed or conflicted execution requires a request failure."
            )

        if (
            self.execution_status
            in {
                BaselineExecutionStatus.FAILED,
                BaselineExecutionStatus.CONFLICT,
            }
            and self.publication_status is not BaselinePublicationStatus.NOT_PUBLISHED
        ):
            raise ValueError("failed or conflicted execution must not be published.")

        if self.publication_status in {
            BaselinePublicationStatus.COMMITTED,
            BaselinePublicationStatus.NO_CHANGE,
        } and self.execution_status in {
            BaselineExecutionStatus.FAILED,
            BaselineExecutionStatus.CONFLICT,
        }:
            raise ValueError(
                "failed or conflicted execution cannot have a publication."
            )
