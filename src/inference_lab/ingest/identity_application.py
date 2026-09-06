"""Atomic persistence plans for resolved discovery identity outcomes."""

from dataclasses import dataclass

from structlog.contextvars import bound_contextvars

from inference_lab.ingest.discovery import DiscoveredSource
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    DocumentIdentity,
    DocumentVersion,
    IdentityResolutionOutcome,
    ResolutionKind,
    ResolutionReason,
    SourceObservationState,
)
from inference_lab.storage import (
    MetadataTransaction,
    SourceFileIdentityHint,
    SourceFingerprint,
    SourceLocatorRevision,
    StorageConflictError,
)


@dataclass(frozen=True, slots=True)
class IdentityApplicationResult:
    """Report a resolved source and its downstream parse eligibility."""

    source: DiscoveredSource
    parse_eligible: bool

    def __post_init__(self) -> None:
        """Validate source type and outcome-owned eligibility."""
        if not isinstance(self.source, DiscoveredSource):
            raise TypeError("source must be a DiscoveredSource.")
        if not isinstance(self.parse_eligible, bool):
            raise TypeError("parse_eligible must be a boolean.")
        expected = self.source.identity_outcome.kind in {
            ResolutionKind.NEW_VERSION,
            ResolutionKind.NEW_DOCUMENT,
            ResolutionKind.DUPLICATE,
        }
        if self.parse_eligible is not expected:
            raise ValueError("parse eligibility does not match the identity outcome.")


def _assigned_identity(
    outcome: IdentityResolutionOutcome,
) -> tuple[DocumentIdentity, DocumentVersion]:
    document = outcome.document
    version = outcome.version
    if document is None or version is None:
        raise ValueError(
            "successful identity outcome must assign a document and version."
        )
    return document, version


def _membership(
    corpus_id: CorpusId,
    version: DocumentVersion,
) -> CorpusDocumentVersionMembership:
    return CorpusDocumentVersionMembership(
        corpus_id=corpus_id,
        document_id=version.document_id,
        document_version_id=version.document_version_id,
    )


def _initial_observation(
    source: DiscoveredSource,
    corpus_id: CorpusId,
    version: DocumentVersion,
) -> SourceObservationState:
    return SourceObservationState(
        corpus_id=corpus_id,
        source_observation_id=source.source_observation_id,
        observation_revision=1,
        document_id=version.document_id,
        document_version_id=version.document_version_id,
    )


def _source_identity_hint(
    source: DiscoveredSource,
) -> SourceFileIdentityHint | None:
    hint = source.snapshot.file_evidence.opened_after_read.identity_hint
    if hint is None:
        return None
    return SourceFileIdentityHint(hint.device_id, hint.file_id)


def _source_fingerprint(source: DiscoveredSource) -> SourceFingerprint:
    evidence = source.snapshot.file_evidence.opened_after_read
    return SourceFingerprint(
        content_hash=source.snapshot.content_hash,
        byte_count=source.snapshot.byte_count,
        modified_time_ns=evidence.modified_time_ns,
    )


def _write_locator(
    source: DiscoveredSource,
    *,
    corpus_id: CorpusId,
    transaction: MetadataTransaction,
) -> None:
    current = transaction.get_source_locator(
        corpus_id,
        source.source_observation_id,
    )
    identity_hint = _source_identity_hint(source)
    fingerprint = _source_fingerprint(source)
    if current is None:
        transaction.add_source_locator(
            SourceLocatorRevision(
                corpus_id=corpus_id,
                source_observation_id=source.source_observation_id,
                locator_revision=1,
                path=source.snapshot.locator.path,
                identity_hint=identity_hint,
                fingerprint=fingerprint,
            )
        )
        return
    if (
        current.path == source.snapshot.locator.path
        and current.identity_hint == identity_hint
        and current.fingerprint == fingerprint
    ):
        return
    transaction.advance_source_locator(
        SourceLocatorRevision(
            corpus_id=corpus_id,
            source_observation_id=source.source_observation_id,
            locator_revision=current.locator_revision + 1,
            path=source.snapshot.locator.path,
            identity_hint=identity_hint,
            fingerprint=fingerprint,
        )
    )


def _apply_unchanged(
    source: DiscoveredSource,
    *,
    corpus_id: CorpusId,
    transaction: MetadataTransaction,
) -> None:
    outcome = source.identity_outcome
    _, version = _assigned_identity(outcome)
    current = transaction.get_source_observation(
        corpus_id,
        source.source_observation_id,
    )
    if current is None or current.document_version_id != version.document_version_id:
        raise StorageConflictError()
    if outcome.identifier_assertions_to_add:
        transaction.add_external_identifier_assertions(
            corpus_id,
            outcome.identifier_assertions_to_add,
        )
    _write_locator(source, corpus_id=corpus_id, transaction=transaction)


def _apply_new_version(
    source: DiscoveredSource,
    *,
    corpus_id: CorpusId,
    transaction: MetadataTransaction,
) -> None:
    outcome = source.identity_outcome
    _, version = _assigned_identity(outcome)
    transaction.add_document_version(version)
    transaction.add_corpus_document_version_membership(_membership(corpus_id, version))
    if outcome.reason is ResolutionReason.SOURCE_CONTINUITY_MATCH:
        transaction.advance_source_observation(
            corpus_id,
            source.source_observation_id,
            expected_version_id=outcome.matched_version_ids[0],
            new_version_id=version.document_version_id,
        )
    else:
        transaction.add_source_observation(
            _initial_observation(source, corpus_id, version)
        )
    _write_locator(source, corpus_id=corpus_id, transaction=transaction)


def _apply_new_identity(
    source: DiscoveredSource,
    *,
    corpus_id: CorpusId,
    transaction: MetadataTransaction,
) -> None:
    outcome = source.identity_outcome
    document, version = _assigned_identity(outcome)
    transaction.add_document(document)
    transaction.add_document_version(version)
    transaction.add_corpus_document_version_membership(_membership(corpus_id, version))
    transaction.add_source_observation(_initial_observation(source, corpus_id, version))
    for relationship in outcome.duplicate_relationships:
        transaction.add_duplicate_relationship(relationship)
    _write_locator(source, corpus_id=corpus_id, transaction=transaction)


def _apply_with_bound_corpus(
    source: DiscoveredSource,
    *,
    corpus_id: CorpusId,
    transaction: MetadataTransaction,
) -> IdentityApplicationResult:
    kind = source.identity_outcome.kind
    if kind in {ResolutionKind.AMBIGUOUS, ResolutionKind.CONFLICT}:
        return IdentityApplicationResult(source=source, parse_eligible=False)
    if kind is ResolutionKind.UNCHANGED:
        _apply_unchanged(source, corpus_id=corpus_id, transaction=transaction)
        return IdentityApplicationResult(source=source, parse_eligible=False)
    if kind is ResolutionKind.NEW_VERSION:
        _apply_new_version(source, corpus_id=corpus_id, transaction=transaction)
    else:
        _apply_new_identity(source, corpus_id=corpus_id, transaction=transaction)
    return IdentityApplicationResult(source=source, parse_eligible=True)


def apply_discovery_identity_outcome(
    source: DiscoveredSource,
    *,
    corpus_id: CorpusId,
    transaction: MetadataTransaction,
) -> IdentityApplicationResult:
    """Apply one complete outcome within a caller-owned transaction."""
    if not isinstance(source, DiscoveredSource):
        raise TypeError("source must be a DiscoveredSource.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")

    with bound_contextvars(corpus_id=corpus_id.as_str()):
        return _apply_with_bound_corpus(
            source,
            corpus_id=corpus_id,
            transaction=transaction,
        )
