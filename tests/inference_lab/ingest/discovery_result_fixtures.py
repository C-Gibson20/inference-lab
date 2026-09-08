"""Shared builders for discovery-result tests."""

from pathlib import Path

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    ClassificationInventory,
    DiscoveredSource,
    DiscoveryFailure,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    FileIdentityHint,
    FileStatEvidence,
    IdentityApplicationResult,
    SourceClassification,
    SourceClassifierRegistry,
    SourceClassifierRule,
    SourceLocator,
    SourceSupport,
    StableFileEvidence,
    TraversalCandidate,
    TraversalInventory,
    VerifiedSourceSnapshot,
    classify_source_inventory,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    HashAlgorithm,
    IdentityResolutionOutcome,
    ResolutionKind,
    ResolutionReason,
    SourceObservationId,
)

_DIGEST = "b" * 64
_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")


def _classification() -> SourceClassification:
    return SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
        classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
        parser_capability="text-native",
        encoding="utf-8",
    )


def _custom_registry() -> SourceClassifierRegistry:
    classification = SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name="custom-classifier",
        classifier_version="2.0.0",
        parser_capability="text-native",
        encoding="utf-8",
    )
    return SourceClassifierRegistry(
        classifier_name="custom-classifier",
        classifier_version="2.0.0",
        rules=(SourceClassifierRule(".txt", classification),),
    )


def _snapshot(ordinal: int) -> VerifiedSourceSnapshot:
    classification = _classification()
    locator = SourceLocator(
        path=Path("private") / f"source-{ordinal}.txt",
        request_root_ordinal=0,
        relative_path=Path(f"source-{ordinal}.txt"),
    )
    stat = FileStatEvidence(
        byte_count=1,
        modified_time_ns=ordinal + 1,
        identity_hint=FileIdentityHint(device_id=1, file_id=ordinal + 1),
    )
    return VerifiedSourceSnapshot(
        locator=locator,
        media_type=classification.media_type,
        encoding=classification.encoding,
        byte_count=1,
        content_hash=ContentHash(HashAlgorithm.SHA256, _DIGEST),
        file_evidence=StableFileEvidence(stat, stat, stat, stat),
    )


def _assigned_values() -> tuple[DocumentIdentity, DocumentVersion]:
    document_id = DocumentId.generate()
    version = DocumentVersion(
        document_version_id=DocumentVersionId.generate(),
        document_id=document_id,
        content_hash=ContentHash(HashAlgorithm.SHA256, _DIGEST),
    )
    return DocumentIdentity(document_id), version


def _outcome(kind: ResolutionKind) -> IdentityResolutionOutcome:
    document, version = _assigned_values()
    if kind is ResolutionKind.UNCHANGED:
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.EXACT_HASH_MATCH,
            document=document,
            version=version,
            matched_version_ids=(version.document_version_id,),
        )
    if kind is ResolutionKind.NEW_VERSION:
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.SOURCE_CONTINUITY_MATCH,
            document=document,
            version=version,
            matched_version_ids=(DocumentVersionId.generate(),),
        )
    if kind is ResolutionKind.NEW_DOCUMENT:
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.NO_MATCH,
            document=document,
            version=version,
        )
    if kind is ResolutionKind.DUPLICATE:
        matched_id = DocumentVersionId.generate()
        first_id, second_id = sorted(
            (version.document_version_id, matched_id),
            key=DocumentVersionId.as_str,
        )
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.DUPLICATE_HASH_MATCH,
            document=document,
            version=version,
            matched_version_ids=(matched_id,),
            duplicate_relationships=(
                DuplicateRelationship(
                    first_version_id=first_id,
                    second_version_id=second_id,
                    duplicate_class=DuplicateClass.BYTE_IDENTICAL,
                ),
            ),
        )
    if kind is ResolutionKind.AMBIGUOUS:
        matched_ids = tuple(
            sorted(
                (DocumentVersionId.generate(), DocumentVersionId.generate()),
                key=DocumentVersionId.as_str,
            )
        )
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.MULTIPLE_CANDIDATES,
            matched_version_ids=matched_ids,
        )
    return IdentityResolutionOutcome(
        kind=ResolutionKind.CONFLICT,
        reason=ResolutionReason.CONFLICTING_EVIDENCE,
    )


def _source(
    ordinal: int,
    kind: ResolutionKind,
    *,
    corpus_id: CorpusId | None = None,
) -> DiscoveredSource:
    return DiscoveredSource(
        ordinal=ordinal,
        corpus_id=_CORPUS_ID if corpus_id is None else corpus_id,
        policy=CURRENT_DISCOVERY_POLICY,
        source_observation_id=SourceObservationId.generate(),
        snapshot=_snapshot(ordinal),
        external_identifiers=(),
        family_candidates=(),
        classification=_classification(),
        identity_outcome=_outcome(kind),
    )


def _request() -> DiscoveryRequest:
    return DiscoveryRequest(
        corpus_id=CorpusId.generate(),
        inputs=(DiscoveryInput(Path("private"), DiscoveryInputKind.DIRECTORY),),
    )


def _applied_source(
    ordinal: int,
    kind: ResolutionKind,
    *,
    corpus_id: CorpusId,
) -> IdentityApplicationResult:
    return IdentityApplicationResult(
        source=_source(ordinal, kind, corpus_id=corpus_id),
        created_document_version=kind
        in {
            ResolutionKind.NEW_VERSION,
            ResolutionKind.NEW_DOCUMENT,
            ResolutionKind.DUPLICATE,
        },
    )


def _classified_inventory(
    corpus_id: CorpusId,
    *,
    supported_ordinals: tuple[int, ...] = (),
    unknown_ordinals: tuple[int, ...] = (),
    traversal_failures: tuple[DiscoveryFailure, ...] = (),
    registry: SourceClassifierRegistry | None = None,
) -> ClassificationInventory:
    candidates = tuple(
        sorted(
            (
                TraversalCandidate(
                    ordinal=ordinal,
                    locator=_snapshot(ordinal).locator,
                    stat_evidence=_snapshot(ordinal).file_evidence.pre_open,
                )
                for ordinal in supported_ordinals
            ),
            key=lambda candidate: candidate.ordinal,
        )
    ) + tuple(
        TraversalCandidate(
            ordinal=ordinal,
            locator=SourceLocator(
                path=Path("private") / f"source-{ordinal}.bin",
                request_root_ordinal=0,
                relative_path=Path(f"source-{ordinal}.bin"),
            ),
            stat_evidence=_snapshot(ordinal).file_evidence.pre_open,
        )
        for ordinal in unknown_ordinals
    )
    candidates = tuple(sorted(candidates, key=lambda candidate: candidate.ordinal))
    traversal = TraversalInventory(
        corpus_id=corpus_id,
        root_count=1,
        candidates=candidates,
        failures=traversal_failures,
        complete_root_ordinals=(0,) if not traversal_failures else (),
    )
    if registry is None:
        return classify_source_inventory(traversal)
    return classify_source_inventory(traversal, registry=registry)
