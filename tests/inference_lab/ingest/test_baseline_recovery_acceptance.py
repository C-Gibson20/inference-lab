"""ING-05/STOR-03 cold-process recovery and privacy acceptance."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_DISCOVERY_POLICY,
    CURRENT_PARSER_VERSION,
    BaselineChunkErrorCategory,
    BaselineChunkExecutionError,
    BaselineChunkRequest,
    BaselineParseRequest,
    BaselineParseResult,
    BaselineStage,
    BaselineStageDisposition,
    ChunkingPolicy,
    ChunkingRequest,
    ClassifiedSourceCandidate,
    DiscoveredSource,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    FakeChunkTokeniser,
    ParsedDocument,
    SourceClassification,
    SourceSupport,
    VerifiedSourceSnapshot,
    build_chunk_set,
    execute_baseline_chunk,
    execute_baseline_parse,
    hash_source_candidate,
    inventory_local_sources,
)
from inference_lab.ingest.baseline_checkpoints import BaselineWorkKey
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    IdentityResolutionOutcome,
    RecordId,
    ResolutionKind,
    ResolutionReason,
    SourceObservationId,
)
from inference_lab.settings import Settings
from inference_lab.storage import open_metadata_repository
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import fixture_settings  # noqa: TID252

_PARSE_CONFIGURATION = ConfigurationIdentity(DigestAlgorithm.SHA256, "a" * 64)
_PRIVATE_PATH = "private-baseline-recovery-source.txt"
_PRIVATE_CONTENT = "private baseline recovery canary"


@dataclass(frozen=True, slots=True)
class _Case:
    database_path: Path
    source_dir: Path
    settings: Settings
    parse_request: BaselineParseRequest
    content: bytes


def _classification() -> SourceClassification:
    return SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
        classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
        parser_capability="text-native",
        encoding="utf-8",
    )


def _case(tmp_path: Path) -> _Case:
    corpus_id = CorpusId.generate()
    source_path = tmp_path / _PRIVATE_PATH
    content = _PRIVATE_CONTENT.encode("utf-8")
    source_path.write_bytes(content)
    candidate = inventory_local_sources(
        DiscoveryRequest(
            corpus_id=corpus_id,
            inputs=(DiscoveryInput(source_path, DiscoveryInputKind.FILE),),
        ),
        excluded_paths=(),
    ).candidates[0]
    classified = ClassifiedSourceCandidate(candidate, _classification())
    snapshot = hash_source_candidate(classified, max_source_bytes=1024)
    assert isinstance(snapshot, VerifiedSourceSnapshot)
    document = DocumentIdentity(DocumentId.generate())
    version = DocumentVersion(
        document_version_id=DocumentVersionId.generate(),
        document_id=document.document_id,
        content_hash=snapshot.content_hash,
    )
    source = DiscoveredSource(
        ordinal=candidate.ordinal,
        corpus_id=corpus_id,
        policy=CURRENT_DISCOVERY_POLICY,
        source_observation_id=SourceObservationId.generate(),
        snapshot=snapshot,
        external_identifiers=(),
        family_candidates=(),
        classification=classified.classification,
        identity_outcome=IdentityResolutionOutcome(
            kind=ResolutionKind.NEW_DOCUMENT,
            reason=ResolutionReason.NO_MATCH,
            document=document,
            version=version,
        ),
    )
    return _Case(
        database_path=tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "managed-source",
        settings=fixture_settings(max_source_bytes=1024),
        parse_request=BaselineParseRequest(
            work_key=BaselineWorkKey(
                proposed_corpus_version=CorpusVersionId.generate(),
                document_version_id=version.document_version_id,
                stage=BaselineStage.PARSE,
                configuration_identity=_PARSE_CONFIGURATION,
            ),
            source=source,
            parser_version=CURRENT_PARSER_VERSION,
            parse_id=RecordId.generate(),
            parsed_record_id=RecordId.generate(),
        ),
        content=content,
    )


def _open_and_seed(case: _Case) -> SQLiteMetadataRepository:
    repository = cast(
        SQLiteMetadataRepository,
        open_metadata_repository(
            case.database_path,
            source_dir=case.source_dir,
            apply_migrations=True,
        ),
    )
    request = case.parse_request
    version = request.document_version
    with repository.transaction() as transaction:
        transaction.add_corpus(
            CorpusMetadata(request.source.corpus_id, "Baseline recovery corpus")
        )
        transaction.add_document(DocumentIdentity(version.document_id))
        transaction.add_document_version(version)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                request.source.corpus_id,
                version.document_id,
                version.document_version_id,
            )
        )
    return repository


def _chunk_request(
    parse_result: BaselineParseResult,
    case: _Case,
) -> BaselineChunkRequest:
    document = parse_result.reconstructed.outcome.document
    assert isinstance(document, ParsedDocument)
    tokeniser = FakeChunkTokeniser()
    policy = ChunkingPolicy(min_tokens=1, max_tokens=8)
    expected = build_chunk_set(
        ChunkingRequest(
            corpus_id=case.parse_request.source.corpus_id,
            parsed_document=document,
            tokeniser=tokeniser,
            policy=policy,
        )
    )
    return BaselineChunkRequest(
        corpus_id=case.parse_request.source.corpus_id,
        work_key=BaselineWorkKey(
            proposed_corpus_version=parse_result.work_key.proposed_corpus_version,
            document_version_id=document.document_version_id,
            stage=BaselineStage.CHUNK,
            configuration_identity=expected.configuration_identity,
        ),
        parse_result=parse_result,
        tokeniser=tokeniser,
        policy=policy,
        chunker_version=CURRENT_CHUNKER_VERSION,
    )


def test_baseline_stages_reuse_exact_outputs_after_cold_restart(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    first = _open_and_seed(case)
    try:
        parsed = execute_baseline_parse(
            first,
            case.parse_request,
            settings=case.settings,
        )
        chunked = execute_baseline_chunk(first, _chunk_request(parsed, case))
        assert parsed.disposition is BaselineStageDisposition.COMPLETED
        assert chunked.disposition is BaselineStageDisposition.COMPLETED
    finally:
        first.close()

    second = _open_and_seed_without_mutation(case)
    try:
        with capture_logs() as logs:
            parsed_replay = execute_baseline_parse(
                second,
                case.parse_request,
                settings=case.settings,
            )
            chunked_replay = execute_baseline_chunk(
                second,
                _chunk_request(parsed_replay, case),
            )

        assert parsed_replay.disposition is BaselineStageDisposition.REUSED
        assert chunked_replay.disposition is BaselineStageDisposition.REUSED
        assert parsed_replay.reconstructed.outcome.document is not None
        assert chunked_replay.chunk_set.chunks
        events = [record["event"] for record in logs]
        assert "baseline_parse_reused" in events
        assert "baseline_chunk_reused" in events
        rendered = repr(logs)
        assert _PRIVATE_PATH not in rendered
        assert _PRIVATE_CONTENT not in rendered
        assert case.content.decode("utf-8") not in rendered
    finally:
        second.close()


def _open_and_seed_without_mutation(case: _Case) -> SQLiteMetadataRepository:
    return cast(
        SQLiteMetadataRepository,
        open_metadata_repository(
            case.database_path,
            source_dir=case.source_dir,
            apply_migrations=True,
        ),
    )


def test_recovery_acceptance_fails_closed_after_durable_chunk_corruption(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    repository = _open_and_seed(case)
    try:
        parsed = execute_baseline_parse(
            repository,
            case.parse_request,
            settings=case.settings,
        )
        execute_baseline_chunk(repository, _chunk_request(parsed, case))
    finally:
        repository.close()

    connection = sqlite3.connect(case.database_path)
    try:
        connection.execute(
            "UPDATE chunk_sets SET canonical_payload = ?",
            (b'{"corrupt":true}',),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = _open_and_seed_without_mutation(case)
    try:
        parsed_replay = execute_baseline_parse(
            reopened,
            case.parse_request,
            settings=case.settings,
        )
        with pytest.raises(BaselineChunkExecutionError) as error:
            execute_baseline_chunk(reopened, _chunk_request(parsed_replay, case))

        assert (
            error.value.category is BaselineChunkErrorCategory.CHUNK_PERSISTENCE_FAILED
        )
        assert "corrupt" not in str(error.value)
    finally:
        reopened.close()
