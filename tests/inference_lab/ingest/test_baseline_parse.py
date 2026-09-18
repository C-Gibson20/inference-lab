"""Tests for the resumable baseline parse-stage executor."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    CURRENT_PARSER_VERSION,
    BaselineParseErrorCategory,
    BaselineParseExecutionError,
    BaselineParseRequest,
    BaselineStage,
    BaselineStageDisposition,
    ClassifiedSourceCandidate,
    DiscoveredSource,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    ParserRegistration,
    ParserRegistry,
    SourceClassification,
    SourceSupport,
    TraversalCandidate,
    VerifiedSourceSnapshot,
    execute_baseline_parse,
    hash_source_candidate,
    inventory_local_sources,
)
from inference_lab.ingest.baseline_checkpoints import BaselineWorkKey
from inference_lab.ingest.parser import ParseOutcome, ParserInput
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusDocumentVersionMembership,
    CorpusId,
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
from inference_lab.storage import ParsedRepresentation, SourceArtefact
from inference_lab.storage import ParseOutcome as StorageParseOutcome

from .fixtures import fixture_settings  # noqa: TID252

_CONFIGURATION = ConfigurationIdentity(DigestAlgorithm.SHA256, "a" * 64)


class _RaisingParser:
    """Inject an adapter exception to exercise parse-stage failure mapping."""

    def parse(self, source: ParserInput, *, settings: Settings) -> ParseOutcome:
        del source, settings
        raise RuntimeError("injected parser failure")


def _classification() -> SourceClassification:
    return SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
        classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
        parser_capability="text-native",
        encoding="utf-8",
    )


def _candidate(path: Path) -> TraversalCandidate:
    request = DiscoveryRequest(
        corpus_id=CorpusId.generate(),
        inputs=(DiscoveryInput(path, DiscoveryInputKind.FILE),),
    )
    return inventory_local_sources(request, excluded_paths=()).candidates[0]


@dataclass(frozen=True, slots=True)
class _Case:
    request: BaselineParseRequest
    repository: "_Repository"
    content: bytes


class _Repository:
    """Small repository double that stores only public parse aggregates."""

    def __init__(self, request: BaselineParseRequest) -> None:
        version = request.document_version
        self.membership = CorpusDocumentVersionMembership(
            corpus_id=request.source.corpus_id,
            document_id=version.document_id,
            document_version_id=version.document_version_id,
        )
        self.document_version = version
        self.parse_outcome: StorageParseOutcome | None = None
        self.representation: ParsedRepresentation | None = None
        self.publication_count = 0
        self.fail_publication = False

    def get_parse_outcome(
        self,
        corpus_id: CorpusId,
        parse_id: RecordId,
    ) -> StorageParseOutcome | None:
        assert corpus_id == self.membership.corpus_id
        if self.parse_outcome is not None:
            assert parse_id == self.parse_outcome.parse_id
        return self.parse_outcome

    def get_corpus_document_version_membership(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> CorpusDocumentVersionMembership | None:
        assert corpus_id == self.membership.corpus_id
        assert document_version_id == self.membership.document_version_id
        return self.membership

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        assert document_version_id == self.document_version.document_version_id
        return self.document_version

    def get_source_artefact(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        del corpus_id, document_version_id
        return None

    def get_parsed_representation(
        self,
        corpus_id: CorpusId,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        assert corpus_id == self.membership.corpus_id
        if self.representation is not None:
            assert record_id == self.representation.record_id
        return self.representation

    def persist_source_artefact(
        self,
        corpus_id: CorpusId,
        artefact: object,
        chunks: Iterable[bytes],
        *,
        max_source_bytes: int,
        parse_outcome: StorageParseOutcome | None = None,
        parsed_representation: ParsedRepresentation | None = None,
    ) -> None:
        del artefact, chunks, max_source_bytes
        assert corpus_id == self.membership.corpus_id
        if self.fail_publication:
            raise RuntimeError("injected publication failure")
        assert parse_outcome is not None
        self.parse_outcome = parse_outcome
        self.representation = parsed_representation
        self.publication_count += 1


def _case(tmp_path: Path, *, content: bytes = b"one paragraph") -> _Case:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "private-parse-source.txt"
    path.write_bytes(content)
    candidate = _candidate(path)
    classified = ClassifiedSourceCandidate(candidate, _classification())
    snapshot = hash_source_candidate(classified, max_source_bytes=1024)
    assert isinstance(snapshot, VerifiedSourceSnapshot)
    document_id = DocumentId.generate()
    version = DocumentVersion(
        document_version_id=DocumentVersionId.generate(),
        document_id=document_id,
        content_hash=snapshot.content_hash,
    )
    source = DiscoveredSource(
        ordinal=candidate.ordinal,
        corpus_id=CorpusId.generate(),
        policy=CURRENT_DISCOVERY_POLICY,
        source_observation_id=SourceObservationId.generate(),
        snapshot=snapshot,
        external_identifiers=(),
        family_candidates=(),
        classification=classified.classification,
        identity_outcome=IdentityResolutionOutcome(
            kind=ResolutionKind.NEW_DOCUMENT,
            reason=ResolutionReason.NO_MATCH,
            document=DocumentIdentity(document_id),
            version=version,
        ),
    )
    request = BaselineParseRequest(
        work_key=BaselineWorkKey(
            proposed_corpus_version=CorpusVersionId.generate(),
            document_version_id=version.document_version_id,
            stage=BaselineStage.PARSE,
            configuration_identity=_CONFIGURATION,
        ),
        source=source,
        parser_version=CURRENT_PARSER_VERSION,
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
    )
    repository = _Repository(request)
    return _Case(request=request, repository=repository, content=content)


def test_parse_stage_publishes_then_reuses_reconstructed_output(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    settings = fixture_settings(max_source_bytes=1024)

    with capture_logs() as logs:
        published = execute_baseline_parse(
            case.repository,
            case.request,
            settings=settings,
        )
        reused = execute_baseline_parse(
            case.repository,
            case.request,
            settings=settings,
        )

    assert published.disposition is BaselineStageDisposition.COMPLETED
    assert reused.disposition is BaselineStageDisposition.REUSED
    assert published.reconstructed == reused.reconstructed
    assert published.reconstructed.outcome.document is not None
    assert published.parse_id == case.request.parse_id
    assert published.parsed_record_id == case.request.parsed_record_id
    assert case.repository.publication_count == 1
    assert [record["event"] for record in logs] == [
        "baseline_parse_started",
        "source_reacquired",
        "document_started",
        "document_completed",
        "parse_persistence_started",
        "parse_persistence_completed",
        "baseline_parse_completed",
        "baseline_parse_started",
        "baseline_parse_reused",
    ]
    rendered = repr(logs)
    assert "private-parse-source.txt" not in rendered
    assert case.content.decode() not in rendered


def test_parse_stage_reuses_a_persisted_fatal_outcome(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, content=b"\xff")
    failed_request = BaselineParseRequest(
        work_key=case.request.work_key,
        source=case.request.source,
        parser_version=case.request.parser_version,
        parse_id=case.request.parse_id,
        parsed_record_id=None,
    )

    first = execute_baseline_parse(
        case.repository,
        failed_request,
        settings=fixture_settings(max_source_bytes=1024),
    )
    second = execute_baseline_parse(
        case.repository,
        failed_request,
        settings=fixture_settings(max_source_bytes=1024),
    )

    assert first.reconstructed.outcome.status == "failed"
    assert second.disposition is BaselineStageDisposition.REUSED
    assert case.repository.publication_count == 1


def test_parse_stage_maps_source_parser_and_publication_failures(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    case.request.source.snapshot.locator.path.write_bytes(b"changed")
    with pytest.raises(BaselineParseExecutionError) as source_error:
        execute_baseline_parse(
            case.repository,
            case.request,
            settings=fixture_settings(max_source_bytes=1024),
        )
    assert (
        source_error.value.category
        is BaselineParseErrorCategory.SOURCE_ACQUISITION_FAILED
    )

    case = _case(tmp_path / "parser")
    with pytest.raises(BaselineParseExecutionError) as parser_error:
        execute_baseline_parse(
            case.repository,
            case.request,
            settings=fixture_settings(max_source_bytes=1024),
            parser_registry=ParserRegistry(registrations=()),
        )
    assert parser_error.value.category is BaselineParseErrorCategory.PARSER_UNAVAILABLE

    case = _case(tmp_path / "parser-exception")
    raising_registry = ParserRegistry(
        registrations=(
            ParserRegistration(
                capability="text-native",
                parser_version=CURRENT_PARSER_VERSION,
                parser=_RaisingParser(),
            ),
        ),
    )
    with pytest.raises(BaselineParseExecutionError) as parser_failure:
        execute_baseline_parse(
            case.repository,
            case.request,
            settings=fixture_settings(max_source_bytes=1024),
            parser_registry=raising_registry,
        )
    assert parser_failure.value.category is BaselineParseErrorCategory.PARSE_FAILED

    case = _case(tmp_path / "publication")
    case.repository.fail_publication = True
    with pytest.raises(BaselineParseExecutionError) as publication_error:
        execute_baseline_parse(
            case.repository,
            case.request,
            settings=fixture_settings(max_source_bytes=1024),
        )
    assert (
        publication_error.value.category
        is BaselineParseErrorCategory.PARSE_PERSISTENCE_FAILED
    )


def test_parse_stage_rejects_incompatible_persisted_replay(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    execute_baseline_parse(
        case.repository,
        case.request,
        settings=fixture_settings(max_source_bytes=1024),
    )
    assert case.repository.parse_outcome is not None
    case.repository.parse_outcome = StorageParseOutcome(
        parse_id=case.repository.parse_outcome.parse_id,
        document_id=case.repository.parse_outcome.document_id,
        document_version_id=case.repository.parse_outcome.document_version_id,
        status=case.repository.parse_outcome.status,
        parser=case.repository.parse_outcome.parser,
        normalisation_version="2.0.0",
        parsed_record_id=case.repository.parse_outcome.parsed_record_id,
        failures=case.repository.parse_outcome.failures,
    )

    with pytest.raises(BaselineParseExecutionError) as error:
        execute_baseline_parse(
            case.repository,
            case.request,
            settings=fixture_settings(max_source_bytes=1024),
        )

    assert error.value.category is BaselineParseErrorCategory.CORRUPT_STATE
