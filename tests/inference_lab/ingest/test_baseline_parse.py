"""Tests for the resumable baseline parse-stage executor."""

from dataclasses import dataclass
from datetime import UTC, datetime
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
    BaselineProfilingContext,
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
from inference_lab.profiling import FailureEvent, StageTiming
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


class _CollectingSink:
    """Collect shared profiling records without rendering them."""

    def __init__(self) -> None:
        self.records: list[object] = []

    def emit(self, record: object) -> None:
        self.records.append(record)


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
    assert [
        record["event"]
        for record in logs
        if record["event"] not in {"stage_started", "stage_completed"}
    ] == [
        "source_reacquired",
        "document_started",
        "document_completed",
        "baseline_parse_reused",
    ]
    rendered = repr(logs)
    assert "private-parse-source.txt" not in rendered
    assert case.content.decode() not in rendered


def test_parse_stage_profiles_each_attempt_and_replay_without_payloads(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    sink = _CollectingSink()
    clock_values = iter(range(0, 80_000_000, 10_000_000))
    profiling = BaselineProfilingContext.create(
        run_id="run-parse",
        sink=sink,
        clock=lambda: next(clock_values),
        observed_at_factory=lambda: datetime(2026, 9, 18, tzinfo=UTC),
    )

    execute_baseline_parse(
        case.repository,
        case.request,
        settings=fixture_settings(max_source_bytes=1024),
        profiling=profiling,
    )

    timings = [record for record in sink.records if isinstance(record, StageTiming)]
    assert [
        (record.event, record.pipeline_stage, record.status) for record in timings
    ] == [
        ("stage_started", "parse_persistence", "started"),
        ("stage_completed", "parse_persistence", "completed"),
        ("stage_started", "source_acquisition", "started"),
        ("stage_completed", "source_acquisition", "completed"),
        ("stage_started", "parse", "started"),
        ("stage_completed", "parse", "completed"),
        ("stage_started", "parse_persistence", "started"),
        ("stage_completed", "parse_persistence", "completed"),
    ]
    assert [record.duration_ms for record in timings[1::2]] == [10.0] * 4
    assert all(record.run_id == "run-parse" for record in timings)
    assert all(record.document_id is not None for record in timings)
    assert len({record.stage_id for record in timings}) == 4
    assert not [record for record in sink.records if isinstance(record, FailureEvent)]
    assert "private-parse-source.txt" not in repr(sink.records)
    assert case.content.decode() not in repr(sink.records)


def test_parse_stage_maps_source_failure_to_one_safe_failure_record(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    case.request.source.snapshot.locator.path.write_bytes(b"changed")
    sink = _CollectingSink()
    clock_values = iter(range(0, 40_000_000, 10_000_000))
    profiling = BaselineProfilingContext.create(
        run_id="run-parse-failure",
        sink=sink,
        clock=lambda: next(clock_values),
        observed_at_factory=lambda: datetime(2026, 9, 18, tzinfo=UTC),
    )

    with pytest.raises(BaselineParseExecutionError):
        execute_baseline_parse(
            case.repository,
            case.request,
            settings=fixture_settings(max_source_bytes=1024),
            profiling=profiling,
        )

    timings = [record for record in sink.records if isinstance(record, StageTiming)]
    failures = [record for record in sink.records if isinstance(record, FailureEvent)]
    assert [
        (record.pipeline_stage, record.status)
        for record in timings
        if record.event == "stage_completed"
    ] == [
        ("parse_persistence", "completed"),
        ("source_acquisition", "failed"),
    ]
    assert len(failures) == 1
    assert failures[0].failure_category == "dependency"
    assert failures[0].failure_code == "source_acquisition_failed"
    assert "changed" not in repr(sink.records)


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
