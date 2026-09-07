"""Cold-process restart acceptance for persisted parser outcomes."""

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParseOutcome,
    ParsePersistenceRequest,
    ParserFailure,
    TextNativeParser,
    persist_parse_request,
    serialise_parsed_document,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
    HashAlgorithm,
    RecordId,
)
from inference_lab.storage import MetadataRepository, open_metadata_repository

from .fixtures import fixture_settings, fixture_source  # noqa: TID252

_CORPUS_ID = CorpusId("b0000000-0000-4000-8000-000000000001")
_READER = """
import json
import sys
from pathlib import Path

from inference_lab.ingest import reconstruct_parse_outcome, serialise_parsed_document
from inference_lab.provenance import CorpusId, DocumentVersionId, RecordId
from inference_lab.storage import open_metadata_repository

database_path = Path(sys.argv[1])
source_dir = Path(sys.argv[2])
result_path = Path(sys.argv[3])
corpus_id = CorpusId.from_str(sys.argv[4])
document_version_id = DocumentVersionId.from_str(sys.argv[5])
parsed_record_id = RecordId.from_str(sys.argv[6]) if sys.argv[6] != "-" else None
parse_id = RecordId.from_str(sys.argv[7])

repository = open_metadata_repository(database_path, source_dir=source_dir)
try:
    source_bytes = repository.read_source_bytes(corpus_id, document_version_id)
    storage_outcome = repository.get_parse_outcome(corpus_id, parse_id)
    representation = (
        repository.get_parsed_representation(corpus_id, parsed_record_id)
        if parsed_record_id is not None
        else None
    )
    assert source_bytes is not None
    assert storage_outcome is not None
    reconstructed = reconstruct_parse_outcome(storage_outcome, representation)
    document = reconstructed.outcome.document
    payload = {
        "source_bytes_hex": source_bytes.hex(),
        "parser_version": {
            "implementation": reconstructed.parser_version.implementation,
            "contract_version": reconstructed.parser_version.contract_version,
            "normalisation_version": reconstructed.parser_version.normalisation_version,
        },
        "outcome": {
            "status": reconstructed.outcome.status,
            "parsed_record_id": (
                storage_outcome.parsed_record_id.as_str()
                if storage_outcome.parsed_record_id is not None
                else None
            ),
            "failures": [
                {
                    "category": failure.category,
                    "severity": failure.severity,
                    "stage": failure.stage,
                    "unit_kind": failure.unit_kind,
                    "unit_index": failure.unit_index,
                }
                for failure in reconstructed.outcome.failures
            ],
        },
        "document_serialised_hex": (
            serialise_parsed_document(document).hex()
            if document is not None
            else None
        ),
    }
finally:
    repository.close()

result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
"""


@dataclass(frozen=True, slots=True)
class _Case:
    request: ParsePersistenceRequest
    document_serialised: bytes | None
    source_bytes: bytes


def _successful_request() -> ParsePersistenceRequest:
    source = fixture_source("basic.txt")
    outcome = TextNativeParser().parse(source, settings=fixture_settings())
    return ParsePersistenceRequest(
        corpus_id=_CORPUS_ID,
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
    )


def _recoverable_request() -> ParsePersistenceRequest:
    request = _successful_request()
    document = request.outcome.document
    assert document is not None
    outcome = ParseOutcome(
        status="completed_with_recoverable_failures",
        document=document,
        failures=(
            ParserFailure(
                category="unsupported_structure",
                severity="recoverable",
                stage="structure",
                unit_kind="table",
                unit_index=3,
            ),
        ),
    )
    return ParsePersistenceRequest(
        corpus_id=request.corpus_id,
        parser_input=request.parser_input,
        parser_version=request.parser_version,
        outcome=outcome,
        parse_id=request.parse_id,
        parsed_record_id=request.parsed_record_id,
    )


def _fatal_request() -> ParsePersistenceRequest:
    source = fixture_source("malformed_utf8.bin")
    outcome = ParseOutcome(
        status="failed",
        document=None,
        failures=(
            ParserFailure(
                category="decode_failed",
                severity="fatal",
                stage="decode",
            ),
        ),
    )
    return ParsePersistenceRequest(
        corpus_id=_CORPUS_ID,
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=None,
    )


def _outcome_signature(request: ParsePersistenceRequest) -> dict[str, object]:
    return {
        "status": request.outcome.status,
        "parsed_record_id": (
            request.parsed_record_id.as_str()
            if request.parsed_record_id is not None
            else None
        ),
        "failures": [
            {
                "category": failure.category,
                "severity": failure.severity,
                "stage": failure.stage,
                "unit_kind": failure.unit_kind,
                "unit_index": failure.unit_index,
            }
            for failure in request.outcome.failures
        ],
    }


def _case(request: ParsePersistenceRequest) -> _Case:
    document = request.outcome.document
    return _Case(
        request=request,
        document_serialised=(
            serialise_parsed_document(document) if document is not None else None
        ),
        source_bytes=request.parser_input.content,
    )


def _seed_and_publish(case: _Case, tmp_path: Path) -> tuple[Path, Path, Path]:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    result_path = tmp_path / "restart-result.json"
    repository: MetadataRepository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    source = case.request.parser_input
    version = DocumentVersion(
        document_version_id=source.document_version_id,
        document_id=source.document_id,
        content_hash=ContentHash(
            HashAlgorithm.SHA256,
            hashlib.sha256(source.content).hexdigest(),
        ),
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "ING-03 restart"))
            transaction.add_document(DocumentIdentity(source.document_id))
            transaction.add_document_version(version)
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _CORPUS_ID,
                    source.document_id,
                    source.document_version_id,
                )
            )
        persist_parse_request(
            repository,
            case.request,
            settings=fixture_settings(),
        )
    finally:
        repository.close()
    return database_path, source_dir, result_path


@pytest.mark.parametrize(
    "request_factory",
    [_successful_request, _recoverable_request, _fatal_request],
    ids=["successful", "recoverable", "fatal"],
)
def test_persisted_outcome_round_trips_through_a_fresh_process(
    tmp_path: Path,
    request_factory: Callable[[], ParsePersistenceRequest],
) -> None:
    request = request_factory()
    case = _case(request)
    database_path, source_dir, result_path = _seed_and_publish(case, tmp_path)
    parsed_record_id = case.request.parsed_record_id

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _READER,
            str(database_path),
            str(source_dir),
            str(result_path),
            _CORPUS_ID.as_str(),
            case.request.parser_input.document_version_id.as_str(),
            parsed_record_id.as_str() if parsed_record_id is not None else "-",
            case.request.parse_id.as_str(),
        ],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert bytes.fromhex(payload["source_bytes_hex"]) == case.source_bytes
    assert payload["parser_version"] == {
        "implementation": CURRENT_PARSER_VERSION.implementation,
        "contract_version": CURRENT_PARSER_VERSION.contract_version,
        "normalisation_version": CURRENT_PARSER_VERSION.normalisation_version,
    }
    assert payload["outcome"] == _outcome_signature(case.request)
    expected_document = (
        case.document_serialised.hex() if case.document_serialised is not None else None
    )
    assert payload["document_serialised_hex"] == expected_document
