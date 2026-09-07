"""Tests for ingest parse-persistence request contracts."""

from dataclasses import FrozenInstanceError, replace

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParseOutcome,
    ParsePersistenceRequest,
    ParserFailure,
    ParserInput,
    ParserVersion,
    TextNativeParser,
)
from inference_lab.provenance import CorpusId, DocumentId, DocumentVersionId, RecordId

from .fixtures import fixture_settings  # noqa: TID252


def _source(*, content: bytes = b"private-source-payload") -> ParserInput:
    return ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        content=content,
        media_type="text/plain",
        encoding="utf-8",
    )


def _successful_request() -> ParsePersistenceRequest:
    source = _source()
    outcome = TextNativeParser().parse(source, settings=fixture_settings())
    return ParsePersistenceRequest(
        corpus_id=CorpusId.generate(),
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
    )


def _failed_outcome() -> ParseOutcome:
    return ParseOutcome(
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


def test_request_retains_caller_owned_attempt_identity_immutably() -> None:
    request = _successful_request()

    assert request.outcome.status == "succeeded"
    assert request.parsed_record_id is not None
    assert request.supersedes_record_id is None

    with pytest.raises(FrozenInstanceError):
        request.parse_id = RecordId.generate()  # type: ignore[misc]


def test_request_repr_does_not_expose_source_or_parsed_payload() -> None:
    marker = "private-source-payload"
    request = _successful_request()

    rendered = repr(request)

    assert marker not in rendered
    assert "parser_input=" not in rendered
    assert "outcome=" not in rendered


def test_failed_request_retains_explicit_source_and_parser_attribution() -> None:
    source = _source(content=b"\xff")
    request = ParsePersistenceRequest(
        corpus_id=CorpusId.generate(),
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=_failed_outcome(),
        parse_id=RecordId.generate(),
        parsed_record_id=None,
    )

    assert request.parser_input.document_id == source.document_id
    assert request.parser_input.document_version_id == source.document_version_id
    assert request.parser_version == CURRENT_PARSER_VERSION


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("corpus_id", "corpus", "CorpusId"),
        ("parser_input", object(), "ParserInput"),
        ("parser_version", object(), "ParserVersion"),
        ("outcome", object(), "ParseOutcome"),
        ("parse_id", "parse", "RecordId"),
        ("parsed_record_id", "parsed", "RecordId"),
        ("supersedes_record_id", "previous", "RecordId"),
    ],
)
def test_request_rejects_wrong_runtime_types(
    field: str,
    value: object,
    match: str,
) -> None:
    request = _successful_request()

    with pytest.raises(TypeError, match=match):
        replace(request, **{field: value})


def test_represented_request_requires_parsed_record_id() -> None:
    with pytest.raises(ValueError, match="requires a parsed record ID"):
        replace(_successful_request(), parsed_record_id=None)


def test_failed_request_rejects_representation_and_supersession_ids() -> None:
    source = _source(content=b"\xff")
    values = {
        "corpus_id": CorpusId.generate(),
        "parser_input": source,
        "parser_version": CURRENT_PARSER_VERSION,
        "outcome": _failed_outcome(),
        "parse_id": RecordId.generate(),
        "parsed_record_id": RecordId.generate(),
    }

    with pytest.raises(ValueError, match="must not name a parsed record"):
        ParsePersistenceRequest(**values)  # type: ignore[arg-type]

    values["parsed_record_id"] = None
    values["supersedes_record_id"] = RecordId.generate()
    with pytest.raises(ValueError, match="must not supersede"):
        ParsePersistenceRequest(**values)  # type: ignore[arg-type]


def test_request_rejects_document_and_parser_disagreement() -> None:
    request = _successful_request()
    wrong_source = ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=request.parser_input.document_version_id,
        content=request.parser_input.content,
        media_type="text/plain",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wrong document identity"):
        replace(request, parser_input=wrong_source)

    wrong_version_source = ParserInput(
        document_id=request.parser_input.document_id,
        document_version_id=DocumentVersionId.generate(),
        content=request.parser_input.content,
        media_type="text/plain",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wrong document version"):
        replace(request, parser_input=wrong_version_source)

    other_parser_version = ParserVersion(
        implementation="text-native",
        contract_version="3.1.0",
        normalisation_version="1.0.0",
    )
    with pytest.raises(ValueError, match="wrong parser version"):
        replace(request, parser_version=other_parser_version)


@pytest.mark.parametrize(
    "duplicate_field", ["parsed_record_id", "supersedes_record_id"]
)
def test_request_requires_distinct_record_identities(duplicate_field: str) -> None:
    request = _successful_request()

    with pytest.raises(ValueError, match="identities must be distinct"):
        replace(request, **{duplicate_field: request.parse_id})
