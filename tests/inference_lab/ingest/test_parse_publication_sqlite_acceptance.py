"""SQLite integration coverage for replay, conflict, and supersession."""

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParsePersistenceRequest,
    ParserVersion,
    TextNativeParser,
    generate_source_span_id,
    map_parse_persistence_request,
    persist_parse_request,
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
from inference_lab.storage import (
    MetadataRepository,
    StorageConflictError,
    open_metadata_repository,
)

from .fixtures import fixture_settings, fixture_source  # noqa: TID252

_CORPUS_ID = CorpusId("a0000000-0000-4000-8000-000000000001")
_VARIANT_PARSER_VERSION = ParserVersion(
    implementation="text-native",
    contract_version="4.0.0",
    normalisation_version="1.0.0",
)


@dataclass(frozen=True, slots=True)
class _Case:
    repository: MetadataRepository
    request: ParsePersistenceRequest


@contextmanager
def _case(tmp_path: Path) -> Iterator[_Case]:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    repository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    source = fixture_source("basic.txt")
    outcome = TextNativeParser().parse(source, settings=fixture_settings())
    request = ParsePersistenceRequest(
        corpus_id=_CORPUS_ID,
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
    )
    content_hash = ContentHash(
        HashAlgorithm.SHA256,
        hashlib.sha256(source.content).hexdigest(),
    )
    version = DocumentVersion(
        document_version_id=source.document_version_id,
        document_id=source.document_id,
        content_hash=content_hash,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "ING-03 acceptance"))
        transaction.add_document(DocumentIdentity(source.document_id))
        transaction.add_document_version(version)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS_ID,
                source.document_id,
                source.document_version_id,
            )
        )
    try:
        yield _Case(repository, request)
    finally:
        repository.close()


def _variant_request(
    request: ParsePersistenceRequest,
    *,
    supersedes_record_id: RecordId | None = None,
) -> ParsePersistenceRequest:
    document = request.outcome.document
    parsed_record_id = request.parsed_record_id
    assert document is not None
    assert parsed_record_id is not None

    blocks_by_span_id = {}
    variant_blocks = []
    for ordinal, block in enumerate(document.blocks):
        span = block.source_span
        variant_span = replace(
            span,
            source_span_id=generate_source_span_id(
                span.document_version_id,
                parser_version=_VARIANT_PARSER_VERSION,
                ordinal=ordinal,
                kind=block.kind,
                start_offset=span.start_offset,
                end_offset=span.end_offset,
            ),
        )
        variant_block = replace(block, source_span=variant_span)
        blocks_by_span_id[span.source_span_id] = variant_block
        variant_blocks.append(variant_block)

    def variant_collection(collection: tuple | None) -> tuple | None:
        if collection is None:
            return None
        return tuple(
            blocks_by_span_id[block.source_span.source_span_id] for block in collection
        )

    variant_document = replace(
        document,
        blocks=tuple(variant_blocks),
        pages=variant_collection(document.pages),
        sections=variant_collection(document.sections),
        paragraphs=variant_collection(document.paragraphs),
        parser_version=_VARIANT_PARSER_VERSION,
    )
    variant_outcome = replace(request.outcome, document=variant_document)
    return replace(
        request,
        parser_version=_VARIANT_PARSER_VERSION,
        outcome=variant_outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
        supersedes_record_id=supersedes_record_id,
    )


def _changed_payload_request(
    request: ParsePersistenceRequest,
) -> ParsePersistenceRequest:
    document = request.outcome.document
    assert document is not None
    changed_document = replace(document, language="en")
    return replace(
        request,
        outcome=replace(request.outcome, document=changed_document),
        parse_id=RecordId.generate(),
    )


def test_exact_replay_is_idempotent_and_returns_the_same_result(
    tmp_path: Path,
) -> None:
    with _case(tmp_path) as case:
        document = case.request.outcome.document
        assert document is not None
        first = persist_parse_request(
            case.repository,
            case.request,
            settings=fixture_settings(),
        )
        second = persist_parse_request(
            case.repository,
            case.request,
            settings=fixture_settings(),
        )

        assert second == first
        assert (
            case.repository.get_source_artefact(
                _CORPUS_ID,
                case.request.parser_input.document_version_id,
            )
            is not None
        )
        assert case.repository.list_parsed_representations(
            _CORPUS_ID,
            case.request.parser_input.document_version_id,
        ) == (map_parse_persistence_request(case.request).representation,)
        assert (
            case.repository.get_parse_outcome(
                _CORPUS_ID,
                case.request.parse_id,
            )
            == map_parse_persistence_request(case.request).outcome
        )
        assert len(
            case.repository.list_known_source_spans(
                _CORPUS_ID,
                case.request.parser_input.document_version_id,
            )
        ) == len(document.blocks)


def test_changed_payload_under_one_natural_identity_is_a_conflict(
    tmp_path: Path,
) -> None:
    with _case(tmp_path) as case:
        persist_parse_request(
            case.repository,
            case.request,
            settings=fixture_settings(),
        )
        changed = _changed_payload_request(case.request)

        with pytest.raises(StorageConflictError) as error:
            persist_parse_request(
                case.repository,
                changed,
                settings=fixture_settings(),
            )

        assert error.value.retryable is False
        representations = case.repository.list_parsed_representations(
            _CORPUS_ID,
            case.request.parser_input.document_version_id,
        )
        assert len(representations) == 1
        assert representations[0].language != "en"
        assert (
            case.repository.get_parse_outcome(
                _CORPUS_ID,
                changed.parse_id,
            )
            is None
        )


def test_new_parser_version_can_explicitly_supersede_an_older_representation(
    tmp_path: Path,
) -> None:
    with _case(tmp_path) as case:
        first = persist_parse_request(
            case.repository,
            case.request,
            settings=fixture_settings(),
        )
        variant = _variant_request(
            case.request,
            supersedes_record_id=first.parsed_record_id,
        )

        second = persist_parse_request(
            case.repository,
            variant,
            settings=fixture_settings(),
        )

        assert second.parse_id == variant.parse_id
        assert second.parsed_record_id == variant.parsed_record_id
        representations = case.repository.list_parsed_representations(
            _CORPUS_ID,
            case.request.parser_input.document_version_id,
        )
        assert len(representations) == 2
        stored_variant = next(
            representation
            for representation in representations
            if representation.record_id == variant.parsed_record_id
        )
        assert stored_variant.parser.contract_version == "4.0.0"
        assert stored_variant.supersedes_record_id == first.parsed_record_id
        assert (
            case.repository.get_parse_outcome(
                _CORPUS_ID,
                variant.parse_id,
            )
            is not None
        )
