"""Tests for read-only parse persistence preflight validation."""

import hashlib
from dataclasses import dataclass, replace
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParseOutcome,
    ParsePersistencePreflight,
    ParsePersistenceRequest,
    ParsePreflightError,
    ParsePreflightErrorCategory,
    ParsePreflightReader,
    ParserFailure,
    TextNativeParser,
    map_parse_persistence_request,
    preflight_parse_persistence,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    DocumentId,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    RecordId,
)
from inference_lab.storage import (
    ParsedRepresentation,
    ProducerIdentity,
    RepositoryUnavailableError,
    SourceArtefact,
    build_source_blob_storage_key,
)

from .fixtures import fixture_settings, fixture_source  # noqa: TID252


class _Reader:
    def __init__(
        self,
        *,
        membership: CorpusDocumentVersionMembership | None,
        document_version: DocumentVersion | None,
        source_artefact: SourceArtefact | None = None,
        predecessor: ParsedRepresentation | None = None,
    ) -> None:
        self.membership = membership
        self.document_version = document_version
        self.source_artefact = source_artefact
        self.predecessor = predecessor
        self.calls: list[tuple[str, object, object | None]] = []

    def get_corpus_document_version_membership(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> CorpusDocumentVersionMembership | None:
        self.calls.append(("membership", corpus_id, document_version_id))
        return self.membership

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        self.calls.append(("document_version", document_version_id, None))
        return self.document_version

    def get_source_artefact(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        self.calls.append(("source_artefact", corpus_id, document_version_id))
        return self.source_artefact

    def get_parsed_representation(
        self,
        corpus_id: CorpusId,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        self.calls.append(("predecessor", corpus_id, record_id))
        return self.predecessor


class _FailingReader(_Reader):
    def get_corpus_document_version_membership(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> CorpusDocumentVersionMembership | None:
        del corpus_id, document_version_id
        raise RepositoryUnavailableError


@dataclass(frozen=True, slots=True)
class _Case:
    request: ParsePersistenceRequest
    membership: CorpusDocumentVersionMembership
    document_version: DocumentVersion
    reader: _Reader


def _request() -> ParsePersistenceRequest:
    source = fixture_source("basic.txt")
    outcome = TextNativeParser().parse(source, settings=fixture_settings())
    return ParsePersistenceRequest(
        corpus_id=CorpusId.generate(),
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
    )


def _case(request: ParsePersistenceRequest | None = None) -> _Case:
    request = request or _request()
    content_hash = ContentHash(
        HashAlgorithm.SHA256,
        hashlib.sha256(request.parser_input.content).hexdigest(),
    )
    membership = CorpusDocumentVersionMembership(
        corpus_id=request.corpus_id,
        document_id=request.parser_input.document_id,
        document_version_id=request.parser_input.document_version_id,
    )
    document_version = DocumentVersion(
        document_version_id=request.parser_input.document_version_id,
        document_id=request.parser_input.document_id,
        content_hash=content_hash,
    )
    reader = _Reader(
        membership=membership,
        document_version=document_version,
    )
    return _Case(request, membership, document_version, reader)


def _fatal_request() -> ParsePersistenceRequest:
    source = fixture_source("malformed_utf8.bin")
    return ParsePersistenceRequest(
        corpus_id=CorpusId.generate(),
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=ParseOutcome(
            status="failed",
            document=None,
            failures=(
                ParserFailure(
                    category="decode_failed",
                    severity="fatal",
                    stage="decode",
                ),
            ),
        ),
        parse_id=RecordId.generate(),
        parsed_record_id=None,
    )


def _source_artefact(case: _Case, **changes: object) -> SourceArtefact:
    content_hash = changes.get("content_hash", case.document_version.content_hash)
    assert isinstance(content_hash, ContentHash)
    values: dict[str, object] = {
        "document_id": case.document_version.document_id,
        "document_version_id": case.document_version.document_version_id,
        "content_hash": content_hash,
        "byte_count": len(case.request.parser_input.content),
        "media_type": case.request.parser_input.media_type,
        "storage_key": build_source_blob_storage_key(content_hash),
    }
    values.update(changes)
    return SourceArtefact(**values)  # type: ignore[arg-type]


def _predecessor(*, same_natural_identity: bool = False) -> ParsedRepresentation:
    mapped = map_parse_persistence_request(_request())
    representation = mapped.representation
    assert representation is not None
    if same_natural_identity:
        return representation
    return replace(
        representation,
        parser=ProducerIdentity(
            implementation="text-native",
            contract_version="2.0.0",
        ),
    )


def _call_names(reader: _Reader) -> list[str]:
    return [name for name, _, _ in reader.calls]


def test_preflight_validates_source_in_read_order_without_publication() -> None:
    case = _case()

    result = preflight_parse_persistence(
        case.reader,
        case.request,
        max_source_bytes=len(case.request.parser_input.content),
    )

    assert result.membership == case.membership
    assert result.document_version == case.document_version
    assert result.source_byte_count == len(case.request.parser_input.content)
    assert _call_names(case.reader) == [
        "membership",
        "document_version",
        "source_artefact",
    ]
    assert case.reader.calls[0][1:] == (
        case.request.corpus_id,
        case.request.parser_input.document_version_id,
    )


def test_matching_existing_source_metadata_is_accepted() -> None:
    case = _case()
    case.reader.source_artefact = _source_artefact(case)

    result = preflight_parse_persistence(
        case.reader,
        case.request,
        max_source_bytes=len(case.request.parser_input.content),
    )

    assert result.document_version == case.document_version


def test_fatal_parse_preflight_preserves_request_owned_parser_identity() -> None:
    case = _case(_fatal_request())

    result = preflight_parse_persistence(
        case.reader,
        case.request,
        max_source_bytes=len(case.request.parser_input.content),
    )

    assert result.source_byte_count == len(case.request.parser_input.content)
    assert case.request.parser_version == CURRENT_PARSER_VERSION
    assert _call_names(case.reader) == [
        "membership",
        "document_version",
        "source_artefact",
    ]


def test_absent_corpus_and_absent_membership_share_one_failure_boundary() -> None:
    case = _case()
    reader = _Reader(membership=None, document_version=case.document_version)

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(
            reader,
            case.request,
            max_source_bytes=100,
        )

    assert error.value.category is ParsePreflightErrorCategory.SOURCE_NOT_AVAILABLE
    assert str(error.value) == "Parse persistence preflight failed."
    assert _call_names(reader) == ["membership"]
    assert not hasattr(reader, "get_corpus")


@pytest.mark.parametrize("wrong_membership_field", ["corpus_id", "document_version_id"])
def test_wrong_membership_scope_is_indistinguishable_from_absence(
    wrong_membership_field: str,
) -> None:
    case = _case()
    value = (
        CorpusId.generate()
        if wrong_membership_field == "corpus_id"
        else DocumentVersionId.generate()
    )
    case.reader.membership = replace(
        case.membership,
        **{wrong_membership_field: value},
    )

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, case.request, max_source_bytes=100)

    assert error.value.category is ParsePreflightErrorCategory.SOURCE_NOT_AVAILABLE
    assert _call_names(case.reader) == ["membership"]


def test_membership_document_mismatch_fails_before_version_lookup() -> None:
    case = _case()
    case.reader.membership = replace(
        case.membership,
        document_id=DocumentId.generate(),
    )

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, case.request, max_source_bytes=100)

    assert error.value.category is (
        ParsePreflightErrorCategory.SOURCE_IDENTITY_MISMATCH
    )
    assert _call_names(case.reader) == ["membership"]


def test_missing_document_version_fails_after_membership_only() -> None:
    case = _case()
    case.reader.document_version = None

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, case.request, max_source_bytes=100)

    assert error.value.category is ParsePreflightErrorCategory.SOURCE_NOT_AVAILABLE
    assert _call_names(case.reader) == ["membership", "document_version"]


@pytest.mark.parametrize("wrong_version_field", ["document_id", "document_version_id"])
def test_immutable_version_must_match_membership(
    wrong_version_field: str,
) -> None:
    case = _case()
    value = (
        DocumentId.generate()
        if wrong_version_field == "document_id"
        else DocumentVersionId.generate()
    )
    case.reader.document_version = replace(
        case.document_version,
        **{wrong_version_field: value},
    )

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, case.request, max_source_bytes=100)

    assert error.value.category is (
        ParsePreflightErrorCategory.SOURCE_IDENTITY_MISMATCH
    )
    assert _call_names(case.reader) == ["membership", "document_version"]


def test_source_size_limit_is_checked_before_later_reads() -> None:
    case = _case()

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(
            case.reader,
            case.request,
            max_source_bytes=len(case.request.parser_input.content) - 1,
        )

    assert error.value.category is ParsePreflightErrorCategory.SOURCE_TOO_LARGE
    assert _call_names(case.reader) == ["membership", "document_version"]


def test_exact_source_sha256_must_match_immutable_version() -> None:
    case = _case()
    case.reader.document_version = replace(
        case.document_version,
        content_hash=ContentHash(HashAlgorithm.SHA256, "a" * 64),
    )

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, case.request, max_source_bytes=100)

    assert error.value.category is ParsePreflightErrorCategory.SOURCE_CONTENT_MISMATCH
    assert _call_names(case.reader) == ["membership", "document_version"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_id", DocumentId.generate()),
        ("document_version_id", DocumentVersionId.generate()),
        ("byte_count", 1),
    ],
)
def test_existing_source_identity_and_length_must_match(
    field: str,
    value: object,
) -> None:
    case = _case()
    case.reader.source_artefact = _source_artefact(case, **{field: value})

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, case.request, max_source_bytes=100)

    assert error.value.category is (
        ParsePreflightErrorCategory.SOURCE_IDENTITY_MISMATCH
    )
    assert _call_names(case.reader) == [
        "membership",
        "document_version",
        "source_artefact",
    ]


def test_existing_source_hash_must_match_immutable_version() -> None:
    case = _case()
    other_hash = ContentHash(HashAlgorithm.SHA256, "b" * 64)
    case.reader.source_artefact = _source_artefact(case, content_hash=other_hash)

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, case.request, max_source_bytes=100)

    assert error.value.category is (
        ParsePreflightErrorCategory.SOURCE_IDENTITY_MISMATCH
    )


def test_existing_source_media_type_must_match_parser_input() -> None:
    case = _case()
    case.reader.source_artefact = _source_artefact(
        case,
        media_type="application/pdf",
    )

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, case.request, max_source_bytes=100)

    assert error.value.category is (
        ParsePreflightErrorCategory.SOURCE_MEDIA_TYPE_MISMATCH
    )


def test_explicit_compatible_predecessor_is_read_through_selected_corpus() -> None:
    case = _case()
    predecessor = _predecessor()
    request = replace(case.request, supersedes_record_id=predecessor.record_id)
    case.reader.predecessor = predecessor

    result = preflight_parse_persistence(
        case.reader,
        request,
        max_source_bytes=100,
    )

    assert result.document_version == case.document_version
    assert _call_names(case.reader)[-1] == "predecessor"
    assert case.reader.calls[-1][1:] == (
        request.corpus_id,
        predecessor.record_id,
    )


def test_missing_or_foreign_predecessor_is_rejected() -> None:
    case = _case()
    request = replace(
        case.request,
        supersedes_record_id=RecordId.generate(),
    )

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, request, max_source_bytes=100)

    assert error.value.category is ParsePreflightErrorCategory.INVALID_PREDECESSOR


@pytest.mark.parametrize(
    "wrong_field", ["record_id", "document_id", "document_version_id"]
)
def test_predecessor_must_name_the_exact_same_document_version(
    wrong_field: str,
) -> None:
    case = _case()
    predecessor = _predecessor()
    request = replace(case.request, supersedes_record_id=predecessor.record_id)
    value = (
        RecordId.generate()
        if wrong_field == "record_id"
        else (
            DocumentId.generate()
            if wrong_field == "document_id"
            else DocumentVersionId.generate()
        )
    )
    object.__setattr__(predecessor, wrong_field, value)
    case.reader.predecessor = predecessor

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, request, max_source_bytes=100)

    assert error.value.category is ParsePreflightErrorCategory.INVALID_PREDECESSOR


def test_predecessor_with_same_natural_identity_cannot_create_a_variant() -> None:
    case = _case()
    predecessor = _predecessor(same_natural_identity=True)
    request = replace(case.request, supersedes_record_id=predecessor.record_id)
    case.reader.predecessor = predecessor

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, request, max_source_bytes=100)

    assert error.value.category is ParsePreflightErrorCategory.INVALID_PREDECESSOR


def test_ocr_predecessor_is_outside_ing03_contract() -> None:
    case = _case()
    predecessor = _predecessor()
    predecessor = replace(
        predecessor,
        ocr_producer=ProducerIdentity(
            implementation="ocr-engine",
            contract_version="1.0.0",
        ),
    )
    request = replace(case.request, supersedes_record_id=predecessor.record_id)
    case.reader.predecessor = predecessor

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, request, max_source_bytes=100)

    assert error.value.category is ParsePreflightErrorCategory.INVALID_PREDECESSOR


def test_attributed_parser_predecessor_is_outside_ing03_contract() -> None:
    case = _case()
    predecessor = _predecessor()
    predecessor = replace(
        predecessor,
        parser=ProducerIdentity(
            implementation="text-native",
            contract_version="2.0.0",
            artefact_identity="model",
            artefact_revision="revision",
        ),
    )
    request = replace(case.request, supersedes_record_id=predecessor.record_id)
    case.reader.predecessor = predecessor

    with pytest.raises(ParsePreflightError) as error:
        preflight_parse_persistence(case.reader, request, max_source_bytes=100)

    assert error.value.category is ParsePreflightErrorCategory.INVALID_PREDECESSOR


@pytest.mark.parametrize("max_source_bytes", [0, -1])
def test_nonpositive_size_limit_is_rejected_before_reads(
    max_source_bytes: int,
) -> None:
    case = _case()

    with pytest.raises(ValueError, match="must be positive"):
        preflight_parse_persistence(
            case.reader,
            case.request,
            max_source_bytes=max_source_bytes,
        )

    assert case.reader.calls == []


@pytest.mark.parametrize("max_source_bytes", [True, 1.5])
def test_noninteger_size_limit_is_rejected_before_reads(
    max_source_bytes: object,
) -> None:
    case = _case()

    with pytest.raises(TypeError, match="must be an integer"):
        preflight_parse_persistence(
            case.reader,
            case.request,
            max_source_bytes=max_source_bytes,  # type: ignore[arg-type]
        )

    assert case.reader.calls == []


def test_invalid_reader_and_request_are_rejected_before_reads() -> None:
    case = _case()

    with pytest.raises(TypeError, match="reader"):
        preflight_parse_persistence(
            cast("ParsePreflightReader", object()),
            case.request,
            max_source_bytes=100,
        )
    with pytest.raises(TypeError, match="request"):
        preflight_parse_persistence(
            case.reader,
            cast("ParsePersistenceRequest", object()),
            max_source_bytes=100,
        )

    assert case.reader.calls == []


def test_storage_read_errors_propagate_without_translation() -> None:
    case = _case()
    reader = _FailingReader(
        membership=case.membership,
        document_version=case.document_version,
    )

    with pytest.raises(RepositoryUnavailableError):
        preflight_parse_persistence(reader, case.request, max_source_bytes=100)


def test_preflight_result_and_errors_do_not_expose_source_data() -> None:
    case = _case()
    result = preflight_parse_persistence(
        case.reader,
        case.request,
        max_source_bytes=100,
    )
    rendered = repr(result)

    assert "First paragraph" not in rendered
    assert case.document_version.content_hash.digest not in rendered
    assert "document_version=" not in rendered
    assert "membership=" not in rendered

    missing_reader = _Reader(membership=None, document_version=None)
    with pytest.raises(ParsePreflightError) as captured:
        preflight_parse_persistence(
            missing_reader,
            case.request,
            max_source_bytes=100,
        )
    assert "First paragraph" not in repr(captured.value)


def test_preflight_emits_no_logs() -> None:
    case = _case()

    with capture_logs() as logs:
        preflight_parse_persistence(
            case.reader,
            case.request,
            max_source_bytes=100,
        )

    assert logs == []


def test_preflight_result_validates_its_contract() -> None:
    case = _case()

    with pytest.raises(TypeError, match="membership"):
        ParsePersistencePreflight(
            membership=cast("CorpusDocumentVersionMembership", object()),
            document_version=case.document_version,
            source_byte_count=1,
        )
    with pytest.raises(TypeError, match="document_version"):
        ParsePersistencePreflight(
            membership=case.membership,
            document_version=cast("DocumentVersion", object()),
            source_byte_count=1,
        )
    with pytest.raises(TypeError, match="source_byte_count"):
        ParsePersistencePreflight(  # type: ignore[arg-type]
            membership=case.membership,
            document_version=case.document_version,
            source_byte_count=True,
        )
    with pytest.raises(ValueError, match="must not be negative"):
        ParsePersistencePreflight(
            membership=case.membership,
            document_version=case.document_version,
            source_byte_count=-1,
        )
    with pytest.raises(ValueError, match="must agree"):
        ParsePersistencePreflight(
            membership=case.membership,
            document_version=replace(
                case.document_version,
                document_id=DocumentId.generate(),
            ),
            source_byte_count=1,
        )
