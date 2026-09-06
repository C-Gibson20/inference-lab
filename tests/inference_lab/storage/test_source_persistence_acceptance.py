"""End-to-end acceptance tests for restart-safe source persistence."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from time import monotonic, perf_counter, sleep

import pytest
from structlog.contextvars import merge_contextvars
from structlog.testing import capture_logs

from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
)
from inference_lab.storage import (
    BoundingBox,
    ContentOrigin,
    CoordinateOrigin,
    CoordinateUnit,
    MetadataRepository,
    PageLayoutDescriptor,
    PageRotation,
    ParsedRepresentation,
    ParseOutcome,
    ParseStatus,
    SourceArtefact,
    StorageConcurrencyError,
    StorageConflictError,
    open_metadata_repository,
    serialise_parse_outcome,
    serialise_parsed_representation,
)

from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    FIRST_SPAN_ID,
    PARSE_ID,
    PARSED_RECORD_ID,
    PARSER,
    SECOND_SPAN_ID,
    SOURCE_BYTES,
    SOURCE_HASH,
    parsed_representation,
)

_REPOSITORY_ROOT = Path(__file__).parents[3]
_CORPUS = CorpusMetadata(
    CorpusId("90000000-0000-4000-8000-000000000001"),
    "Cold-process acceptance corpus",
)
_OTHER_CORPUS = CorpusMetadata(
    CorpusId("90000000-0000-4000-8000-000000000002"),
    "Other acceptance corpus",
)
_VERSION = DocumentVersion(DOCUMENT_VERSION_ID, DOCUMENT_ID, SOURCE_HASH)
_ARTEFACT = SourceArtefact(
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    content_hash=SOURCE_HASH,
    byte_count=len(SOURCE_BYTES),
    media_type="application/pdf",
    storage_key=f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}",
)
_OUTCOME = ParseOutcome(
    parse_id=PARSE_ID,
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    status=ParseStatus.SUCCEEDED,
    parser=PARSER,
    normalisation_version="1.0.0",
    parsed_record_id=PARSED_RECORD_ID,
    failures=(),
)
_FIRST_ONLY_BYTES = b"first corpus only"
_FIRST_ONLY_DOCUMENT_ID = DocumentId("91000000-0000-4000-8000-000000000001")
_FIRST_ONLY_VERSION_ID = DocumentVersionId("92000000-0000-4000-8000-000000000001")
_FIRST_ONLY_HASH = ContentHash(
    HashAlgorithm.SHA256,
    hashlib.sha256(_FIRST_ONLY_BYTES).hexdigest(),
)
_FIRST_ONLY_VERSION = DocumentVersion(
    _FIRST_ONLY_VERSION_ID,
    _FIRST_ONLY_DOCUMENT_ID,
    _FIRST_ONLY_HASH,
)
_FIRST_ONLY_ARTEFACT = SourceArtefact(
    document_id=_FIRST_ONLY_DOCUMENT_ID,
    document_version_id=_FIRST_ONLY_VERSION_ID,
    content_hash=_FIRST_ONLY_HASH,
    byte_count=len(_FIRST_ONLY_BYTES),
    media_type="text/plain",
    storage_key=(f"sha256/{_FIRST_ONLY_HASH.digest[:2]}/{_FIRST_ONLY_HASH.digest}"),
)
_SECOND_ONLY_BYTES = b"second corpus only"
_SECOND_ONLY_DOCUMENT_ID = DocumentId("91000000-0000-4000-8000-000000000002")
_SECOND_ONLY_VERSION_ID = DocumentVersionId("92000000-0000-4000-8000-000000000002")
_SECOND_ONLY_HASH = ContentHash(
    HashAlgorithm.SHA256,
    hashlib.sha256(_SECOND_ONLY_BYTES).hexdigest(),
)
_SECOND_ONLY_VERSION = DocumentVersion(
    _SECOND_ONLY_VERSION_ID,
    _SECOND_ONLY_DOCUMENT_ID,
    _SECOND_ONLY_HASH,
)
_SECOND_ONLY_ARTEFACT = SourceArtefact(
    document_id=_SECOND_ONLY_DOCUMENT_ID,
    document_version_id=_SECOND_ONLY_VERSION_ID,
    content_hash=_SECOND_ONLY_HASH,
    byte_count=len(_SECOND_ONLY_BYTES),
    media_type="text/plain",
    storage_key=(f"sha256/{_SECOND_ONLY_HASH.digest[:2]}/{_SECOND_ONLY_HASH.digest}"),
)

_COLD_PROCESS_READER = """
import json
import sys
from pathlib import Path

from inference_lab.provenance import CorpusId, DocumentVersionId, RecordId, SourceSpanId
from inference_lab.storage import (
    MetadataRepository,
    open_metadata_repository,
    serialise_parse_outcome,
    serialise_parsed_representation,
)

database_path = Path(sys.argv[1])
source_dir = Path(sys.argv[2])
result_path = Path(sys.argv[3])
corpus_id = CorpusId.from_str(sys.argv[4])
document_version_id = DocumentVersionId.from_str(sys.argv[5])
record_id = RecordId.from_str(sys.argv[6])
parse_id = RecordId.from_str(sys.argv[7])
span_ids = tuple(SourceSpanId.from_str(value) for value in sys.argv[8:])

repository = open_metadata_repository(database_path, source_dir=source_dir)
assert isinstance(repository, MetadataRepository)
try:
    source_bytes = repository.read_source_bytes(corpus_id, document_version_id)
    source_stream = repository.stream_source_bytes(corpus_id, document_version_id)
    representation = repository.get_parsed_representation(corpus_id, record_id)
    outcome = repository.get_parse_outcome(corpus_id, parse_id)
    known_spans = repository.list_known_source_spans(corpus_id, document_version_id)
    assert source_bytes is not None
    assert source_stream is not None
    assert representation is not None
    assert outcome is not None

    resolved_payloads = []
    for span_id in span_ids:
        resolved = repository.resolve_source_span(corpus_id, span_id)
        assert resolved is not None
        location = resolved.location
        box = location.bounding_box
        layout = resolved.page_layout
        resolved_payloads.append(
            {
                "document_id": resolved.document_id.as_str(),
                "document_version_id": location.document_version_id.as_str(),
                "source_span_id": location.source_span_id.as_str(),
                "parsed_record_id": location.parsed_record_id.as_str(),
                "offsets": [location.start_offset, location.end_offset],
                "decoded_text": representation.source_text[
                    location.start_offset : location.end_offset
                ],
                "page_number": location.page_number,
                "bounding_box": (
                    None
                    if box is None
                    else [box.left, box.top, box.right, box.bottom]
                ),
                "layout": (
                    None
                    if layout is None
                    else {
                        "page_number": layout.page_number,
                        "coordinate_unit": layout.coordinate_unit.value,
                        "coordinate_origin": layout.coordinate_origin.value,
                        "page_width": layout.page_width,
                        "page_height": layout.page_height,
                        "rotation": layout.rotation.value,
                        "content_origin": layout.content_origin.value,
                    }
                ),
            }
        )

    payload = {
        "source_bytes_hex": source_bytes.hex(),
        "streamed_bytes_hex": b"".join(source_stream).hex(),
        "representation": serialise_parsed_representation(representation).decode(),
        "outcome": serialise_parse_outcome(outcome).decode(),
        "resolved_spans": resolved_payloads,
        "known_spans": [
            [span.document_version_id.as_str(), span.source_span_id.as_str()]
            for span in known_spans
        ],
    }
finally:
    repository.close()

result_path.write_text(
    json.dumps(payload, ensure_ascii=False, sort_keys=True),
    encoding="utf-8",
)
"""

_CONTENDED_PUBLICATION_WRITER = """
import sys
from pathlib import Path
from time import monotonic, sleep

from inference_lab.storage import (
    MetadataRepository,
    ParseOutcome,
    ParseStatus,
    SourceArtefact,
    open_metadata_repository,
)
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    PARSE_ID,
    PARSED_RECORD_ID,
    PARSER,
    SOURCE_BYTES,
    SOURCE_HASH,
    parsed_representation,
)
from inference_lab.provenance import CorpusId

database_path = Path(sys.argv[1])
source_dir = Path(sys.argv[2])
ready_path = Path(sys.argv[3])
release_path = Path(sys.argv[4])
corpus_id = CorpusId.from_str(sys.argv[5])
artefact = SourceArtefact(
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    content_hash=SOURCE_HASH,
    byte_count=len(SOURCE_BYTES),
    media_type="application/pdf",
    storage_key=f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}",
)
outcome = ParseOutcome(
    parse_id=PARSE_ID,
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    status=ParseStatus.SUCCEEDED,
    parser=PARSER,
    normalisation_version="1.0.0",
    parsed_record_id=PARSED_RECORD_ID,
    failures=(),
)

def controlled_chunks():
    ready_path.touch()
    deadline = monotonic() + 10
    while not release_path.exists():
        if monotonic() >= deadline:
            raise TimeoutError("publication release was not signalled")
        sleep(0.01)
    yield SOURCE_BYTES

repository = open_metadata_repository(database_path, source_dir=source_dir)
assert isinstance(repository, MetadataRepository)
try:
    repository.persist_source_artefact(
        corpus_id,
        artefact,
        controlled_chunks(),
        max_source_bytes=len(SOURCE_BYTES),
        parse_outcome=outcome,
        parsed_representation=parsed_representation(),
    )
finally:
    repository.close()
"""


def _representation_with_optional_page_region() -> ParsedRepresentation:
    representation = parsed_representation()
    layout = PageLayoutDescriptor(
        page_number=1,
        coordinate_unit=CoordinateUnit.PDF_POINTS,
        coordinate_origin=CoordinateOrigin.TOP_LEFT,
        page_width=612.0,
        page_height=792.0,
        rotation=PageRotation.DEGREES_0,
        content_origin=ContentOrigin.NATIVE,
        producer=PARSER,
    )
    first_location = replace(
        representation.source_locations[0],
        page_number=1,
        bounding_box=BoundingBox(
            left=10.0,
            top=20.0,
            right=110.0,
            bottom=40.0,
        ),
    )
    return replace(
        representation,
        page_count=1,
        source_locations=(first_location, representation.source_locations[1]),
        page_layouts=(layout,),
    )


def _publish_complete_source(
    database_path: Path,
    source_dir: Path,
    representation: ParsedRepresentation,
) -> None:
    repository: MetadataRepository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_CORPUS)
            transaction.add_document(DocumentIdentity(DOCUMENT_ID))
            transaction.add_document_version(_VERSION)
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _CORPUS.corpus_id,
                    DOCUMENT_ID,
                    DOCUMENT_VERSION_ID,
                )
            )
        repository.persist_source_artefact(
            _CORPUS.corpus_id,
            _ARTEFACT,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
            parse_outcome=_OUTCOME,
            parsed_representation=representation,
        )
    finally:
        repository.close()


def test_fresh_process_recovers_complete_source_parse_and_locations(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    result_path = tmp_path / "cold-process-result.json"
    representation = _representation_with_optional_page_region()
    _publish_complete_source(database_path, source_dir, representation)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _COLD_PROCESS_READER,
            str(database_path),
            str(source_dir),
            str(result_path),
            _CORPUS.corpus_id.as_str(),
            DOCUMENT_VERSION_ID.as_str(),
            PARSED_RECORD_ID.as_str(),
            PARSE_ID.as_str(),
            FIRST_SPAN_ID.as_str(),
            SECOND_SPAN_ID.as_str(),
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "repository_open_completed" in completed.stdout
    assert "source_reconciliation_completed" in completed.stdout
    assert "repository_close_completed" in completed.stdout
    for private_value in (
        str(database_path),
        str(source_dir),
        _ARTEFACT.storage_key,
        SOURCE_HASH.digest,
        SOURCE_BYTES.decode(),
    ):
        assert private_value not in completed.stdout
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "source_bytes_hex": SOURCE_BYTES.hex(),
        "streamed_bytes_hex": SOURCE_BYTES.hex(),
        "representation": serialise_parsed_representation(representation).decode(),
        "outcome": serialise_parse_outcome(_OUTCOME).decode(),
        "resolved_spans": [
            {
                "document_id": DOCUMENT_ID.as_str(),
                "document_version_id": DOCUMENT_VERSION_ID.as_str(),
                "source_span_id": FIRST_SPAN_ID.as_str(),
                "parsed_record_id": PARSED_RECORD_ID.as_str(),
                "offsets": [0, 5],
                "decoded_text": "Cafe\u0301",
                "page_number": 1,
                "bounding_box": [10.0, 20.0, 110.0, 40.0],
                "layout": {
                    "page_number": 1,
                    "coordinate_unit": "pdf_points",
                    "coordinate_origin": "top_left",
                    "page_width": 612.0,
                    "page_height": 792.0,
                    "rotation": 0,
                    "content_origin": "native",
                },
            },
            {
                "document_id": DOCUMENT_ID.as_str(),
                "document_version_id": DOCUMENT_VERSION_ID.as_str(),
                "source_span_id": SECOND_SPAN_ID.as_str(),
                "parsed_record_id": PARSED_RECORD_ID.as_str(),
                "offsets": [9, 16],
                "decoded_text": "Results",
                "page_number": None,
                "bounding_box": None,
                "layout": None,
            },
        ],
        "known_spans": [
            [DOCUMENT_VERSION_ID.as_str(), FIRST_SPAN_ID.as_str()],
            [DOCUMENT_VERSION_ID.as_str(), SECOND_SPAN_ID.as_str()],
        ],
    }


def test_two_corpora_reuse_only_explicitly_shared_source_and_parse_records(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    representation = _representation_with_optional_page_region()
    repository: MetadataRepository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_CORPUS)
            transaction.add_corpus(_OTHER_CORPUS)
            for document_id in (
                DOCUMENT_ID,
                _FIRST_ONLY_DOCUMENT_ID,
                _SECOND_ONLY_DOCUMENT_ID,
            ):
                transaction.add_document(DocumentIdentity(document_id))
            for version in (
                _VERSION,
                _FIRST_ONLY_VERSION,
                _SECOND_ONLY_VERSION,
            ):
                transaction.add_document_version(version)
            for corpus_id, document_id, version_id in (
                (_CORPUS.corpus_id, DOCUMENT_ID, DOCUMENT_VERSION_ID),
                (
                    _CORPUS.corpus_id,
                    _FIRST_ONLY_DOCUMENT_ID,
                    _FIRST_ONLY_VERSION_ID,
                ),
                (
                    _OTHER_CORPUS.corpus_id,
                    _SECOND_ONLY_DOCUMENT_ID,
                    _SECOND_ONLY_VERSION_ID,
                ),
            ):
                transaction.add_corpus_document_version_membership(
                    CorpusDocumentVersionMembership(
                        corpus_id,
                        document_id,
                        version_id,
                    )
                )

        repository.persist_source_artefact(
            _CORPUS.corpus_id,
            _ARTEFACT,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
            parse_outcome=_OUTCOME,
            parsed_representation=representation,
        )
        repository.persist_source_artefact(
            _CORPUS.corpus_id,
            _FIRST_ONLY_ARTEFACT,
            (_FIRST_ONLY_BYTES,),
            max_source_bytes=len(_FIRST_ONLY_BYTES),
        )
        repository.persist_source_artefact(
            _OTHER_CORPUS.corpus_id,
            _SECOND_ONLY_ARTEFACT,
            (_SECOND_ONLY_BYTES,),
            max_source_bytes=len(_SECOND_ONLY_BYTES),
        )

        assert (
            repository.get_source_artefact(
                _OTHER_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert (
            repository.read_source_bytes(
                _OTHER_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert (
            repository.get_parsed_representation(
                _OTHER_CORPUS.corpus_id,
                PARSED_RECORD_ID,
            )
            is None
        )
        assert (
            repository.list_parsed_representations(
                _OTHER_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == ()
        )
        assert repository.get_parse_outcome(_OTHER_CORPUS.corpus_id, PARSE_ID) is None
        assert (
            repository.resolve_source_span(
                _OTHER_CORPUS.corpus_id,
                FIRST_SPAN_ID,
            )
            is None
        )
        assert (
            repository.list_known_source_spans(
                _OTHER_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == ()
        )
        with pytest.raises(StorageConflictError):
            repository.persist_source_artefact(
                _OTHER_CORPUS.corpus_id,
                _ARTEFACT,
                (SOURCE_BYTES,),
                max_source_bytes=len(SOURCE_BYTES),
                parse_outcome=_OUTCOME,
                parsed_representation=representation,
            )

        assert (
            repository.read_source_bytes(
                _CORPUS.corpus_id,
                _SECOND_ONLY_VERSION_ID,
            )
            is None
        )
        assert (
            repository.read_source_bytes(
                _OTHER_CORPUS.corpus_id,
                _FIRST_ONLY_VERSION_ID,
            )
            is None
        )

        with repository.transaction() as transaction:
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _OTHER_CORPUS.corpus_id,
                    DOCUMENT_ID,
                    DOCUMENT_VERSION_ID,
                )
            )
        repository.persist_source_artefact(
            _OTHER_CORPUS.corpus_id,
            _ARTEFACT,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
            parse_outcome=_OUTCOME,
            parsed_representation=representation,
        )
    finally:
        repository.close()

    reopened: MetadataRepository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
    )
    try:
        for corpus_id in (_CORPUS.corpus_id, _OTHER_CORPUS.corpus_id):
            assert (
                reopened.get_source_artefact(corpus_id, DOCUMENT_VERSION_ID)
                == _ARTEFACT
            )
            assert (
                reopened.read_source_bytes(corpus_id, DOCUMENT_VERSION_ID)
                == SOURCE_BYTES
            )
            assert (
                reopened.get_parsed_representation(corpus_id, PARSED_RECORD_ID)
                == representation
            )
            assert reopened.list_parsed_representations(
                corpus_id,
                DOCUMENT_VERSION_ID,
            ) == (representation,)
            assert reopened.get_parse_outcome(corpus_id, PARSE_ID) == _OUTCOME
            assert reopened.resolve_source_span(corpus_id, FIRST_SPAN_ID) is not None
            assert (
                reopened.list_known_source_spans(
                    corpus_id,
                    DOCUMENT_VERSION_ID,
                )
                == representation.known_source_spans()
            )

        assert (
            reopened.read_source_bytes(
                _CORPUS.corpus_id,
                _FIRST_ONLY_VERSION_ID,
            )
            == _FIRST_ONLY_BYTES
        )
        assert (
            reopened.read_source_bytes(
                _OTHER_CORPUS.corpus_id,
                _SECOND_ONLY_VERSION_ID,
            )
            == _SECOND_ONLY_BYTES
        )
        assert (
            reopened.read_source_bytes(
                _CORPUS.corpus_id,
                _SECOND_ONLY_VERSION_ID,
            )
            is None
        )
        assert (
            reopened.read_source_bytes(
                _OTHER_CORPUS.corpus_id,
                _FIRST_ONLY_VERSION_ID,
            )
            is None
        )
    finally:
        reopened.close()

    managed_blobs = tuple(
        path for path in (source_dir / "sha256").glob("*/*") if path.is_file()
    )
    assert len(managed_blobs) == 3
    assert (source_dir / _ARTEFACT.storage_key).is_file()


def test_cross_process_publication_contention_is_bounded_and_retryable(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    database_path = tmp_path / "private-metadata" / "metadata.sqlite3"
    source_dir = tmp_path / "private-source-data"
    ready_path = tmp_path / "publication-lock-held"
    release_path = tmp_path / "release-publication-lock"
    representation = parsed_representation()
    repository: MetadataRepository = open_metadata_repository(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    request.addfinalizer(repository.close)
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS)
        transaction.add_document(DocumentIdentity(DOCUMENT_ID))
        transaction.add_document_version(_VERSION)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS.corpus_id,
                DOCUMENT_ID,
                DOCUMENT_VERSION_ID,
            )
        )

    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CONTENDED_PUBLICATION_WRITER,
            str(database_path),
            str(source_dir),
            str(ready_path),
            str(release_path),
            _CORPUS.corpus_id.as_str(),
        ],
        cwd=_REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_stdout = child_stderr = ""
    try:
        deadline = monotonic() + 10
        while not ready_path.exists() and writer.poll() is None:
            if monotonic() >= deadline:
                break
            sleep(0.01)
        assert ready_path.is_file(), "child did not acquire the publication lock"

        with capture_logs(processors=(merge_contextvars,)) as logs:
            started_at = perf_counter()
            with pytest.raises(StorageConcurrencyError) as raised:
                repository.persist_source_artefact(
                    _CORPUS.corpus_id,
                    _ARTEFACT,
                    (SOURCE_BYTES,),
                    max_source_bytes=len(SOURCE_BYTES),
                    parse_outcome=_OUTCOME,
                    parsed_representation=representation,
                )
            elapsed = perf_counter() - started_at

        assert elapsed < 1.0
        assert raised.value.retryable is True
        assert "locked" not in str(raised.value).lower()
        assert str(database_path) not in str(raised.value)
        assert str(source_dir) not in str(raised.value)
        assert [record["event"] for record in logs] == [
            "source_persistence_started",
            "source_persistence_failed",
        ]
        assert logs[-1]["error_category"] == "concurrency"
        assert logs[-1]["retryable"] is True
        assert str(database_path) not in repr(logs)
        assert str(source_dir) not in repr(logs)
        assert (
            repository.get_source_artefact(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert (
            repository.get_parsed_representation(
                _CORPUS.corpus_id,
                PARSED_RECORD_ID,
            )
            is None
        )
        assert repository.get_parse_outcome(_CORPUS.corpus_id, PARSE_ID) is None
        assert not (source_dir / _ARTEFACT.storage_key).exists()
    finally:
        release_path.touch()
        try:
            child_stdout, child_stderr = writer.communicate(timeout=12)
        except subprocess.TimeoutExpired:
            writer.kill()
            child_stdout, child_stderr = writer.communicate()
            raise

    try:
        assert writer.returncode == 0, child_stderr
        assert child_stderr == ""
        assert "source_persistence_completed" in child_stdout
        for private_value in (
            str(database_path),
            str(source_dir),
            _ARTEFACT.storage_key,
            SOURCE_HASH.digest,
            SOURCE_BYTES.decode(),
        ):
            assert private_value not in child_stdout

        assert (
            repository.read_source_bytes(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == SOURCE_BYTES
        )
        assert (
            repository.get_parsed_representation(
                _CORPUS.corpus_id,
                PARSED_RECORD_ID,
            )
            == representation
        )
        assert repository.get_parse_outcome(_CORPUS.corpus_id, PARSE_ID) == _OUTCOME

        repository.persist_source_artefact(
            _CORPUS.corpus_id,
            _ARTEFACT,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
            parse_outcome=_OUTCOME,
            parsed_representation=representation,
        )
        assert (
            repository.read_source_bytes(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == SOURCE_BYTES
        )
    finally:
        repository.close()
