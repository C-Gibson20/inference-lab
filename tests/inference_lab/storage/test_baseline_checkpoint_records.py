"""Tests for storage-owned baseline-ingest checkpoint records."""

import pytest

from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
    RecordId,
)
from inference_lab.storage.baseline_checkpoint_records import (
    BaselineCheckpointIdentityRecord,
    BaselineCheckpointRecord,
    BaselineCheckpointRecordFailureCategory,
    BaselineCheckpointRecordIdentityKind,
    BaselineCheckpointRecordStage,
    BaselineCheckpointRecordStatus,
    BaselineManifestRecord,
    BaselineManifestRecordState,
)
from inference_lab.storage.versioning import ComponentKind

_PROPOSED_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_CONFIGURATION = ConfigurationIdentity(
    algorithm=DigestAlgorithm.SHA256,
    digest="0" * 64,
)
_DISCOVERY_DIGEST = OutputDigest(
    algorithm=DigestAlgorithm.SHA256,
    digest="1" * 64,
)
_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_DOCUMENT_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")
_RECORD_ID = RecordId("77777777-7777-4777-8777-777777777777")


def _record(**overrides: object) -> BaselineManifestRecord:
    values: dict[str, object] = {
        "corpus_id": _CORPUS_ID,
        "proposed_corpus_version": _PROPOSED_VERSION,
        "expected_corpus_version": None,
        "configuration_identity": _CONFIGURATION,
        "required_component_kinds": (ComponentKind("chunks"),),
        "discovery_summary_identity": _DISCOVERY_DIGEST,
        "state": BaselineManifestRecordState.PREPARING,
    }
    values.update(overrides)
    return BaselineManifestRecord(**values)  # type: ignore[arg-type]


def test_manifest_record_canonicalises_component_kinds() -> None:
    record = _record(
        required_component_kinds=(
            ComponentKind("parser"),
            ComponentKind("chunks"),
        )
    )

    assert record.key() == (
        _CORPUS_ID.as_str(),
        _PROPOSED_VERSION.as_str(),
    )
    assert tuple(kind.as_str() for kind in record.required_component_kinds) == (
        "chunks",
        "parser",
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("corpus_id", "not-an-id"),
        ("configuration_identity", "sha256:" + ("0" * 64)),
        ("discovery_summary_identity", "sha256:" + ("1" * 64)),
        ("state", "preparing"),
    ],
)
def test_manifest_record_rejects_wrong_field_types(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError):
        _record(**{field_name: value})


def test_manifest_record_rejects_equal_proposed_and_expected_versions() -> None:
    with pytest.raises(ValueError, match="must differ"):
        _record(expected_corpus_version=_PROPOSED_VERSION)


def test_manifest_record_rejects_duplicate_component_kinds() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        _record(
            required_component_kinds=(
                ComponentKind("chunks"),
                ComponentKind("chunks"),
            )
        )


def test_checkpoint_record_vocabularies_are_stable() -> None:
    assert BaselineCheckpointRecordStage.PARSE.value == "parse"
    assert BaselineCheckpointRecordStatus.COMPLETED.value == "completed"
    assert (
        BaselineCheckpointRecordIdentityKind.DOCUMENT_VERSION.value
        == "document_version"
    )
    assert BaselineCheckpointRecordFailureCategory.CHUNK_FAILED.value == "chunk_failed"


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (
            BaselineCheckpointRecordIdentityKind.DOCUMENT,
            "55555555-5555-4555-8555-555555555555",
        ),
        (
            BaselineCheckpointRecordIdentityKind.DOCUMENT_VERSION,
            "66666666-6666-4666-8666-666666666666",
        ),
        (
            BaselineCheckpointRecordIdentityKind.RECORD,
            "77777777-7777-4777-8777-777777777777",
        ),
    ],
)
def test_identity_record_validates_typed_values(
    kind: BaselineCheckpointRecordIdentityKind,
    value: str,
) -> None:
    identity = BaselineCheckpointIdentityRecord(kind=kind, value=value)

    assert identity.key() == (kind.value, value)


def test_identity_record_rejects_wrong_identity_domain() -> None:
    with pytest.raises(ValueError):
        BaselineCheckpointIdentityRecord(
            kind=BaselineCheckpointRecordIdentityKind.DOCUMENT,
            value="not-a-uuid",
        )


def test_identity_record_rejects_wrong_field_types() -> None:
    with pytest.raises(TypeError):
        BaselineCheckpointIdentityRecord(
            kind="document",  # type: ignore[arg-type]
            value="55555555-5555-4555-8555-555555555555",
        )

    with pytest.raises(TypeError):
        BaselineCheckpointIdentityRecord(
            kind=BaselineCheckpointRecordIdentityKind.DOCUMENT,
            value=123,  # type: ignore[arg-type]
        )


def _checkpoint_record(**overrides: object) -> BaselineCheckpointRecord:
    values: dict[str, object] = {
        "corpus_id": _CORPUS_ID,
        "proposed_corpus_version": _PROPOSED_VERSION,
        "document_version_id": _DOCUMENT_VERSION_ID,
        "stage": BaselineCheckpointRecordStage.PARSE,
        "configuration_identity": _CONFIGURATION,
        "document_id": _DOCUMENT_ID,
        "input_identities": (
            BaselineCheckpointIdentityRecord(
                kind=BaselineCheckpointRecordIdentityKind.DOCUMENT_VERSION,
                value=_DOCUMENT_VERSION_ID.as_str(),
            ),
        ),
        "output_identities": (
            BaselineCheckpointIdentityRecord(
                kind=BaselineCheckpointRecordIdentityKind.RECORD,
                value=_RECORD_ID.as_str(),
            ),
        ),
        "status": BaselineCheckpointRecordStatus.COMPLETED,
    }
    values.update(overrides)
    return BaselineCheckpointRecord(**values)  # type: ignore[arg-type]


def test_checkpoint_record_canonicalises_identity_order_and_builds_key() -> None:
    document_identity = BaselineCheckpointIdentityRecord(
        kind=BaselineCheckpointRecordIdentityKind.DOCUMENT,
        value=_DOCUMENT_ID.as_str(),
    )
    record_identity = BaselineCheckpointIdentityRecord(
        kind=BaselineCheckpointRecordIdentityKind.RECORD,
        value=_RECORD_ID.as_str(),
    )

    checkpoint = _checkpoint_record(
        input_identities=(record_identity, document_identity),
    )

    assert checkpoint.input_identities == (document_identity, record_identity)
    assert checkpoint.key() == (
        _CORPUS_ID.as_str(),
        _PROPOSED_VERSION.as_str(),
        _DOCUMENT_VERSION_ID.as_str(),
        "parse",
        "sha256:" + ("0" * 64),
    )


def test_started_checkpoint_has_no_outputs() -> None:
    checkpoint = _checkpoint_record(
        output_identities=(),
        status=BaselineCheckpointRecordStatus.STARTED,
    )

    assert checkpoint.status is BaselineCheckpointRecordStatus.STARTED


def test_completed_checkpoint_may_have_zero_outputs() -> None:
    checkpoint = _checkpoint_record(output_identities=())

    assert checkpoint.output_identities == ()


def test_failed_checkpoint_requires_bounded_failure_category() -> None:
    with pytest.raises(ValueError, match="require a failure category"):
        _checkpoint_record(
            output_identities=(),
            status=BaselineCheckpointRecordStatus.FAILED,
        )


def test_failed_checkpoint_rejects_partial_outputs() -> None:
    with pytest.raises(ValueError, match="must not have output identities"):
        _checkpoint_record(
            status=BaselineCheckpointRecordStatus.FAILED,
            failure_category=BaselineCheckpointRecordFailureCategory.PARSE_FAILED,
        )


def test_checkpoint_record_rejects_duplicate_identities() -> None:
    identity = BaselineCheckpointIdentityRecord(
        kind=BaselineCheckpointRecordIdentityKind.RECORD,
        value=_RECORD_ID.as_str(),
    )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        _checkpoint_record(input_identities=(identity, identity))


def test_checkpoint_record_contains_no_payload_or_path() -> None:
    rendered = repr(_checkpoint_record())

    assert "/private" not in rendered
    assert "source_text" not in rendered
    assert "chunk_text" not in rendered
    assert "exception" not in rendered
