"""Tests for storage-owned baseline-ingest checkpoint records."""

import pytest

from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    OutputDigest,
)
from inference_lab.storage.baseline_checkpoint_records import (
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
