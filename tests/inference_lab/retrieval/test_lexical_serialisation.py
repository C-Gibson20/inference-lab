"""Strict serialisation and privacy tests for RET-02 manifests."""

import json

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import (
    InvalidLexicalSerialisationError,
    LexicalIndexConfiguration,
    LexicalIndexManifest,
    deserialise_lexical_manifest,
    serialise_lexical_manifest,
)


def _manifest() -> LexicalIndexManifest:
    """Return one deterministic manifest for malformed-payload tests."""
    configuration = LexicalIndexConfiguration()
    return LexicalIndexManifest(
        corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000001"),
        corpus_version=CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002"),
        generation_id="lexical-0002",
        configuration=configuration,
        configuration_digest=configuration.configuration_identity.digest,
        source_manifest_digest="a" * 64,
        database_checksum="b" * 64,
        member_count=0,
        chunk_count=0,
        identifier_count=0,
    )


@pytest.mark.parametrize(
    "data",
    [
        b"not json",
        b"\xff{}",
        b'{"schema_version":"1.0.0","schema_version":"1.0.0"}',
        b"[]",
    ],
)
def test_manifest_deserialisation_rejects_malformed_json(data: bytes) -> None:
    with pytest.raises(InvalidLexicalSerialisationError):
        deserialise_lexical_manifest(data)


def test_manifest_deserialisation_does_not_echo_or_log_payloads() -> None:
    marker = "private lexical query and source marker"
    payload = json.loads(serialise_lexical_manifest(_manifest()))
    payload["generation_id"] = marker

    with capture_logs() as logs:
        with pytest.raises(InvalidLexicalSerialisationError) as error:
            deserialise_lexical_manifest(json.dumps(payload).encode())

    assert marker not in str(error.value)
    assert logs == []
