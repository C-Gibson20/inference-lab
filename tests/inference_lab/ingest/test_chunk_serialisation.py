"""Tests for canonical chunk serialisation and strict reconstruction."""

from typing import cast

import pytest

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.ingest import (
    ChunkingRequest,
    FakeChunkTokeniser,
    InvalidChunkSerialisationError,
    build_chunk_set,
    chunk_payload,
    chunk_set_payload,
    deserialise_chunk,
    deserialise_chunk_set,
    serialise_chunk,
    serialise_chunk_set,
)
from inference_lab.provenance import CorpusId

from .fixtures import fixture_parsed_document  # noqa: TID252


def _chunk_set():
    """Build a small complete chunk set for wire-format tests."""
    document = fixture_parsed_document()
    return build_chunk_set(
        ChunkingRequest(
            corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000301"),
            parsed_document=document,
            tokeniser=FakeChunkTokeniser(),
        )
    )


def _tampered_bytes(payload: JsonObject) -> bytes:
    """Encode one deliberately modified payload canonically."""
    return canonical_json_bytes(payload)


def test_chunk_round_trip_is_byte_stable() -> None:
    chunk = _chunk_set().chunks[0]

    encoded = serialise_chunk(chunk)
    reconstructed = deserialise_chunk(encoded)

    assert encoded.endswith(b"\n")
    assert reconstructed == chunk
    assert serialise_chunk(reconstructed) == encoded


def test_chunk_set_round_trip_preserves_complete_lineage() -> None:
    chunk_set = _chunk_set()

    encoded = serialise_chunk_set(chunk_set)
    reconstructed = deserialise_chunk_set(encoded)

    assert reconstructed == chunk_set
    assert serialise_chunk_set(reconstructed) == encoded
    assert reconstructed.lineage == chunk_set.lineage


def test_serialisation_rejects_unknown_and_missing_fields() -> None:
    chunk = _chunk_set().chunks[0]
    unknown = chunk_payload(chunk)
    unknown["unexpected"] = True
    missing = chunk_payload(chunk)
    del missing["text"]

    with pytest.raises(InvalidChunkSerialisationError):
        deserialise_chunk(_tampered_bytes(unknown))
    with pytest.raises(InvalidChunkSerialisationError):
        deserialise_chunk(_tampered_bytes(missing))


def test_serialisation_rejects_unknown_schema_and_duplicate_keys() -> None:
    chunk = _chunk_set().chunks[0]
    payload = chunk_payload(chunk)
    payload["payload_schema_version"] = "2.0.0"

    with pytest.raises(InvalidChunkSerialisationError):
        deserialise_chunk(_tampered_bytes(payload))
    with pytest.raises(InvalidChunkSerialisationError):
        deserialise_chunk(
            b'{"payload_schema_version":"1.0.0","payload_schema_version":"1.0.0"}'
        )


def test_serialisation_rejects_payload_tampering_and_noncanonical_ids() -> None:
    chunk = _chunk_set().chunks[0]
    changed_text = chunk_payload(chunk)
    changed_text["text"] = "tampered"
    changed_id = chunk_payload(chunk)
    changed_id["document_id"] = "00000000-0000-4000-8000-00000000030A"

    with pytest.raises(InvalidChunkSerialisationError):
        deserialise_chunk(_tampered_bytes(changed_text))
    with pytest.raises(InvalidChunkSerialisationError):
        deserialise_chunk(_tampered_bytes(changed_id))


def test_chunk_set_rejects_configuration_disagreement() -> None:
    payload = chunk_set_payload(_chunk_set())
    configuration = cast(JsonObject, payload["configuration_identity"])
    configuration["digest"] = "0" * 64

    with pytest.raises(InvalidChunkSerialisationError):
        deserialise_chunk_set(_tampered_bytes(payload))


def test_chunk_set_rejects_incomplete_source_catalogue() -> None:
    payload = chunk_set_payload(_chunk_set())
    lineage = cast(JsonObject, payload["lineage"])
    lineage["source_spans"] = []

    with pytest.raises(InvalidChunkSerialisationError):
        deserialise_chunk_set(_tampered_bytes(payload))


def test_deserialisation_never_exposes_source_payload_in_failures() -> None:
    payload = chunk_payload(_chunk_set().chunks[0])
    payload["text"] = "private source payload"

    with pytest.raises(InvalidChunkSerialisationError) as error:
        deserialise_chunk(_tampered_bytes(payload))

    assert "private source payload" not in str(error.value)
