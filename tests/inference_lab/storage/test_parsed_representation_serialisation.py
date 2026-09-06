"""Tests for canonical parsed-representation digest and serialisation."""

import hashlib
import json
from dataclasses import replace

import pytest
from structlog.testing import capture_logs

from inference_lab._json import canonical_json_bytes
from inference_lab.provenance import DigestAlgorithm, RecordId
from inference_lab.storage import (
    build_parsed_representation_output_digest,
    serialise_parsed_representation,
)
from inference_lab.storage.parsed_records import parsed_representation_output_payload
from inference_lab.storage.parsed_serialisation import parsed_representation_payload
from tests.inference_lab.storage.source_fixtures import (
    PARSED_RECORD_ID,
    SUPERSEDED_RECORD_ID,
    parsed_representation,
)


def test_output_digest_covers_the_canonical_semantic_payload() -> None:
    representation = parsed_representation()
    expected = hashlib.sha256(
        canonical_json_bytes(parsed_representation_output_payload(representation))
    ).hexdigest()

    assert representation.output_digest.algorithm is DigestAlgorithm.SHA256
    assert representation.output_digest.digest == expected
    assert (
        build_parsed_representation_output_digest(representation)
        == representation.output_digest
    )


def test_semantic_changes_change_the_output_digest() -> None:
    representation = parsed_representation()

    changed = replace(representation, language="fr")

    assert changed.output_digest != representation.output_digest


def test_record_identity_and_supersession_do_not_change_output_digest() -> None:
    other_record = RecordId.from_str("40000000-0000-4000-8000-000000000002")
    representation = parsed_representation()
    reidentified = parsed_representation(record_id=other_record)
    superseding = parsed_representation(
        supersedes_record_id=SUPERSEDED_RECORD_ID,
    )

    assert reidentified.output_digest == representation.output_digest
    assert superseding.output_digest == representation.output_digest


def test_serialisation_is_deterministic_and_complete() -> None:
    representation = parsed_representation(
        supersedes_record_id=SUPERSEDED_RECORD_ID,
    )

    first = serialise_parsed_representation(representation)
    second = serialise_parsed_representation(representation)
    payload = json.loads(first)

    assert first == second
    assert first.endswith(b"\n")
    assert payload["record_schema_version"] == "1.0.0"
    assert payload["record_id"] == PARSED_RECORD_ID.as_str()
    assert payload["supersedes_record_id"] == SUPERSEDED_RECORD_ID.as_str()
    assert payload["source_text"] != payload["text"]
    assert payload["collection_availability"] == {
        "pages": "available",
        "paragraphs": "available",
        "sections": "unavailable",
    }
    assert [block["ordinal"] for block in payload["blocks"]] == [0, 1]
    assert payload["source_locations"][1]["start_offset"] == 9
    assert payload["page_layouts"] == []
    assert payload["output_digest"] == {
        "algorithm": "sha256",
        "digest": representation.output_digest.digest,
    }


def test_digest_payload_excludes_record_and_replacement_identity() -> None:
    payload = parsed_representation_output_payload(parsed_representation())

    assert "record_id" not in payload
    assert "supersedes_record_id" not in payload
    assert "output_digest" not in payload


@pytest.mark.parametrize(
    "function",
    [
        build_parsed_representation_output_digest,
        parsed_representation_output_payload,
        parsed_representation_payload,
        serialise_parsed_representation,
    ],
)
def test_parsed_representation_helpers_reject_wrong_runtime_type(
    function: object,
) -> None:
    with pytest.raises(TypeError, match="ParsedRepresentation"):
        function(object())  # type: ignore[operator]


def test_contract_digest_and_serialisation_emit_no_logs() -> None:
    with capture_logs() as logs:
        representation = parsed_representation()
        serialise_parsed_representation(representation)

    assert logs == []
