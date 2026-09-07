"""Canonical encoding for complete parsed representations."""

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.storage.parsed_records import (
    ParsedRepresentation,
    parsed_representation_output_payload,
)


def parsed_representation_payload(
    representation: ParsedRepresentation,
) -> JsonObject:
    """Return the complete canonical parsed-representation record."""
    if not isinstance(representation, ParsedRepresentation):
        raise TypeError("representation must be a ParsedRepresentation.")
    return {
        **parsed_representation_output_payload(representation),
        "record_id": representation.record_id.as_str(),
        "supersedes_record_id": (
            representation.supersedes_record_id.as_str()
            if representation.supersedes_record_id is not None
            else None
        ),
        "output_digest": {
            "algorithm": representation.output_digest.algorithm.value,
            "digest": representation.output_digest.digest,
        },
    }


def serialise_parsed_representation(representation: ParsedRepresentation) -> bytes:
    """Serialise one parsed representation as canonical UTF-8 JSON."""
    return canonical_json_bytes(parsed_representation_payload(representation))
