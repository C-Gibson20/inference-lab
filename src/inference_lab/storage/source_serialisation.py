"""Canonical encoding for durable managed source-artefact contracts."""

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.storage.source_records import SourceArtefact


def source_artefact_payload(artefact: SourceArtefact) -> JsonObject:
    """Return canonical JSON-compatible source-artefact data."""
    if not isinstance(artefact, SourceArtefact):
        raise TypeError("artefact must be a SourceArtefact.")
    return {
        "record_schema_version": artefact.record_schema_version,
        "document_id": artefact.document_id.as_str(),
        "document_version_id": artefact.document_version_id.as_str(),
        "content_hash": {
            "algorithm": artefact.content_hash.algorithm.value,
            "digest": artefact.content_hash.digest,
        },
        "byte_count": artefact.byte_count,
        "media_type": artefact.media_type,
        "storage_key": artefact.storage_key,
    }


def serialise_source_artefact(artefact: SourceArtefact) -> bytes:
    """Serialise one source artefact as canonical UTF-8 JSON."""
    return canonical_json_bytes(source_artefact_payload(artefact))
