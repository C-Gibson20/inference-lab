"""Canonical encoding for durable parse outcomes."""

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.storage.parse_records import ParseFailureRecord, ParseOutcome
from inference_lab.storage.producers import ProducerIdentity


def _producer_payload(producer: ProducerIdentity) -> JsonObject:
    """Return canonical data for one producer identity."""
    return {
        "implementation": producer.implementation,
        "contract_version": producer.contract_version,
        "artefact_identity": producer.artefact_identity,
        "artefact_revision": producer.artefact_revision,
    }


def _failure_payload(failure: ParseFailureRecord) -> JsonObject:
    """Return canonical data for one bounded parser failure."""
    return {
        "category": failure.category.value,
        "severity": failure.severity.value,
        "stage": failure.stage.value,
        "unit_kind": failure.unit_kind.value if failure.unit_kind is not None else None,
        "unit_index": failure.unit_index,
    }


def parse_outcome_payload(outcome: ParseOutcome) -> JsonObject:
    """Return canonical JSON-compatible parse-outcome data."""
    if not isinstance(outcome, ParseOutcome):
        raise TypeError("outcome must be a ParseOutcome.")
    return {
        "record_schema_version": outcome.record_schema_version,
        "parse_id": outcome.parse_id.as_str(),
        "document_id": outcome.document_id.as_str(),
        "document_version_id": outcome.document_version_id.as_str(),
        "status": outcome.status.value,
        "parser": _producer_payload(outcome.parser),
        "normalisation_version": outcome.normalisation_version,
        "ocr_producer": (
            _producer_payload(outcome.ocr_producer)
            if outcome.ocr_producer is not None
            else None
        ),
        "parsed_record_id": (
            outcome.parsed_record_id.as_str()
            if outcome.parsed_record_id is not None
            else None
        ),
        "failures": [_failure_payload(failure) for failure in outcome.failures],
    }


def serialise_parse_outcome(outcome: ParseOutcome) -> bytes:
    """Serialise one parse outcome as canonical UTF-8 JSON."""
    return canonical_json_bytes(parse_outcome_payload(outcome))
