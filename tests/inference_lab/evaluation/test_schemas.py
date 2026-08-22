"""Tests for reusable evaluation schemas."""

import pytest
from pydantic import ValidationError

from inference_lab.evaluation.schemas import EvidenceReference


IDS = {
    "corpus_version": "00000000-0000-4000-8000-000000000002",
    "document_id": "00000000-0000-4000-8000-000000000003",
    "document_version_id": "00000000-0000-4000-8000-000000000004",
    "source_span_id": "00000000-0000-4000-8000-000000000005",
}


def test_source_span_evidence_is_serializable() -> None:
    evidence = EvidenceReference(**IDS)

    assert evidence.model_dump() == IDS | {
        "kind": "source_span",
        "role": "authoritative",
        "record_id": None,
    }


def test_source_span_requires_source_location() -> None:
    with pytest.raises(ValidationError, match="source_span evidence"):
        EvidenceReference(
            corpus_version=IDS["corpus_version"],
            document_id=IDS["document_id"],
        )


def test_provenance_ids_must_be_canonical_uuids() -> None:
    invalid = dict(IDS, document_id="document-1")

    with pytest.raises(ValidationError, match="valid UUID"):
        EvidenceReference(**invalid)


def test_authoritative_document_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be authoritative"):
        EvidenceReference(
            corpus_version=IDS["corpus_version"],
            document_id=IDS["document_id"],
            kind="document",
        )
