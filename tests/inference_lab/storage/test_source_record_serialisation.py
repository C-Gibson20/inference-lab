"""Tests for deterministic canonical source-artefact serialisation."""

import json
import subprocess
import sys

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import SourceArtefact, serialise_source_artefact
from inference_lab.storage.source_serialisation import source_artefact_payload
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    SOURCE_BYTES,
    SOURCE_HASH,
)


def _source_artefact() -> SourceArtefact:
    return SourceArtefact(
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        content_hash=SOURCE_HASH,
        byte_count=len(SOURCE_BYTES),
        media_type="text/plain",
        storage_key=f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}",
    )


def test_source_artefact_serialisation_is_deterministic_canonical_json() -> None:
    artefact = _source_artefact()

    first = serialise_source_artefact(artefact)
    second = serialise_source_artefact(artefact)
    payload = json.loads(first)

    assert first == second
    assert first.endswith(b"\n")
    assert payload["record_schema_version"] == "1.0.0"
    assert payload["content_hash"] == {
        "algorithm": "sha256",
        "digest": SOURCE_HASH.digest,
    }
    assert "source_path" not in payload


def test_source_artefact_payload_rejects_wrong_runtime_type() -> None:
    with pytest.raises(TypeError, match="SourceArtefact"):
        source_artefact_payload(object())  # type: ignore[arg-type]


def test_source_contract_values_and_serialisation_emit_no_logs() -> None:
    with capture_logs() as logs:
        serialise_source_artefact(_source_artefact())

    assert logs == []


def test_storage_source_contract_does_not_import_ingest() -> None:
    script = """
import sys
import inference_lab.storage.source_records
raise SystemExit(any(name.startswith('inference_lab.ingest') for name in sys.modules))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
