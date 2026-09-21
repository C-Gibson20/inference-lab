"""Tests for canonical retrieval-evaluation artefacts."""

import hashlib
import json
from pathlib import Path

import pytest

from inference_lab._json import canonical_json_bytes
from inference_lab.evaluation import (
    PublishedArtifact,
    PublishedRetrievalEvaluationManifest,
    RetrievalEvaluationReport,
    deserialise_case_results,
    deserialise_retrieval_manifest,
    deserialise_retrieval_report,
    deserialise_run_artifact,
    retrieval_result_digest,
    run_retrieval_evaluation,
    serialise_case_results,
    serialise_retrieval_manifest,
    serialise_retrieval_report,
    serialise_run_artifact,
)

from .test_retrieval_execution import (  # noqa: TID252
    FakeAdapter,
    request_for,
    system_identity,
)


def checksum(seed: str) -> str:
    """Return a deterministic test checksum."""
    return f"sha256:{hashlib.sha256(seed.encode()).hexdigest()}"


def evaluation_run(tmp_path: Path):
    """Build a small deterministic run with both empty and ranked outcomes."""
    adapter = FakeAdapter(
        system_identity(),
        {"case-001": "completed", "case-002": "empty"},
    )
    return run_retrieval_evaluation(
        request_for(tmp_path, adapter, case_ids=("case-001", "case-002"))
    )


def test_all_canonical_artifacts_round_trip_without_payload_text(
    tmp_path: Path,
) -> None:
    """All strict codecs preserve safe values and omit question content."""
    run = evaluation_run(tmp_path)
    result_digest = retrieval_result_digest(run)
    assert isinstance(run.identity.run_fingerprint, str)
    run_fingerprint = run.identity.run_fingerprint
    limitations = ("fixture adapter returns bounded synthetic results",)

    run_bytes = serialise_run_artifact(
        run.identity,
        result_digest,
        limitations,
    )
    cases_bytes = serialise_case_results(run)
    report = RetrievalEvaluationReport(
        run_fingerprint=run_fingerprint,
        result_digest=result_digest,
        aggregate=run.aggregate,
        limitations=limitations,
    )
    report_bytes = serialise_retrieval_report(report)
    manifest = PublishedRetrievalEvaluationManifest(
        run_fingerprint=run_fingerprint,
        result_digest=result_digest,
        artefacts=tuple(
            PublishedArtifact(name, index + 1, checksum(name))
            for index, name in enumerate(
                ("run.json", "cases.jsonl", "report.json", "report.md")
            )
        ),
    )
    manifest_bytes = serialise_retrieval_manifest(manifest)

    assert deserialise_run_artifact(run_bytes).identity == run.identity
    assert deserialise_run_artifact(run_bytes).result_digest == result_digest
    assert deserialise_case_results(cases_bytes, run.identity) == run.case_results
    assert deserialise_retrieval_report(report_bytes) == report
    assert deserialise_retrieval_manifest(manifest_bytes) == manifest
    assert b"question" not in run_bytes + cases_bytes + report_bytes
    assert b"candidate content sentinel" not in run_bytes + cases_bytes + report_bytes
    assert run_bytes.endswith(b"\n") and not run_bytes.endswith(b"\n\n")
    assert report_bytes.endswith(b"\n") and not report_bytes.endswith(b"\n\n")
    assert cases_bytes.endswith(b"\n") and not cases_bytes.endswith(b"\n\n")


def test_readers_reject_unknown_duplicate_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    """Readers reject structural ambiguity before constructing contracts."""
    run = evaluation_run(tmp_path)
    run_bytes = serialise_run_artifact(run.identity, retrieval_result_digest(run))

    duplicate_key = (
        b'{"identity":{},"identity":{},"limitations":[],"result_digest":"'
        + retrieval_result_digest(run).encode()
        + b'"}\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        deserialise_run_artifact(duplicate_key)

    payload = json.loads(run_bytes)
    payload["unknown"] = True
    unknown_field = canonical_json_bytes(payload)
    with pytest.raises(ValueError, match="fields do not match"):
        deserialise_run_artifact(unknown_field)

    with pytest.raises(ValueError, match="canonical"):
        deserialise_run_artifact(run_bytes.replace(b"\n", b" \n", 1))


def test_case_reader_rejects_identity_order_and_trailing_newline_errors(
    tmp_path: Path,
) -> None:
    """JSONL records must match the selected run identity exactly."""
    run = evaluation_run(tmp_path)
    cases_bytes = serialise_case_results(run)
    lines = cases_bytes.decode().splitlines()
    assert isinstance(run.identity.run_fingerprint, str)

    with pytest.raises(ValueError, match="case order"):
        deserialise_case_results(
            (lines[1] + "\n" + lines[0] + "\n").encode(),
            run.identity,
        )

    with pytest.raises(ValueError, match="exactly one trailing newline"):
        deserialise_case_results(cases_bytes + b"\n", run.identity)

    foreign = lines[0].replace(run.identity.run_fingerprint, checksum("foreign"))
    with pytest.raises(ValueError, match="run identity"):
        deserialise_case_results(
            (foreign + "\n" + lines[1] + "\n").encode(),
            run.identity,
        )
