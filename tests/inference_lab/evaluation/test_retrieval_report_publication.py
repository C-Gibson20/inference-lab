"""Tests for atomic retrieval-evaluation publication."""

import hashlib
from pathlib import Path

import pytest

from inference_lab.evaluation import (
    RetrievalEvaluationPublicationError,
    deserialise_case_results,
    deserialise_retrieval_manifest,
    deserialise_retrieval_report,
    deserialise_run_artifact,
    publish_retrieval_evaluation,
    retrieval_result_digest,
)
from inference_lab.evaluation.retrieval_report_publication import (
    _rename_directory_without_replacement,
    atomic_write,
)

from .test_retrieval_report_codec import evaluation_run  # noqa: TID252


def publication_run(tmp_path: Path):
    """Build the shared run fixture below an explicitly created directory."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    return evaluation_run(tmp_path)


def test_publish_writes_and_reopens_complete_immutable_bundle(tmp_path: Path) -> None:
    """Publication writes every required artefact with verified inventory."""
    run = publication_run(tmp_path / "fixture")
    destination = tmp_path / "published"
    markdown = "# Retrieval evaluation report\n\n- status: `complete`\n"

    manifest = publish_retrieval_evaluation(destination, run, markdown)

    assert tuple(sorted(path.name for path in destination.iterdir())) == (
        "cases.jsonl",
        "manifest.json",
        "report.json",
        "report.md",
        "run.json",
    )
    assert (
        deserialise_run_artifact((destination / "run.json").read_bytes()).identity
        == run.identity
    )
    assert (
        deserialise_case_results(
            (destination / "cases.jsonl").read_bytes(), run.identity
        )
        == run.case_results
    )
    assert (
        deserialise_retrieval_report(
            (destination / "report.json").read_bytes()
        ).aggregate
        == run.aggregate
    )
    assert (
        deserialise_retrieval_manifest((destination / "manifest.json").read_bytes())
        == manifest
    )
    assert manifest.result_digest == retrieval_result_digest(run)
    for artefact in manifest.artefacts:
        content = (destination / artefact.name).read_bytes()
        assert artefact.size_bytes == len(content)
        assert artefact.checksum == "sha256:" + hashlib.sha256(content).hexdigest()
    assert (destination / "report.md").read_text(encoding="utf-8") == markdown


def test_publish_rejects_remote_and_existing_destinations(tmp_path: Path) -> None:
    """Publication never treats an existing directory as replaceable."""
    run = publication_run(tmp_path / "fixture")

    with pytest.raises(ValueError, match="local filesystem path"):
        publish_retrieval_evaluation("https://example.test/run", run, "# report\n")

    destination = tmp_path / "published"
    destination.mkdir()
    marker = destination / "marker"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        publish_retrieval_evaluation(destination, run, "# report\n")
    assert marker.read_text(encoding="utf-8") == "existing"


def test_publish_cleans_staging_after_write_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged-byte failure cannot expose a partial destination."""
    run = publication_run(tmp_path / "fixture")
    destination = tmp_path / "published"
    real_atomic_write = atomic_write

    def corrupt_manifest(path: str | Path, content: bytes) -> None:
        real_atomic_write(
            path,
            b"corrupted" if Path(path).name == "manifest.json" else content,
        )

    monkeypatch.setattr(
        "inference_lab.evaluation.retrieval_report_publication.atomic_write",
        corrupt_manifest,
    )

    with pytest.raises(RetrievalEvaluationPublicationError, match="publication failed"):
        publish_retrieval_evaluation(destination, run, "# report\n")

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".published.*.staging"))


def test_publish_preserves_competing_destination_at_atomic_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent destination remains untouched when promotion loses the race."""
    run = publication_run(tmp_path / "fixture")
    destination = tmp_path / "published"

    def publish_after_competitor(source: Path, target: Path) -> None:
        destination.mkdir()
        _rename_directory_without_replacement(source, target)

    monkeypatch.setattr(
        "inference_lab.evaluation.retrieval_report_publication."
        "_rename_directory_without_replacement",
        publish_after_competitor,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        publish_retrieval_evaluation(destination, run, "# report\n")

    assert destination.is_dir()
    assert not tuple(destination.iterdir())
    assert not tuple(tmp_path.glob(".published.*.staging"))
