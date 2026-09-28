"""EVAL-03 end-to-end acceptance for reproducible retrieval evaluation."""

import json
from pathlib import Path

from .test_retrieval_operator import (  # noqa: TID252
    _script_main,
    _write_factory,
)


def test_identical_reruns_are_byte_identical_across_destinations(
    tmp_path: Path,
    capsys,
) -> None:
    """Different output directories and clocks do not change quality artefacts."""
    factory = tmp_path / "request_factory.py"
    _write_factory(factory)
    first_destination = tmp_path / "run-001"
    second_destination = tmp_path / "run-002"
    main = _script_main()

    assert (
        main(
            [
                "--request-factory",
                f"{factory}:build_request",
                "--output",
                str(first_destination),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "--request-factory",
                f"{factory}:build_request",
                "--compare-to",
                str(first_destination),
                "--output",
                str(second_destination),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    comparison = json.loads(output.out)["comparison"]
    assert comparison["equivalent"] is True
    assert output.err == ""
    for name in (
        "run.json",
        "cases.jsonl",
        "report.json",
        "report.md",
        "manifest.json",
    ):
        assert (first_destination / name).read_bytes() == (
            second_destination / name
        ).read_bytes()


def test_corrupt_compare_target_fails_closed_without_partial_publication(
    tmp_path: Path,
    capsys,
) -> None:
    """A corrupted prior bundle cannot be compared or replaced silently."""
    factory = tmp_path / "request_factory.py"
    _write_factory(factory)
    first_destination = tmp_path / "run-001"
    changed_destination = tmp_path / "run-002"
    main = _script_main()

    assert (
        main(
            [
                "--request-factory",
                f"{factory}:build_request",
                "--output",
                str(first_destination),
            ]
        )
        == 0
    )
    capsys.readouterr()
    report_path = first_destination / "report.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")

    assert (
        main(
            [
                "--request-factory",
                f"{factory}:build_request",
                "--compare-to",
                str(first_destination),
                "--output",
                str(changed_destination),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.strip() == "retrieval evaluation runner failed"
    assert not changed_destination.exists()


def test_published_bundle_contains_no_private_fixture_payloads(
    tmp_path: Path,
    capsys,
) -> None:
    """Questions, candidate text, paths, and exceptions stay out of artefacts."""
    factory = tmp_path / "request_factory.py"
    _write_factory(factory)
    destination = tmp_path / "run-001"

    assert (
        _script_main()(
            [
                "--request-factory",
                f"{factory}:build_request",
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    bundle = b"".join(path.read_bytes() for path in destination.iterdir())
    for forbidden in (
        b"What assay was used?",
        b"candidate content sentinel",
        b"private-source-canary",
        str(tmp_path).encode(),
    ):
        assert forbidden not in bundle
        assert forbidden not in output.out.encode()
