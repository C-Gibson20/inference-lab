#!/usr/bin/env python3
"""Run and publish a frozen local retrieval baseline."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
import json
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any

from inference_lab.evaluation import (
    HeldOutBaselineRequest,
    compare_heldout_baseline_publication,
    publish_heldout_baseline_report,
    run_held_out_baseline,
    verify_heldout_baseline_publication,
)
from inference_lab.evaluation.heldout_publication import (
    HeldOutPublicationMismatchError,
)
from inference_lab.evaluation.privacy import validate_local_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and publish a frozen local retrieval baseline."
    )
    parser.add_argument(
        "--request-factory",
        required=True,
        metavar="MODULE:FUNCTION",
        help="Importable module or .py file and zero-argument baseline request factory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIRECTORY",
        help="New local directory for the aggregate baseline report.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        metavar="DIRECTORY",
        help="Existing report bundle used for byte-identical repeat comparison.",
    )
    return parser


def _load_module(module_spec: str) -> ModuleType:
    """Load one deployment-local module without exposing import failures."""
    if module_spec.endswith(".py"):
        module_path = Path(module_spec)
        loader = importlib.util.spec_from_file_location(
            "inference_lab_local_held_out_request",
            module_path,
        )
        if loader is None or loader.loader is None:
            raise RuntimeError
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        return module
    return importlib.import_module(module_spec)


def _load_request(factory_spec: str) -> HeldOutBaselineRequest:
    """Load and validate one complete baseline request."""
    module_spec, separator, attribute = factory_spec.rpartition(":")
    if not separator or not module_spec or not attribute:
        raise ValueError
    factory: Any = getattr(_load_module(module_spec), attribute)
    if not callable(factory):
        raise TypeError
    request = factory()
    if not isinstance(request, HeldOutBaselineRequest):
        raise TypeError
    return request


def _validate_destinations(
    output: Path,
    compare_to: Path | None,
) -> tuple[Path, Path | None]:
    """Validate local, distinct destinations before running private queries."""
    raw_output = Path(output).expanduser()
    local_output = validate_local_path(output, field_name="output")
    if raw_output.is_symlink() or local_output.exists() or local_output.is_symlink():
        raise FileExistsError
    local_comparison: Path | None = None
    if compare_to is not None:
        raw_comparison = Path(compare_to).expanduser()
        local_comparison = validate_local_path(compare_to, field_name="compare_to")
        if raw_comparison.is_symlink() or local_comparison.is_symlink():
            raise ValueError
        if local_output == local_comparison:
            raise ValueError
        verify_heldout_baseline_publication(local_comparison)
    return local_output, local_comparison


def _run(args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    """Run, optionally compare, and publish one aggregate baseline report."""
    output, compare_to = _validate_destinations(args.output, args.compare_to)
    request = _load_request(args.request_factory)
    report = run_held_out_baseline(
        request.contract,
        request.cases,
        request.run_channel,
        limitations=request.limitations,
    )
    comparison: dict[str, object]
    if compare_to is None:
        comparison = {"status": "not_requested"}
    else:
        try:
            equivalent = compare_heldout_baseline_publication(compare_to, report)
        except HeldOutPublicationMismatchError:
            equivalent = False
        comparison = {"equivalent": equivalent}
        if not equivalent:
            return {
                "comparison": comparison,
                "report_digest": report.digest,
                "run_fingerprint": report.contract.fingerprint,
            }, False
    manifest = publish_heldout_baseline_report(output, report)
    return {
        "comparison": comparison,
        "outcome": report.outcome.value,
        "report_digest": manifest.report_digest,
        "run_fingerprint": manifest.run_fingerprint,
    }, True


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and return a process exit status."""
    args = _parser().parse_args(argv)
    try:
        with redirect_stdout(io.StringIO()):
            result, published = _run(args)
    except Exception:
        print("held-out baseline runner failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if published else 1


if __name__ == "__main__":
    raise SystemExit(main())
