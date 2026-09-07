#!/usr/bin/env python3
"""Run and publish one locally assembled corpus-audit request.

The request factory is intentionally supplied by the deployment.  This keeps
storage-specific snapshot reading and private source handling outside the
storage-neutral evaluation package.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from inference_lab.evaluation import (
    CorpusAuditRequest,
    publish_audit_report,
    run_and_publish_audit,
    run_corpus_audit,
    serialise_audit_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and publish one local corpus audit."
    )
    parser.add_argument(
        "--request-factory",
        required=True,
        metavar="MODULE:FUNCTION",
        help="Importable module or .py file and zero-argument request factory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIRECTORY",
        help="New local directory for report.json, report.md, and manifest.json.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        metavar="DIRECTORY",
        help="Existing report directory whose report.json must byte-match this run.",
    )
    return parser


def _load_module(module_spec: str) -> ModuleType:
    if module_spec.endswith(".py"):
        module_path = Path(module_spec)
        loader = importlib.util.spec_from_file_location(
            "inference_lab_local_audit_request", module_path
        )
        if loader is None or loader.loader is None:
            raise RuntimeError
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        return module
    return importlib.import_module(module_spec)


def _load_request(factory_spec: str) -> CorpusAuditRequest:
    module_spec, separator, attribute = factory_spec.rpartition(":")
    if not separator or not module_spec or not attribute:
        raise ValueError
    factory: Any = getattr(_load_module(module_spec), attribute)
    if not callable(factory):
        raise TypeError
    request = factory()
    if not isinstance(request, CorpusAuditRequest):
        raise TypeError
    return request


def _run(args: argparse.Namespace) -> dict[str, object]:
    request = _load_request(args.request_factory)
    if args.compare_to is not None:
        # A comparison target is checked after the new report is aggregated,
        # before publication can create a destination with mismatched bytes.
        existing_bytes = (args.compare_to / "report.json").read_bytes()
        report = run_corpus_audit(request)
        if serialise_audit_report(report) != existing_bytes:
            raise RuntimeError
        manifest = publish_audit_report(args.output, report)
        report_id = report.report_id
        input_digest = report.input_digest
        report_digest = manifest.report_digest
    else:
        run_result = run_and_publish_audit(request, args.output)
        report_id = run_result.report.report_id
        input_digest = run_result.report.input_digest
        report_digest = run_result.manifest.report_digest
    return {
        "input_digest": input_digest,
        "report_digest": report_digest,
        "report_id": report_id,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and return a process exit status."""
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except Exception:
        print("corpus audit runner failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
