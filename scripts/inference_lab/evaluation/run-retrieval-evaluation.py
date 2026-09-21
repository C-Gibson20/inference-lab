#!/usr/bin/env python3
"""Run and publish one locally assembled retrieval evaluation.

The deployment supplies the zero-argument request factory.  Retrieval query
construction and private dataset handling remain inside that deployment-local
adapter and never become command-line or result payloads.
"""

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
    RetrievalEvaluationRequest,
    compare_retrieval_evaluations,
    load_published_retrieval_evaluation,
    publish_retrieval_evaluation,
    render_retrieval_report,
    retrieval_result_digest,
    run_retrieval_evaluation,
)
from inference_lab.evaluation.retrieval_report_contracts import (
    RetrievalEvaluationReport,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and publish one local retrieval evaluation."
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
        help="New local directory for the five evaluation artefacts.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        metavar="DIRECTORY",
        help="Existing published evaluation used for semantic comparison.",
    )
    return parser


def _load_module(module_spec: str) -> ModuleType:
    """Load one deployment-local module without exposing its failures."""
    if module_spec.endswith(".py"):
        module_path = Path(module_spec)
        loader = importlib.util.spec_from_file_location(
            "inference_lab_local_retrieval_request",
            module_path,
        )
        if loader is None or loader.loader is None:
            raise RuntimeError
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        return module
    return importlib.import_module(module_spec)


def _load_request(factory_spec: str) -> RetrievalEvaluationRequest:
    """Load and validate one zero-argument retrieval request factory."""
    module_spec, separator, attribute = factory_spec.rpartition(":")
    if not separator or not module_spec or not attribute:
        raise ValueError
    factory: Any = getattr(_load_module(module_spec), attribute)
    if not callable(factory):
        raise TypeError
    request = factory()
    if not isinstance(request, RetrievalEvaluationRequest):
        raise TypeError
    return request


def _fingerprint(run: object) -> str:
    """Return a validated run fingerprint for safe CLI output."""
    identity = getattr(run, "identity")
    value = getattr(identity, "run_fingerprint")
    if not isinstance(value, str):
        raise ValueError
    return value


def _run(args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    """Run, optionally compare, and publish one evaluation."""
    request = _load_request(args.request_factory)
    run = run_retrieval_evaluation(request)
    run_fingerprint = _fingerprint(run)
    result_digest = retrieval_result_digest(run)

    comparison: dict[str, object]
    if args.compare_to is None:
        comparison = {"status": "not_requested"}
    else:
        previous_run = load_published_retrieval_evaluation(args.compare_to)
        comparison_result = compare_retrieval_evaluations(previous_run, run)
        comparison = comparison_result.as_dict()
        if not comparison_result.equivalent:
            return {
                "run_fingerprint": run_fingerprint,
                "result_digest": result_digest,
                "comparison": comparison,
            }, False

    report = RetrievalEvaluationReport(
        run_fingerprint=run_fingerprint,
        result_digest=result_digest,
        aggregate=run.aggregate,
    )
    markdown = render_retrieval_report(report, run.identity)
    manifest = publish_retrieval_evaluation(
        args.output,
        run,
        markdown,
    )
    return {
        "run_fingerprint": run_fingerprint,
        "result_digest": manifest.result_digest,
        "comparison": comparison,
    }, True


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and return a process exit status."""
    args = _parser().parse_args(argv)
    try:
        with redirect_stdout(io.StringIO()):
            result, published = _run(args)
    except Exception:
        print("retrieval evaluation runner failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if published else 1


if __name__ == "__main__":
    raise SystemExit(main())
