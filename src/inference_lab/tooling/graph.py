import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog

from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / ".dev"
OUTPUT_DIR = OUTPUT_ROOT / "graphify-out"
GRAPHIFY_ROOT = REPO_ROOT
GRAPHIFY_SCOPE = ("src", "tests")
GRAPHIFY_TIMEOUT_SECONDS = 300
REPOSITORY_IDENTIFIER = "."
OUTPUT_IDENTIFIER = ".dev/graphify-out"


class GraphifyFailure(RuntimeError):
    """Represent a safe, structured Graphify failure without raw diagnostics."""

    def __init__(
        self,
        category: str,
        *,
        stage: str,
        returncode: int | None = None,
    ) -> None:
        super().__init__(category)
        self.category = category
        self.stage = stage
        self.returncode = returncode


def parse_args() -> argparse.Namespace:
    """Parse standalone graph-tool options without loading application settings."""
    parser = argparse.ArgumentParser(description="Generate the local code graph.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Minimum log level (default: INFO).",
    )
    parser.add_argument(
        "--log-format",
        choices=("auto", "console", "json"),
        default="auto",
        help="Log renderer (default: auto).",
    )
    return parser.parse_args()


def require_graphify(log: structlog.stdlib.BoundLogger) -> bool:
    """Check if the ``graphify`` tool is available in the system PATH."""
    if shutil.which("graphify") is None:
        log.error("graphify_not_found", hint="uv tool install graphifyy")
        return False

    return True


def require_scan_scope(log: structlog.stdlib.BoundLogger) -> bool:
    """Check that every intended Graphify scope directory exists."""
    missing = [root for root in GRAPHIFY_SCOPE if not (GRAPHIFY_ROOT / root).is_dir()]

    if missing:
        log.error("scan_scope_missing", missing_roots=missing)
        return False

    return True


def run_graphify_command(
    arguments: list[str],
    *,
    stage: str = "command",
) -> subprocess.CompletedProcess[str]:
    """Run Graphify from the repository root with a bounded execution time."""
    try:
        return subprocess.run(
            ["graphify", *arguments],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=GRAPHIFY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise GraphifyFailure("timeout", stage=stage) from exc
    except OSError as exc:
        raise GraphifyFailure("start_failed", stage=stage) from exc


def run_graphify_stage(
    arguments: list[str],
    *,
    stage: str,
) -> subprocess.CompletedProcess[str]:
    """Run one Graphify stage and discard diagnostics from failed commands."""
    result = run_graphify_command(arguments, stage=stage)
    if result.returncode != 0:
        raise GraphifyFailure(
            "subprocess_failed",
            stage=stage,
            returncode=result.returncode,
        )

    return result


def validate_graph_json(graph_path: Path) -> None:
    """Validate the minimum Graphify graph structure before publication."""
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphifyFailure("invalid_output", stage="extract") from exc

    if (
        not isinstance(graph, dict)
        or not isinstance(graph.get("nodes"), list)
        or not isinstance(graph.get("edges"), list)
    ):
        raise GraphifyFailure("invalid_output", stage="extract")


def validate_html_output(html_path: Path, *, stage: str) -> None:
    """Require an HTML output containing at least one non-whitespace byte."""
    try:
        if not html_path.read_text(encoding="utf-8").strip():
            raise GraphifyFailure("invalid_output", stage=stage)
    except (OSError, UnicodeDecodeError) as exc:
        raise GraphifyFailure("invalid_output", stage=stage) from exc


def build_staged_graph(
    log: structlog.stdlib.BoundLogger,
) -> tuple[Path, Path]:
    stage_root = Path(tempfile.mkdtemp(prefix=".graphify-refresh-", dir=OUTPUT_ROOT))
    staged_dir = stage_root / "graphify-out"

    log.info(
        "graph_generation_started",
        graphify_root=REPOSITORY_IDENTIFIER,
        included_scope=list(GRAPHIFY_SCOPE),
        output_dir=OUTPUT_IDENTIFIER,
    )

    try:
        run_graphify_stage(
            [
                "extract",
                str(GRAPHIFY_ROOT),
                "--code-only",
                "--no-cluster",
                "--max-workers",
                "1",
                "--out",
                str(stage_root),
            ],
            stage="extract",
        )

        graph_path = staged_dir / "graph.json"
        if not graph_path.is_file():
            raise GraphifyFailure("missing_output", stage="extract")
        validate_graph_json(graph_path)

        run_graphify_stage(
            [
                "cluster-only",
                str(stage_root),
                "--no-label",
            ],
            stage="cluster",
        )

        report_path = staged_dir / "graph.html"
        if not report_path.is_file():
            raise GraphifyFailure("missing_output", stage="cluster")
        validate_html_output(report_path, stage="cluster")

        run_graphify_stage(
            [
                "tree",
                "--graph",
                str(graph_path),
                "--output",
                str(staged_dir / "GRAPH_TREE.html"),
                "--root",
                ".",
            ],
            stage="tree",
        )

        tree_path = staged_dir / "GRAPH_TREE.html"
        if not tree_path.is_file():
            raise GraphifyFailure("missing_output", stage="tree")
        validate_html_output(tree_path, stage="tree")

        return stage_root, staged_dir

    except GraphifyFailure:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise GraphifyFailure("generation_failed", stage="build") from exc


def publish_graph(stage_root: Path, staged_dir: Path) -> None:
    """Publish a validated graph while retaining the old output on failure."""
    previous_dir = OUTPUT_ROOT / ".graphify-previous"
    moved_previous = False
    published = False

    try:
        if previous_dir.exists():
            raise GraphifyFailure("publish_failed", stage="publish")

        if OUTPUT_DIR.exists():
            os.replace(OUTPUT_DIR, previous_dir)
            moved_previous = True

        os.replace(staged_dir, OUTPUT_DIR)
        published = True

    except Exception as exc:
        if moved_previous and not OUTPUT_DIR.exists() and previous_dir.exists():
            try:
                os.replace(previous_dir, OUTPUT_DIR)
            except OSError:
                # Keep the backup in place if rollback itself fails.
                pass
        if isinstance(exc, GraphifyFailure):
            raise
        raise GraphifyFailure("publish_failed", stage="publish") from exc

    finally:
        if published:
            # Cleanup is deliberately best-effort after the new output is live.
            shutil.rmtree(previous_dir, ignore_errors=True)
        shutil.rmtree(stage_root, ignore_errors=True)


def refresh_graph(log: structlog.stdlib.BoundLogger) -> bool:
    """Build, validate, and publish a refreshed graph."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        stage_root, staged_dir = build_staged_graph(log)
        publish_graph(stage_root, staged_dir)
    except GraphifyFailure as exc:
        log.error(
            "graph_refresh_failed",
            failure_category=exc.category,
            stage=exc.stage,
            returncode=exc.returncode,
        )
        return False
    except Exception:
        log.error(
            "graph_refresh_failed",
            failure_category="unexpected_failure",
            stage="refresh",
            returncode=None,
        )
        return False

    log.info("graph_refresh_complete", output_dir=OUTPUT_IDENTIFIER)
    return True


def main() -> int:
    """Run the local code-graph refresh command."""
    args = parse_args()
    configure_logging(LoggingConfig(level=args.log_level, format=args.log_format))
    log = structlog.get_logger(__name__)

    if not require_graphify(log):
        return 1

    if not require_scan_scope(log):
        return 1

    if not refresh_graph(log):
        return 1

    return 0
