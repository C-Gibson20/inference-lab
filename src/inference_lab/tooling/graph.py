import argparse
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


def run_graphify_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
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
        raise RuntimeError(
            f"Graphify timed out after {GRAPHIFY_TIMEOUT_SECONDS} seconds"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start Graphify: {exc}") from exc


def build_staged_graph(
    log: structlog.stdlib.BoundLogger,
) -> tuple[Path, Path]:
    stage_root = Path(tempfile.mkdtemp(prefix=".graphify-refresh-", dir=OUTPUT_ROOT))
    staged_dir = stage_root / "graphify-out"

    log.info(
        "graph_generation_started",
        graphify_root=str(GRAPHIFY_ROOT),
        included_scope=list(GRAPHIFY_SCOPE),
        output_dir=str(OUTPUT_DIR),
    )

    try:
        result = run_graphify_command(
            [
                "extract",
                str(GRAPHIFY_ROOT),
                "--code-only",
                "--no-cluster",
                "--max-workers",
                "1",
                "--out",
                str(stage_root),
            ]
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        graph_path = staged_dir / "graph.json"
        if not graph_path.is_file():
            raise RuntimeError(f"Missing generated graph: {graph_path}")

        tree_result = run_graphify_command(
            [
                "tree",
                "--graph",
                str(graph_path),
                "--output",
                str(staged_dir / "GRAPH_TREE.html"),
                "--root",
                ".",
            ]
        )

        if tree_result.returncode != 0:
            raise RuntimeError(tree_result.stderr.strip())

        tree_path = staged_dir / "GRAPH_TREE.html"
        if not tree_path.is_file():
            raise RuntimeError(f"Missing tree output: {tree_path}")

        return stage_root, staged_dir

    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def publish_graph(stage_root: Path, staged_dir: Path) -> None:
    """Atomically publish a validated graph and remove stale output."""
    previous_dir = OUTPUT_ROOT / ".graphify-previous"

    try:
        if previous_dir.exists():
            shutil.rmtree(previous_dir)

        if OUTPUT_DIR.exists():
            os.replace(OUTPUT_DIR, previous_dir)

        os.replace(staged_dir, OUTPUT_DIR)

    except Exception:
        if not OUTPUT_DIR.exists() and previous_dir.exists():
            os.replace(previous_dir, OUTPUT_DIR)
        raise

    finally:
        shutil.rmtree(previous_dir, ignore_errors=True)
        shutil.rmtree(stage_root, ignore_errors=True)


def refresh_graph(log: structlog.stdlib.BoundLogger) -> bool:
    """Build, validate, and publish a refreshed graph."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        stage_root, staged_dir = build_staged_graph(log)
        publish_graph(stage_root, staged_dir)
    except Exception as exc:
        log.error(
            "graph_refresh_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return False

    log.info("graph_refresh_complete", output_dir=str(OUTPUT_DIR))
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
