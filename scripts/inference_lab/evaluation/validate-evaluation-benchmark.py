"""Validate the final corpus-backed evaluation benchmark."""

import argparse
from pathlib import Path

from inference_lab.evaluation import load_dataset, validate_final_benchmark


def main() -> int:
    """Load and validate a final evaluation benchmark manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()

    try:
        dataset = load_dataset(arguments.manifest)
        report = validate_final_benchmark(dataset.cases)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"benchmark validation failed: {error}\n")

    print(
        f"validated {report.total_cases} cases: "
        f"{report.answerable_cases} answerable, "
        f"{report.abstention_cases} abstention, "
        f"{report.held_out_cases} held-out "
        f"({report.held_out_answerable_cases} answerable, "
        f"{report.held_out_abstention_cases} abstention)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
