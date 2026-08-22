"""Validate the final corpus-backed evaluation benchmark."""

import argparse
from pathlib import Path

from inference_lab.evaluation.coverage import validate_final_benchmark
from inference_lab.evaluation.loader import load_dataset


def main() -> None:
    """Load and validate a final evaluation benchmark manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()

    dataset = load_dataset(arguments.manifest)
    report = validate_final_benchmark(dataset.cases)

    print(
        f"validated {report.total_cases} cases: "
        f"{report.answerable_cases} answerable, "
        f"{report.abstention_cases} abstention, "
        f"{report.held_out_cases} held-out"
    )


if __name__ == "__main__":
    main()
