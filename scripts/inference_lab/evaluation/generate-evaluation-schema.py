"""Generate the published evaluation JSON schema artifact."""

import argparse
from pathlib import Path

from inference_lab.evaluation import write_json_schema


def main() -> None:
    """Generate an evaluation schema at the requested destination."""
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    write_json_schema(arguments.destination)


if __name__ == "__main__":
    main()
