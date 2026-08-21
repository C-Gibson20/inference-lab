"""CLI wrapper for the local code-graph generator."""

import sys

from inference_lab.tooling.graph import main


if __name__ == "__main__":
    sys.exit(main())
