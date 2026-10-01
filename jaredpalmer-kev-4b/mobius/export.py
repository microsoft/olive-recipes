#!/usr/bin/env python3
"""Export KEV-4B with Mobius."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.non_generative_mobius import run_recipe  # noqa: E402


def main() -> None:
    """Run the KEV Mobius recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument(
        "--precision",
        choices=("fp32", "mixed", "both"),
        default="mixed",
    )
    args = parser.parse_args()
    run_recipe("kev", args.artifact, args.output_dir, args.precision)


if __name__ == "__main__":
    main()
