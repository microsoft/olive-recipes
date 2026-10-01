#!/usr/bin/env python3
"""Export the mixed CLM-v0.1-8B CUDA package with Mobius."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.non_generative_mobius import run_recipe  # noqa: E402


def main() -> None:
    """Run the CLM Mobius recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Retain the FP32 package in addition to the mixed CUDA package",
    )
    args = parser.parse_args()
    run_recipe(
        "clm",
        args.artifact,
        args.output_dir,
        "both" if args.keep_fp32 else "mixed",
    )


if __name__ == "__main__":
    main()
