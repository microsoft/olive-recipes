#!/usr/bin/env python3
"""
Ruff-based lint and format check for Python files in `.aitk/` and every `*/aitk/` folder.

By default, runs `ruff check` and `ruff format --check` (verification only) and exits 1 on
issues. When called with `fix=True`, runs `ruff check --fix` and `ruff format` to rewrite
files in place.
"""

import subprocess
import sys
from pathlib import Path

from sanitize.utils import printError, printInfo, printTip

LINE_LENGTH = "120"
SELECT_RULES = "F401,F841,I"
REQUIREMENTS_HINT = ".aitk/scripts/requirements.txt"


def _ensure_ruff():
    try:
        subprocess.run(["ruff", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        printError(f"ruff is not installed. Install dependencies first: pip install -r {REQUIREMENTS_HINT}")
        return False


def _collect_targets():
    repo_root = Path(__file__).parent.parent.parent

    targets = []
    dot_aitk = repo_root / ".aitk"
    if dot_aitk.is_dir():
        targets.append(dot_aitk)

    for aitk_dir in sorted(repo_root.glob("*/aitk")):
        if aitk_dir.is_dir():
            targets.append(aitk_dir)

    return targets


def ruff_check(fix: bool = False):
    """
    Run ruff lint + format across `.aitk/` and every `*/aitk/` folder.

    Args:
        fix: When True, apply autofixes and reformat files. When False, only verify.
    """
    if not _ensure_ruff():
        raise SystemExit(1)

    targets = _collect_targets()
    if not targets:
        printInfo("No .aitk / aitk folders found for ruff to check.")
        return

    target_args = [str(t) for t in targets]
    printTip(f"Running ruff on {len(targets)} folder(s)...")

    exclude_ipynb = ["--config", 'extend-exclude=["*.ipynb"]']
    check_cmd = ["ruff", "check", "--select", SELECT_RULES, "--line-length", LINE_LENGTH, *exclude_ipynb]
    format_cmd = ["ruff", "format", "--line-length", LINE_LENGTH, *exclude_ipynb]
    if fix:
        check_cmd.append("--fix")
    else:
        format_cmd.append("--check")

    check_result = subprocess.run(check_cmd + target_args)
    format_result = subprocess.run(format_cmd + target_args)

    if check_result.returncode != 0 or format_result.returncode != 0:
        printError("Ruff reported issues. Re-run `python .aitk/scripts/ruff_check.py --fix` to auto-fix.")
        raise SystemExit(1)

    printInfo("Ruff check passed.")


if __name__ == "__main__":
    ruff_check(fix="--fix" in sys.argv)
