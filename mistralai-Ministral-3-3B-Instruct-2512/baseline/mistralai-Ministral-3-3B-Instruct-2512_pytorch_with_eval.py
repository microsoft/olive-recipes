"""PyTorch baseline evaluation for Ministral-3-3B on AI2D."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


DEFAULT_PYTORCH_MODEL = "mistralai/Ministral-3-3B-Instruct-2512-BF16"


def _has_arg(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def _load_builtin_eval():
    eval_path = Path(__file__).resolve().parents[1] / "builtin" / "eval.py"
    spec = importlib.util.spec_from_file_location("ministral_builtin_eval", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evaluator from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    argv = sys.argv[1:]
    if not _has_arg(argv, "--skip_onnx"):
        argv.insert(0, "--skip_onnx")
    if not _has_arg(argv, "--pytorch_model"):
        argv.extend(["--pytorch_model", DEFAULT_PYTORCH_MODEL])

    sys.argv = [sys.argv[0], *argv]
    _load_builtin_eval().main()


if __name__ == "__main__":
    main()
