"""Evaluate a Qwen3-30B-A3B Mobius CUDA package with lm-eval-harness.

Registers Olive's `LMEvalORTGenAIEvaluator` (the lm-eval `ortgenai` model class)
and scores the exported ORT GenAI package on the CUDA execution provider.

Run from the `Qwen-Qwen3-30B-A3B/` recipe root:

    python eval.py                                   # MMLU, 100 samples per subtask
    python eval.py --limit 0                         # full task, no sample cap
    python eval.py --task arc_challenge --limit 500
    python eval.py --model-path cuda/kquant_fp16/models --max-length 8192
"""

import argparse
import sys
import time
from pathlib import Path

DEFAULT_MODEL_PATH = "cuda/kquant_fp16/models"
DEFAULT_TASK = "mmlu"
DEFAULT_LIMIT = 100
DEFAULT_MAX_LENGTH = 4096

# This recipe only produces a CUDA package.
EXECUTION_PROVIDER = "cuda"


def validate_model_path(model_path: str) -> Path:
    """Return the ORT GenAI model directory, exiting with a clear message if it is unusable."""
    path = Path(model_path)
    if not path.is_dir():
        sys.exit(
            f"ERROR: model directory not found: {path}\n"
            f"Run `olive run --config {path.parent / 'config.json'}` from the recipe root first."
        )
    if not (path / "genai_config.json").is_file():
        sys.exit(f"ERROR: {path} is not an ORT GenAI package: genai_config.json is missing.")
    return path


def nonnegative_int(value: str) -> int:
    """Parse an integer that is zero or greater."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_int(value: str) -> int:
    """Parse an integer greater than zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def load_lm_eval():
    """Import lm-eval lazily so `--help` works without the evaluation extras installed."""
    try:
        # Importing the Olive evaluator registers the "ortgenai" model class with lm-eval.
        import olive.evaluator.lmeval_ort  # noqa: F401
        from lm_eval import simple_evaluate
        from lm_eval.api.registry import get_model
        from lm_eval.tasks import TaskManager
    except ImportError as error:
        sys.exit(
            f"ERROR: evaluation dependencies are missing: {error}\n"
            "Install them with `pip install -r cuda/requirements.txt` from the recipe root."
        )
    return get_model, simple_evaluate, TaskManager


def print_results(results: dict, task: str) -> None:
    """Print every metric lm-eval reported, without assuming a particular metric name."""
    reported = results.get("results") or {}
    if not reported:
        print("  lm-eval returned no results.")
        return

    # Task groups (e.g. `mmlu`) also report their subtasks; show the requested entry when present.
    entries = {task: reported[task]} if task in reported else reported
    for name, metrics in entries.items():
        print(f"  {name}:")
        for key, value in sorted(metrics.items()):
            if key == "alias":
                continue
            print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Qwen3-30B-A3B ONNX package with lm-eval-harness")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"ORT GenAI model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help=f"lm-eval task name (default: {DEFAULT_TASK})")
    parser.add_argument(
        "--limit",
        type=nonnegative_int,
        default=DEFAULT_LIMIT,
        help=f"Samples per task (default: {DEFAULT_LIMIT}, use 0 for the full task)",
    )
    parser.add_argument(
        "--max-length",
        type=positive_int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--num-fewshot",
        type=nonnegative_int,
        default=None,
        help="Few-shot examples per sample (default: the task's own setting)",
    )
    args = parser.parse_args()

    model_path = validate_model_path(args.model_path)
    get_model, simple_evaluate, task_manager_cls = load_lm_eval()
    limit = args.limit if args.limit > 0 else None

    print(f"Model path : {model_path}")
    print(f"EP         : {EXECUTION_PROVIDER}")
    print(f"Task       : {args.task}")
    print(f"Limit      : {limit or 'full'}")
    print(f"Max length : {args.max_length}\n")

    # `past_present_share_buffer` is left unset so the evaluator mirrors the exported genai_config.json.
    print("Loading model ...")
    start = time.perf_counter()
    model = get_model("ortgenai")(
        pretrained=str(model_path),
        batch_size=1,
        max_length=args.max_length,
        ep=EXECUTION_PROVIDER,
        device="cuda",
    )
    print(f"Model loaded in {time.perf_counter() - start:.1f} s.\n")

    print(f"Running {args.task} ...")
    start = time.perf_counter()
    results = simple_evaluate(
        model=model,
        tasks=[args.task],
        task_manager=task_manager_cls(),
        log_samples=False,
        batch_size=1,
        limit=limit,
        num_fewshot=args.num_fewshot,
    )
    print(f"\nCompleted in {time.perf_counter() - start:.1f} s.\n")

    print_results(results, args.task)

    print("\nNote: lm-eval scores the packaged model with loglikelihood ranking over the task's answer")
    print("choices, so results depend on the task, the sample limit, and the few-shot setting used.")


if __name__ == "__main__":
    main()
