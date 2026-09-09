"""Evaluate Gemma 4 ONNX model accuracy using lm-eval-harness.

Uses Olive's LMEvalORTGenAIEvaluator to run benchmarks against the
ORT GenAI model package.

Usage:
    python eval.py                                    # MMLU Pro, CPU, 100 samples
    python eval.py --device gpu                       # MMLU Pro, CUDA
    python eval.py --device gpu --variant int4        # INT4 quantized model
    python eval.py --task mmlu_pro --limit 500        # More samples
    python eval.py --task gpqa_diamond_n_shot          # GPQA Diamond (requires dataset access)
"""

import argparse
import json
import time

# Register Olive's ORT GenAI evaluator with lm-eval
import olive.evaluator.lmeval_ort  # noqa: F401

from lm_eval import simple_evaluate
from lm_eval.api.registry import get_model
from lm_eval.tasks import TaskManager


def resolve_model_path(device: str, variant: str | None) -> str:
    """Resolve the model directory from device + variant args."""
    if device == "cpu":
        variant = variant or "fp32"
        return f"cpu/{variant}/models"
    variant = variant or "int4"
    return f"cuda/{variant}/models"


# Published reference scores for google/gemma-4-E2B-it
REFERENCE_SCORES = {
    "leaderboard_mmlu_pro": 0.600,  # 60.0% (Google reported, CoT format)
    "gpqa_diamond_n_shot": 0.434,   # 43.4% (Google reported)
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma 4 ONNX models")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Target device",
    )
    parser.add_argument(
        "--variant",
        choices=["fp32", "fp16", "int4", "mixed"],
        default=None,
        help="Model variant (cpu defaults to fp32, gpu defaults to int4)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model directory path",
    )
    parser.add_argument(
        "--task",
        default="leaderboard_mmlu_pro",
        help="lm-eval task name (default: leaderboard_mmlu_pro)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100, use 0 for full)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    args = parser.parse_args()

    model_path = args.model_path or resolve_model_path(args.device, args.variant)
    ep = "cuda" if args.device == "gpu" else "cpu"
    limit = args.limit if args.limit > 0 else None

    print(f"Model path: {model_path}")
    print(f"EP: {ep}")
    print(f"Task: {args.task}")
    print(f"Limit: {limit or 'full'}")
    print()

    # Load model
    # Gemma4 requires past_present_share_buffer=False for correct KV cache handling
    print("Loading model...")
    start = time.time()
    model = get_model("ortgenai")(
        pretrained=model_path,
        batch_size=1,
        max_length=args.max_length,
        ep=ep,
        past_present_share_buffer=False,
    )
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Run evaluation
    print(f"Running {args.task}...")
    eval_start = time.time()
    results = simple_evaluate(
        model=model,
        tasks=[args.task],
        task_manager=TaskManager(),
        log_samples=False,
        batch_size=1,
        limit=limit,
    )
    eval_time = time.time() - eval_start

    # Print results
    print(f"\nCompleted in {eval_time:.1f}s")
    print()

    task_results = results["results"].get(args.task, {})
    acc = task_results.get("acc,none")
    acc_stderr = task_results.get("acc_stderr,none")

    if acc is not None:
        print(f"  {args.task}: {acc*100:.1f}% (±{acc_stderr*100:.1f}%)")
        ref = REFERENCE_SCORES.get(args.task)
        if ref is not None:
            print(f"  Published reference: {ref*100:.1f}%")
            print(f"  Delta: {(acc - ref)*100:+.1f} pp")
    else:
        print(f"  {args.task}: {json.dumps(task_results, indent=2)}")

    # Note about evaluation methodology
    print()
    print("Note: lm-eval uses loglikelihood scoring (multiple-choice).")
    print("Google's published scores may use CoT generation, which typically")
    print("produces higher scores on reasoning benchmarks like MMLU Pro.")


if __name__ == "__main__":
    main()
