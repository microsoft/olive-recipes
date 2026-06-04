"""Run vision evaluation on a pre-built VLM ONNX model.

This script is designed to be called from olive_ci.json commands.
It builds the model via olive run, then evaluates it on vision benchmarks
using Olive's genai vision inference path.

Usage:
    # Build + eval (standard CI flow)
    python run_vision_eval.py --config cpu/int4/config.json --benchmarks textvqa --limit 100

    # Eval only (pre-built model)
    python run_vision_eval.py --model-path /path/to/model --benchmarks textvqa,chartqa --limit 100

    # All benchmarks
    python run_vision_eval.py --model-path /path/to/model --benchmarks textvqa,chartqa,docvqa --limit 100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Benchmark definitions — public HuggingFace datasets
BENCHMARKS = {
    "textvqa": {
        "display_name": "TextVQA (exact_match)",
        "dataset_name": "facebook/textvqa",
        "split": "validation",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answers",
        "sub_type": "exact_match",
    },
    "chartqa": {
        "display_name": "ChartQA (relaxed_accuracy)",
        "dataset_name": "HuggingFaceM4/ChartQA",
        "split": "test",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answer",
        "sub_type": "relaxed_accuracy",
    },
    "docvqa": {
        "display_name": "DocumentVQA (word_sort_ratio)",
        "dataset_name": "HuggingFaceM4/DocumentVQA",
        "split": "validation",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answers",
        "sub_type": "word_sort_ratio",
    },
}


def build_model(config_path: str) -> str:
    """Build ONNX model via olive run in a subprocess and return model directory path."""
    print(f"Building model from {config_path}...")
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "olive", "run", "--config", config_path],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: Model build failed (exit code {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    config_data = json.loads(Path(config_path).read_text())
    output_dir = Path(config_data.get("output_dir", "models/output"))

    for p in output_dir.rglob("genai_config.json"):
        model_dir = str(p.parent)
        print(f"Model built in {elapsed:.1f}s: {model_dir}")
        return model_dir

    print(f"ERROR: No genai_config.json found in {output_dir}", file=sys.stderr)
    sys.exit(1)


def run_vision_eval(model_path: str, benchmark: dict, device: str, limit: int | None) -> float:
    """Run a single vision benchmark using Olive's evaluator and return accuracy score."""
    from olive.data.config import DataConfig
    from olive.evaluator.metric import Metric, MetricType, SubMetric
    from olive.evaluator.olive_evaluator import OnnxEvaluator
    from olive.hardware import Device
    from olive.model import ONNXModelHandler

    model_dir = Path(model_path)
    text_onnx = model_dir / "text.onnx"
    if not text_onnx.exists():
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx files in {model_dir}")
        text_onnx = onnx_files[0]

    model = ONNXModelHandler(model_path=str(text_onnx))

    pre_process_params = {
        "type": "vision_vqa_pre_process",
        "image_col": benchmark["image_col"],
        "question_col": benchmark["question_col"],
        "answer_col": benchmark["answer_col"],
    }
    if limit:
        pre_process_params["limit"] = limit

    data_config = DataConfig(
        name="vision_eval_data",
        type="HuggingfaceContainer",
        load_dataset_config={"data_name": benchmark["dataset_name"], "split": benchmark["split"]},
        pre_process_data_config=pre_process_params,
        dataloader_config={"batch_size": 1},
    )
    dc = data_config.to_data_container()
    dataloader = dc.create_dataloader()

    metric = Metric(
        name="vision_accuracy",
        type=MetricType.ACCURACY,
        sub_types=[SubMetric(name=benchmark["sub_type"], priority=1)],
        data_config=data_config,
    )

    eval_device = Device.GPU if device == "gpu" else Device.CPU
    evaluator = OnnxEvaluator()
    result = evaluator._evaluate_onnx_accuracy(model, metric, dataloader, device=eval_device)

    for sub_result in result.value:
        if sub_result.name == benchmark["sub_type"]:
            return sub_result.value

    raise ValueError(f"No result for {benchmark['sub_type']}")


def main():
    parser = argparse.ArgumentParser(description="Vision evaluation for VLM ONNX models")
    parser.add_argument("--config", default=None, help="Olive config to build model (skipped if --model-path set)")
    parser.add_argument("--model-path", default=None, help="Pre-built model directory")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--benchmarks", default="textvqa", help="Comma-separated: textvqa,chartqa,docvqa")
    parser.add_argument("--limit", type=int, default=50, help="Samples per benchmark (0=full)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min accuracy (fails if below)")
    args = parser.parse_args()

    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    elif args.config:
        model_path = build_model(args.config)
    else:
        print("ERROR: Provide --config or --model-path", file=sys.stderr)
        sys.exit(1)

    # Verify it's a vision model
    genai_config = Path(model_path) / "genai_config.json"
    if not genai_config.exists():
        print(f"ERROR: genai_config.json not found in {model_path}", file=sys.stderr)
        sys.exit(1)

    cfg = json.loads(genai_config.read_text())
    if "vision" not in cfg.get("model", {}):
        print("WARNING: Model may not be a VLM (no 'vision' field in genai_config.json)")

    # Run benchmarks
    benchmark_names = [b.strip() for b in args.benchmarks.split(",")]
    limit = args.limit if args.limit > 0 else None
    results = {}
    failed = False

    for name in benchmark_names:
        if name not in BENCHMARKS:
            print(f"ERROR: Unknown benchmark '{name}'. Available: {', '.join(BENCHMARKS)}", file=sys.stderr)
            sys.exit(1)

        benchmark = BENCHMARKS[name]
        print(f"\n{'='*60}")
        print(f"Running {benchmark['display_name']}")
        print(f"  Dataset: {benchmark['dataset_name']} (split={benchmark['split']})")
        print(f"  Limit: {limit or 'full'}")
        print(f"{'='*60}")

        start = time.time()
        acc = run_vision_eval(model_path, benchmark, args.device, limit)
        elapsed = time.time() - start

        results[name] = {"acc": acc, "time": elapsed}
        status = "PASS" if acc >= args.threshold else "FAIL"
        if acc < args.threshold:
            failed = True
        print(f"\n  {status}: {acc:.4f} ({elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for name in benchmark_names:
        acc = results[name]["acc"]
        elapsed = results[name]["time"]
        print(f"  {BENCHMARKS[name]['display_name']:<35} {acc:.4f}  ({elapsed:.1f}s)")

    if failed:
        print(f"\nFAILED: One or more benchmarks below threshold {args.threshold}")
        sys.exit(1)
    else:
        print("\nAll benchmarks passed.")


if __name__ == "__main__":
    main()
