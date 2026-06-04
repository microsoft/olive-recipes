"""Run MMMU vision evaluation on a pre-built VLM ONNX model.

Uses Olive's genai vision inference path to evaluate multimodal models
on the MMMU benchmark (Massive Multi-discipline Multimodal Understanding).

Usage:
    # Eval a pre-built model
    python run_vision_eval.py --model-path /path/to/model --limit 100

    # Build from olive config + eval
    python run_vision_eval.py --config cpu/int4/config.json --limit 100

    # GPU eval
    python run_vision_eval.py --model-path /path/to/model --device gpu --limit 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


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


def run_mmmu_eval(model_path: str, device: str, limit: int | None, subject: str = "Accounting") -> float:
    """Run MMMU evaluation using Olive's vision evaluator and return accuracy."""
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
        "image_col": "image_1",
        "question_col": "question",
        "answer_col": "answer",
    }
    if limit:
        pre_process_params["limit"] = limit

    data_config = DataConfig(
        name="mmmu_eval_data",
        type="HuggingfaceContainer",
        load_dataset_config={
            "data_name": "MMMU/MMMU",
            "subset": subject,
            "split": "validation",
            "trust_remote_code": True,
        },
        pre_process_data_config=pre_process_params,
        dataloader_config={"batch_size": 1},
    )
    dc = data_config.to_data_container()
    dataloader = dc.create_dataloader()

    metric = Metric(
        name="mmmu_accuracy",
        type=MetricType.ACCURACY,
        sub_types=[SubMetric(name="exact_match", priority=1)],
        data_config=data_config,
    )

    eval_device = Device.GPU if device == "gpu" else Device.CPU
    evaluator = OnnxEvaluator()
    result = evaluator._evaluate_onnx_accuracy(model, metric, dataloader, device=eval_device)

    for sub_result in result.value:
        if sub_result.name == "exact_match":
            return sub_result.value

    raise ValueError("No exact_match result found")


def main():
    parser = argparse.ArgumentParser(description="MMMU vision evaluation for VLM ONNX models")
    parser.add_argument("--config", default=None, help="Olive config to build model (skipped if --model-path set)")
    parser.add_argument("--model-path", default=None, help="Pre-built model directory")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--limit", type=int, default=50, help="Samples to evaluate (0=full)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min accuracy (fails if below)")
    parser.add_argument("--subject", default="Accounting", help="MMMU subject subset (default: Accounting)")
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

    limit = args.limit if args.limit > 0 else None

    print(f"\n{'='*60}")
    print(f"MMMU Evaluation ({args.subject})")
    print(f"  Model: {model_path}")
    print(f"  Device: {args.device}")
    print(f"  Limit: {limit or 'full'}")
    print(f"{'='*60}")

    start = time.time()
    acc = run_mmmu_eval(model_path, args.device, limit, args.subject)
    elapsed = time.time() - start

    status = "PASS" if acc >= args.threshold else "FAIL"
    print(f"\n  {status}: MMMU ({args.subject}) exact_match = {acc:.4f} ({elapsed:.1f}s)")
    if args.threshold > 0:
        print(f"  Threshold: {args.threshold:.4f}")

    if acc < args.threshold:
        sys.exit(1)


if __name__ == "__main__":
    main()
