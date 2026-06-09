"""Run AI2D vision evaluation on a pre-built VLM ONNX model.

Uses Olive's genai vision inference path to evaluate multimodal models
on the AI2D benchmark (AI2 Diagram Understanding).

Usage:
    # Eval a pre-built model
    python run_vision_eval.py --model-path /path/to/model --limit 100

    # Compare ONNX vs PyTorch
    python run_vision_eval.py --model-path /path/to/model --pytorch-model Qwen/Qwen3.5-0.8B --limit 100

    # GPU eval
    python run_vision_eval.py --model-path /path/to/model --device gpu --limit 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


def run_ai2d_eval(model_path: str, device: str, limit: int | None) -> float:
    """Run AI2D evaluation using Olive's vision evaluator and return accuracy."""
    from olive.data.config import DataConfig
    from olive.evaluator.metric import Metric, MetricType
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
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answer",
        "options_col": "options",
        "system_prompt": "Answer with only the option number (1, 2, 3, 4, etc.).",
    }
    if limit:
        pre_process_params["limit"] = limit

    eval_device = Device.GPU if device == "gpu" else Device.CPU
    evaluator = OnnxEvaluator()

    data_config = DataConfig(
        name="ai2d_eval",
        type="HuggingfaceContainer",
        load_dataset_config={
            "data_name": "lmms-lab/ai2d",
            "split": "test",
        },
        pre_process_data_config=pre_process_params,
        dataloader_config={
            "type": "vision_vqa_dataloader",
            "batch_size": 1,
        },
    )
    dc = data_config.to_data_container()
    dataloader = dc.create_dataloader()
    n_samples = len(dataloader)

    metric = Metric(
        name="ai2d_accuracy",
        type=MetricType.ACCURACY,
        sub_types=[{"name": "exact_match", "priority": 1}],
        data_config=data_config,
    )

    result = evaluator._evaluate_onnx_accuracy(model, metric, dataloader, device=eval_device)

    for key, sub_result in result.root.items():
        if "exact_match" in key:
            acc = sub_result.value
            correct = round(acc * n_samples)
            print(f"  Overall: {acc:.4f} ({correct}/{n_samples})")
            return acc

    return 0.0


def run_ai2d_pytorch_eval(model_id: str, device: str, limit: int | None) -> float:
    """Run AI2D PyTorch baseline evaluation."""
    import torch
    from datasets import load_dataset
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    system_prompt = "Answer with only the option number (1, 2, 3, 4, etc.)."
    total_correct = 0
    total_samples = 0

    torch_device = "cuda" if device == "gpu" and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch_device == "cuda" else torch.float32

    print(f"  Loading PyTorch model {model_id} on {torch_device} ({dtype})...")
    pt_model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(torch_device)
    pt_proc = AutoProcessor.from_pretrained(model_id)

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    ds = load_dataset("lmms-lab/ai2d", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    for sample in ds:
        pil_image = sample.get("image")
        if pil_image is None or not isinstance(pil_image, Image.Image):
            continue

        question = sample["question"]
        answer = sample["answer"]
        options = sample.get("options", [])

        if options:
            options_text = "\n".join(
                f"{j + 1}. {opt}" for j, opt in enumerate(options)
            )
            question = f"{question}\n{options_text}"

        # Convert 0-based answer index to 1-based
        try:
            answer_idx = int(answer)
            answer = str(answer_idx + 1)
        except (ValueError, IndexError):
            pass

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": question},
            ]},
        ]

        text = pt_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if process_vision_info is not None:
            image_patch_size = int(getattr(pt_proc.image_processor, "patch_size", 16))
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=image_patch_size)
            inputs = pt_proc(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(torch_device)
        else:
            inputs = pt_proc(
                text=[text], images=[pil_image],
                padding=True, return_tensors="pt",
            ).to(torch_device)

        with torch.no_grad():
            out = pt_model.generate(**inputs, max_new_tokens=64, do_sample=False)

        out_ids = out[0][inputs["input_ids"].shape[-1]:]
        pred = pt_proc.decode(out_ids, skip_special_tokens=True).strip()

        num_choices = len(sample.get('options', []))
        if 1 <= num_choices <= 9:
            num_match = re.search(rf"\b([1-{num_choices}])\b", pred)
            if num_match:
                pred = num_match.group(1)

        if pred == answer:
            total_correct += 1
        total_samples += 1

    del pt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"  Overall: {acc:.4f} ({total_correct}/{total_samples})")
    return acc


def main():
    parser = argparse.ArgumentParser(description="AI2D vision evaluation for VLM ONNX models")
    parser.add_argument("--model-path", required=True, help="Pre-built model directory")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--limit", type=int, default=50, help="Samples to evaluate (0=full)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min accuracy (fails if below)")
    parser.add_argument("--pytorch-model", default=None, help="HuggingFace model ID for PyTorch comparison")
    args = parser.parse_args()

    model_path = args.model_path

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
    print("AI2D Evaluation")
    print(f"  Model: {model_path}")
    print(f"  Device: {args.device}")
    print(f"  Limit: {limit or 'full'}")
    print(f"{'='*60}")

    start = time.time()
    acc = run_ai2d_eval(model_path, args.device, limit)
    elapsed = time.time() - start

    status = "PASS" if acc >= args.threshold else "FAIL"
    print(f"\n  {status}: AI2D ONNX exact_match = {acc:.4f} ({elapsed:.1f}s)")

    # PyTorch comparison
    if args.pytorch_model:
        print(f"\n{'='*60}")
        print("PyTorch Baseline")
        print(f"  Model: {args.pytorch_model}")
        print(f"  Device: {args.device}")
        print(f"  Limit: {limit or 'full'}")
        print(f"{'='*60}")

        pt_start = time.time()
        pt_acc = run_ai2d_pytorch_eval(args.pytorch_model, args.device, limit)
        pt_elapsed = time.time() - pt_start
        print(f"\n  PyTorch exact_match = {pt_acc:.4f} ({pt_elapsed:.1f}s)")

        diff = acc - pt_acc
        print(f"\n{'='*60}")
        print(f"  ONNX:    {acc:.4f}")
        print(f"  PyTorch: {pt_acc:.4f}")
        print(f"  Delta:   {diff:+.4f}")
        print(f"{'='*60}")

    if args.threshold > 0:
        print(f"  Threshold: {args.threshold:.4f}")

    if acc < args.threshold:
        sys.exit(1)


if __name__ == "__main__":
    main()
