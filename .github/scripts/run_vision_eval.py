"""Run MMMU vision evaluation on a pre-built VLM ONNX model.

Uses Olive's genai vision inference path to evaluate multimodal models
on the MMMU benchmark (Massive Multi-discipline Multimodal Understanding).

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


ALL_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]


def _resolve_subjects(subject: str) -> list[str]:
    """Resolve subject argument to a list of MMMU subjects."""
    if subject.lower() == "all":
        return ALL_SUBJECTS
    return [s.strip() for s in subject.split(",")]


def run_mmmu_eval(model_path: str, device: str, limit: int | None, subject: str = "all") -> float:
    """Run MMMU evaluation using Olive's vision evaluator and return accuracy."""
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

    subjects = _resolve_subjects(subject)

    pre_process_params = {
        "type": "vision_vqa_pre_process",
        "image_col": "image_1",
        "question_col": "question",
        "answer_col": "answer",
        "options_col": "options",
        "system_prompt": "Answer with only the option letter (A, B, C, D, etc.).",
    }

    # Distribute limit evenly across subjects
    per_subject_limit = max(1, limit // len(subjects)) if limit else None
    if per_subject_limit:
        pre_process_params["limit"] = per_subject_limit

    eval_device = Device.GPU if device == "gpu" else Device.CPU
    evaluator = OnnxEvaluator()
    total_correct = 0
    total_samples = 0
    subject_accs = []

    for subj in subjects:
        data_config = DataConfig(
            name=f"mmmu_{subj}",
            type="HuggingfaceContainer",
            load_dataset_config={
                "data_name": "MMMU/MMMU",
                "subset": subj,
                "split": "validation",
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
            name="mmmu_accuracy",
            type=MetricType.ACCURACY,
            sub_types=[{"name": "exact_match", "priority": 1}],
            data_config=data_config,
        )

        result = evaluator._evaluate_onnx_accuracy(model, metric, dataloader, device=eval_device)

        for key, sub_result in result.root.items():
            if "exact_match" in key:
                subj_acc = sub_result.value
                subj_correct = round(subj_acc * n_samples)
                total_correct += subj_correct
                total_samples += n_samples
                subject_accs.append(subj_acc)
                print(f"    {subj}: {subj_acc:.4f} ({subj_correct}/{n_samples})")
                break

    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    if len(subject_accs) > 1:
        avg_acc = sum(subject_accs) / len(subject_accs)
        print(f"  ----")
        print(f"  Overall: {overall_acc:.4f} ({total_correct}/{total_samples})")
        print(f"  Macro-avg across {len(subject_accs)} subjects: {avg_acc:.4f}")
    return overall_acc


def run_mmmu_pytorch_eval(
    model_id: str, device: str, limit: int | None, subject: str = "all"
) -> float:
    """Run MMMU evaluation using a HuggingFace PyTorch model and return accuracy."""
    import re

    import torch
    from datasets import load_dataset
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_device = "cuda" if device == "gpu" and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch_device == "cuda" else torch.float32

    print(f"  Loading PyTorch model {model_id} on {torch_device} ({dtype})...")
    pt_model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(torch_device)
    pt_proc = AutoProcessor.from_pretrained(model_id)

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    subjects = _resolve_subjects(subject)
    per_subject_limit = max(1, limit // len(subjects)) if limit else None

    system_prompt = "Answer with only the option letter (A, B, C, D, etc.)."
    total_correct = 0
    total_samples = 0
    subject_accs = []

    for subj in subjects:
        ds = load_dataset("MMMU/MMMU", subj, split="validation")
        if per_subject_limit:
            ds = ds.select(range(min(per_subject_limit, len(ds))))

        subj_correct = 0
        subj_total = 0

        for sample in ds:
            pil_image = sample.get("image_1")
            if pil_image is None or not isinstance(pil_image, Image.Image):
                continue

            question = sample["question"]
            answer = sample["answer"]
            options = sample.get("options", [])

            if options:
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                options_text = "\n".join(f"{letters[j]}. {opt}" for j, opt in enumerate(options))
                question = f"{question}\n{options_text}"

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

            letter_match = re.search(r"\b([A-Z])\b", pred)
            if letter_match:
                pred = letter_match.group(1)

            if pred == answer:
                subj_correct += 1
            subj_total += 1

        subj_acc = subj_correct / subj_total if subj_total > 0 else 0.0
        subject_accs.append(subj_acc)
        print(f"    {subj}: {subj_acc:.4f} ({subj_correct}/{subj_total})")
        total_correct += subj_correct
        total_samples += subj_total

    del pt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    if len(subject_accs) > 1:
        avg_acc = sum(subject_accs) / len(subject_accs)
        print(f"  ----")
        print(f"  Overall: {overall_acc:.4f} ({total_correct}/{total_samples})")
        print(f"  Macro-avg across {len(subject_accs)} subjects: {avg_acc:.4f}")
    return overall_acc


def main():
    parser = argparse.ArgumentParser(description="MMMU vision evaluation for VLM ONNX models")
    parser.add_argument("--config", default=None, help="Olive config to build model (skipped if --model-path set)")
    parser.add_argument("--model-path", default=None, help="Pre-built model directory")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--limit", type=int, default=50, help="Samples to evaluate (0=full)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min accuracy (fails if below)")
    parser.add_argument("--subject", default="all", help="MMMU subject (default: all). Use comma-separated for multiple.")
    parser.add_argument("--pytorch-model", default=None, help="HuggingFace model ID for PyTorch comparison")
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
    print(f"\n  {status}: MMMU ({args.subject}) ONNX exact_match = {acc:.4f} ({elapsed:.1f}s)")

    # PyTorch comparison
    pt_acc = None
    if args.pytorch_model:
        print(f"\n{'='*60}")
        print(f"PyTorch Baseline ({args.subject})")
        print(f"  Model: {args.pytorch_model}")
        print(f"  Device: {args.device}")
        print(f"  Limit: {limit or 'full'}")
        print(f"{'='*60}")

        pt_start = time.time()
        pt_acc = run_mmmu_pytorch_eval(args.pytorch_model, args.device, limit, args.subject)
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
