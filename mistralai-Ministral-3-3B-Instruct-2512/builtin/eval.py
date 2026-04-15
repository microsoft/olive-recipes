"""Evaluate Ministral-3-3B VLM (ONNX) vs PyTorch on AI2D (diagram understanding).

AI2D is a multiple-choice visual QA benchmark on scientific diagrams.
Each sample has an image, a question, four answer options, and a ground-truth answer.
Accuracy is the fraction of questions answered with the correct option letter.

Expected precision gaps (ONNX vs PyTorch reference):
    CPU + FP32   → expect ~0 pp gap  (exact parity)
    CUDA + FP16  → expect <2 pp gap  (FP16 precision loss)
    CPU + INT4   → expect <5 pp gap  (quantization loss)

Usage:
    # CPU INT4 model (default)
    python eval.py --device cpu --model_path cpu_and_mobile/models

    # CUDA FP16 model
    python eval.py --device cuda --model_path cuda/models

    # Compare ONNX vs PyTorch reference
    python eval.py --pytorch_model mistralai/Ministral-3-3B-Instruct-2512

    # Larger sample
    python eval.py --num_samples 200
"""

import argparse
import io
import json
import os
import re
import tempfile
import time

import onnxruntime_genai as og
from datasets import load_dataset
from PIL import Image

NUMBERS = ["1", "2", "3", "4"]

# Expected accuracy gap thresholds (percentage points) by precision.
# These help users quickly assess whether a model export is healthy.
EXPECTED_GAP_PP = {
    "fp32": 0.0,
    "fp16": 2.0,
    "int4": 5.0,
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise multiple-choice answering assistant. "
    "When given a question with numbered options, respond with ONLY a single digit (1, 2, 3, or 4). "
    "Do not include any explanation, reasoning, or other text — just the digit."
)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def build_messages(question: str, options: list[str], system_prompt: str = "") -> str:
    """Return a JSON-encoded chat messages list (for apply_chat_template).

    Uses string content with [IMG] prefix instead of structured content
    because ORT GenAI's Jinja does not support the sort() filter needed
    by Mistral3's structured-content template path.
    """
    option_text = "\n".join(f"{N}. {o}" for N, o in zip(NUMBERS, options))
    content = (
        f"[IMG]Look at the diagram and answer the multiple-choice question.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_text}\n\n"
        f"Reply with the number only (1, 2, 3, or 4)."
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return json.dumps(messages)


def parse_answer(text: str) -> str | None:
    """Extract the first 1/2/3/4 digit from a model response."""
    text = text.strip()
    m = re.search(r"\b([1-4])\b", text)
    if m:
        return m.group(1)
    for ch in text:
        if ch in NUMBERS:
            return ch
    return None


def ground_truth_number(sample: dict) -> str | None:
    """Normalise the dataset's answer field to a 1-based number string.

    AI2D stores answer as a 0-based integer index into the options list.
    We map: index 0 → '1', 1 → '2', 2 → '3', 3 → '4'.
    """
    answer = sample.get("answer", "")
    try:
        idx = int(answer)
        if 0 <= idx < 4:
            return NUMBERS[idx]
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Precision detection
# ---------------------------------------------------------------------------


def detect_onnx_precision(model_path: str) -> str:
    """Infer ONNX model precision from the model directory or genai_config.

    Heuristics (in order):
    1. If genai_config.json exists and contains model builder metadata → use it
    2. If path contains 'int4' → 'int4'
    3. If path contains 'cpu_and_mobile' → 'int4' (default for CPU target)
    4. If path contains 'cuda' or 'fp16' → 'fp16'
    5. Fallback → 'fp16'
    """
    path_lower = model_path.lower()

    # Check genai_config for precision hints
    config_path = os.path.join(model_path, "genai_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            # ModelBuilder writes precision into the config
            decoder_cfg = config.get("model", {}).get("decoder", {})
            if "int4" in json.dumps(decoder_cfg).lower():
                return "int4"
        except (json.JSONDecodeError, OSError):
            pass

    if "int4" in path_lower:
        return "int4"
    if "cpu_and_mobile" in path_lower:
        return "int4"
    if "fp16" in path_lower or "cuda" in path_lower:
        return "fp16"
    return "fp16"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def pil_from_sample(sample: dict) -> Image.Image | None:
    """Return PIL image from a dataset sample regardless of field format."""
    img = sample.get("image")
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    return None


def load_ai2d(num_samples: int):
    """Load a deterministic subset of AI2D test samples."""
    print(f"Loading AI2D dataset ({num_samples} samples)…")
    ds = load_dataset("lmms-lab/ai2d", split="test")
    ds = ds.select(range(min(num_samples, len(ds))))
    print(f"  Loaded {len(ds)} samples.")
    return ds


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def build_onnx_runner(model_path: str):
    """Load ONNX model with ORT GenAI."""
    print(f"\nLoading ONNX model from: {model_path}")
    model = og.Model(model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    print("  ONNX model loaded.")
    return model, processor, tokenizer


def run_onnx(
    model, processor, tokenizer, pil_image: Image.Image, messages_json: str
) -> str:
    """Run a single inference with the ONNX GenAI model."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        pil_image.save(f, format="PNG")
        tmp_path = f.name

    try:
        images = og.Images.open(tmp_path)
        prompt = tokenizer.apply_chat_template(
            messages_json, add_generation_prompt=True
        )
        inputs = processor(prompt, images=images)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=2000, do_sample=False)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            tokens.append(generator.get_next_tokens()[0])
        del generator

        return tokenizer.decode(tokens)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# PyTorch inference
# ---------------------------------------------------------------------------


def build_pytorch_runner(model_id: str, device: str = "auto"):
    """Load HuggingFace PyTorch model for comparison.

    Args:
        model_id: HuggingFace model ID or local path.
        device: 'cpu', 'cuda', or 'auto' (auto-detect).
    """
    print(f"\nLoading PyTorch model: {model_id}")
    import torch
    from transformers import AutoProcessor, Mistral3ForConditionalGeneration

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    precision_label = "fp16" if device == "cuda" else "fp32"
    print(f"  Device: {device}, dtype: {dtype} ({precision_label})")

    pt_model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True
    ).to(device)
    pt_proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("  PyTorch model loaded.")
    return pt_model, pt_proc, device, precision_label


def run_pytorch(
    pt_model,
    pt_proc,
    pil_image: Image.Image,
    question: str,
    options: list[str],
    device: str,
    system_prompt: str = "",
) -> str:
    """Run a single inference with the HuggingFace PyTorch model."""
    import torch

    option_text = "\n".join(f"{N}. {o}" for N, o in zip(NUMBERS, options))
    content = (
        f"Look at the diagram and answer the multiple-choice question.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_text}\n\n"
        f"Reply with the number only (1, 2, 3, or 4)."
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": content},
            ],
        }
    )
    text = pt_proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = pt_proc(
        text=[text], images=[pil_image], padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = pt_model.generate(**inputs, max_new_tokens=8, do_sample=False)

    out_ids = out[0][inputs["input_ids"].shape[-1] :]
    return pt_proc.decode(out_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(dataset, runner_fn, label: str) -> dict:
    """Run evaluation on a dataset with the given runner function."""
    correct = 0
    skipped = 0
    total = len(dataset)
    latencies = []

    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {label}  ({total} samples)")
    print(f"{'=' * 60}")

    for i, sample in enumerate(dataset):
        gt = ground_truth_number(sample)
        if gt is None:
            skipped += 1
            continue

        pil_image = pil_from_sample(sample)
        if pil_image is None:
            skipped += 1
            continue

        question = sample.get("question", "")
        options = sample.get("options", [])
        if len(options) < 2:
            skipped += 1
            continue

        try:
            t0 = time.perf_counter()
            raw = runner_fn(pil_image, question, options)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
        except Exception as e:
            print(f"  [WARN] sample {i}: {e}")
            skipped += 1
            continue

        pred = parse_answer(raw)
        hit = pred == gt

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i + 1:4d}/{total}] gt={gt} pred={pred} raw={raw.strip()!r:20}  "
                f"{'✓' if hit else '✗'}  running_acc={correct / (i + 1 - skipped + 1e-9):.3f}"
            )

        if hit:
            correct += 1

    evaluated = total - skipped
    accuracy = correct / evaluated if evaluated > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print(
        f"\n  {label}: {correct}/{evaluated} correct  |  "
        f"accuracy = {accuracy:.4f} ({accuracy * 100:.2f}%)"
    )
    print(f"  avg latency per sample: {avg_lat:.2f}s  |  skipped: {skipped}")
    return {
        "label": label,
        "accuracy": accuracy,
        "correct": correct,
        "evaluated": evaluated,
        "avg_latency_s": avg_lat,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Eval ONNX vs PyTorch Ministral-3-3B VLM on AI2D"
    )
    parser.add_argument(
        "--model_path",
        default="cpu_and_mobile/models",
        help="Path to ONNX model dir (default: cpu_and_mobile/models/)",
    )
    parser.add_argument(
        "--pytorch_model",
        default=None,
        help="HuggingFace model ID for PyTorch comparison",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of AI2D test samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for inference: cpu, cuda, or auto-detect (default: auto)",
    )
    parser.add_argument(
        "--skip_onnx",
        action="store_true",
        help="Skip ONNX evaluation",
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to suppress chain-of-thought. Pass empty string to disable.",
    )
    args = parser.parse_args()

    ds = load_ai2d(args.num_samples)
    results = []

    sys_prompt = args.system_prompt
    if sys_prompt:
        print(f"\nSystem prompt: {sys_prompt!r}")
    else:
        print("\nSystem prompt: (none)")

    # Detect ONNX precision from model path
    onnx_precision = detect_onnx_precision(args.model_path)

    # ---- ONNX ----
    if not args.skip_onnx:
        onnx_model, onnx_proc, onnx_tok = build_onnx_runner(args.model_path)

        def onnx_runner(pil_image, question, options):
            msgs = build_messages(question, options, sys_prompt)
            return run_onnx(onnx_model, onnx_proc, onnx_tok, pil_image, msgs)

        onnx_label = f"ONNX ({onnx_precision.upper()}) @ {args.model_path}"
        results.append(evaluate(ds, onnx_runner, onnx_label))

    # ---- PyTorch (optional) ----
    pt_precision = None
    if args.pytorch_model:
        pt_model, pt_proc, pt_device, pt_precision = build_pytorch_runner(
            args.pytorch_model, device=args.device
        )

        def pt_runner(pil_image, question, options):
            return run_pytorch(
                pt_model, pt_proc, pil_image, question, options, pt_device, sys_prompt
            )

        pt_label = f"PyTorch ({pt_precision.upper()}) @ {args.pytorch_model}"
        results.append(evaluate(ds, pt_runner, pt_label))

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print("  Model       : Ministral-3-3B-Instruct-2512 (VLM)")
    print("  Dataset     : AI2D (science diagram QA, multiple choice)")
    print(f"  Samples     : {args.num_samples}")
    print(f"  ONNX prec   : {onnx_precision.upper()}")
    if pt_precision:
        print(f"  PyTorch prec: {pt_precision.upper()}")
    print(
        f"  System prompt: "
        f"{'(none)' if not sys_prompt else sys_prompt[:80] + ('...' if len(sys_prompt) > 80 else '')}"
    )
    print()
    for r in results:
        print(f"  {r['label']}")
        print(
            f"    Accuracy : {r['accuracy'] * 100:.2f}%  ({r['correct']}/{r['evaluated']})"
        )
        print(f"    Avg lat  : {r['avg_latency_s']:.2f}s/sample")
        print()

    if len(results) == 2:
        delta = results[0]["accuracy"] - results[1]["accuracy"]
        abs_delta = abs(delta) * 100
        print(f"  Accuracy delta (ONNX - PyTorch): {delta * 100:+.2f} pp")
        print(
            f"  Speedup (PyTorch lat / ONNX lat): "
            f"{results[1]['avg_latency_s'] / max(results[0]['avg_latency_s'], 1e-9):.2f}x"
        )

        # Precision gap assessment
        expected_gap = EXPECTED_GAP_PP.get(onnx_precision, 5.0)
        print()
        print(f"  Expected gap for {onnx_precision.upper()}: <{expected_gap:.0f} pp")
        if abs_delta <= expected_gap:
            print(f"  ✓ PASS — {abs_delta:.2f} pp gap is within expected range")
        else:
            print(
                f"  ✗ WARN — {abs_delta:.2f} pp gap exceeds expected {expected_gap:.0f} pp for {onnx_precision.upper()}"
            )
            print(
                "           This may indicate a quality regression in the export pipeline."
            )


if __name__ == "__main__":
    main()
