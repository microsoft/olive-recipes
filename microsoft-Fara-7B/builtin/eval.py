"""Evaluate Fara-7B ONNX model on ScreenSpot-v2 (GUI grounding benchmark).

ScreenSpot-v2 measures a model's ability to localize UI elements given a
natural-language instruction and a screenshot.  Each sample has an image,
an instruction (e.g. "open settings"), and a ground-truth bounding box
[x, y, w, h].  The model must predict a click coordinate that falls inside
the bounding box.

Dataset: OS-Copilot/ScreenSpot-v2
  - screenspot_desktop_v2.json, screenspot_mobile_v2.json, screenspot_web_v2.json
  - screenspotv2_image.zip  (extract to get image files)

Usage:
    # Evaluate ONNX model on desktop split (default)
    python eval.py --image_dir /path/to/screenspotv2_images

    # Evaluate on all splits
    python eval.py --image_dir /path/to/images --split all

    # Compare ONNX vs PyTorch
    python eval.py --image_dir /path/to/images --pytorch_model microsoft/Fara-7B

    # Limit samples
    python eval.py --image_dir /path/to/images --num_samples 50
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import onnxruntime_genai as og
from PIL import Image

# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------

def parse_click_coords(response: str, img_width: int, img_height: int):
    """Extract predicted click (x, y) in pixel coordinates from model response.

    Fara-7B outputs tool calls like:
        {"name": "Click", "arguments": {"x": 450, "y": 230}}
    or with normalized coordinates:
        {"name": "Click", "arguments": {"x": 0.45, "y": 0.23}}

    Returns (x, y) in pixels or None if parsing fails.
    """
    # Try to extract JSON from tool_call block
    tool_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if not tool_match:
        # Try without closing tag (truncated)
        tool_match = re.search(r'<tool_call>\s*(\{.*\})', response, re.DOTALL)

    if tool_match:
        try:
            tool_json = json.loads(tool_match.group(1))
            args = tool_json.get("arguments", tool_json)

            x = args.get("x", args.get("coordinate", [None, None]))
            y = args.get("y", None)

            # Handle coordinate list format
            if isinstance(x, list) and len(x) == 2:
                x, y = x[0], x[1]

            if x is not None and y is not None:
                x, y = float(x), float(y)
                # If normalized (0-1 range), convert to pixels
                if 0 <= x <= 1 and 0 <= y <= 1:
                    x = x * img_width
                    y = y * img_height
                return (x, y)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Tool-call JSON can be malformed or incomplete; ignore and continue
            # to fallback coordinate parsing below.
            pass

    # Fallback: look for coordinate patterns like (450, 230) or [450, 230]
    coord_match = re.search(r'[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]', response)
    if coord_match:
        x, y = float(coord_match.group(1)), float(coord_match.group(2))
        if 0 <= x <= 1 and 0 <= y <= 1:
            x = x * img_width
            y = y * img_height
        return (x, y)

    return None


def is_hit(pred_x: float, pred_y: float, bbox: list) -> bool:
    """Check if predicted click falls within the ground-truth bbox [x, y, w, h]."""
    bx, by, bw, bh = bbox
    return bx <= pred_x <= bx + bw and by <= pred_y <= by + bh


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

SPLIT_FILES = {
    "desktop": "screenspot_desktop_v2.json",
    "mobile": "screenspot_mobile_v2.json",
    "web": "screenspot_web_v2.json",
}


def load_screenspot(data_dir: str, split: str, num_samples: int):
    """Load ScreenSpot-v2 samples for the given split(s)."""
    if split == "all":
        splits = list(SPLIT_FILES.keys())
    else:
        splits = [split]

    samples = []
    for s in splits:
        json_path = os.path.join(data_dir, SPLIT_FILES[s])
        if not os.path.exists(json_path):
            print(f"  Warning: {json_path} not found, skipping {s} split.")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            item["split"] = s
        samples.extend(data)
        print(f"  Loaded {len(data)} samples from {s} split.")

    if num_samples and num_samples < len(samples):
        samples = samples[:num_samples]
    print(f"  Total samples to evaluate: {len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def build_onnx_runner(model_path: str):
    print(f"\nLoading ONNX model from: {model_path}")
    model = og.Model(model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    print("  ONNX model loaded.")
    return model, processor, tokenizer


def run_onnx(model, processor, tokenizer, pil_image: Image.Image, instruction: str, system_prompt: str) -> str:
    """Run a single inference with the ONNX GenAI model."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]
    })

    # Save PIL image to temp file for og.Images
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        pil_image.save(f, format="PNG")
        tmp_path = f.name

    try:
        images = og.Images.open(tmp_path)
        prompt = tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)
        inputs = processor(prompt, images=images)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=4096, do_sample=False)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        tokens = []
        stream = tokenizer.create_stream()
        while not generator.is_done():
            generator.generate_next_token()
            tokens.append(generator.get_next_tokens()[0])
        del generator

        return tokenizer.decode(tokens)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# PyTorch inference (optional)
# ---------------------------------------------------------------------------

def build_pytorch_runner(model_id: str):
    print(f"\nLoading PyTorch model: {model_id}")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  Device: {device}, dtype: {dtype}")

    pt_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    pt_proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("  PyTorch model loaded.")
    return pt_model, pt_proc, device


def run_pytorch(pt_model, pt_proc, pil_image: Image.Image, instruction: str, device: str, system_prompt: str) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": instruction}
        ]
    })

    text = pt_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = pt_proc(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = pt_model.generate(**inputs, max_new_tokens=256, do_sample=False)

    out_ids = out[0][inputs["input_ids"].shape[-1]:]
    return pt_proc.decode(out_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(samples, image_dir, runner_fn, label: str):
    """Run evaluation and compute accuracy by split and data_type."""
    correct = 0
    total = 0
    skipped = 0
    latencies = []

    # Per-category tracking
    by_split = {}
    by_type = {}

    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}  ({len(samples)} samples)")
    print(f"{'='*60}")

    for i, sample in enumerate(samples):
        img_path = os.path.join(image_dir, sample["img_filename"])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        img_w, img_h = pil_image.size
        instruction = sample["instruction"]
        bbox = sample["bbox"]
        split = sample.get("split", "unknown")
        data_type = sample.get("data_type", "unknown")

        try:
            t0 = time.perf_counter()
            response = runner_fn(pil_image, instruction)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
        except Exception as e:
            print(f"  [WARN] sample {i}: {e}")
            skipped += 1
            continue

        coords = parse_click_coords(response, img_w, img_h)
        hit = False
        if coords:
            hit = is_hit(coords[0], coords[1], bbox)

        total += 1
        if hit:
            correct += 1

        # Track by category
        for key, d in [(split, by_split), (data_type, by_type)]:
            if key not in d:
                d[key] = {"correct": 0, "total": 0}
            d[key]["total"] += 1
            if hit:
                d[key]["correct"] += 1

        if (i + 1) % 20 == 0 or i == 0:
            pred_str = f"({coords[0]:.0f},{coords[1]:.0f})" if coords else "None"
            bbox_str = f"({bbox[0]},{bbox[1]},{bbox[0]+bbox[2]},{bbox[1]+bbox[3]})"
            print(f"  [{i+1:4d}/{len(samples)}] {'HIT' if hit else 'MISS':4s} "
                  f"pred={pred_str:14s} bbox={bbox_str:20s} "
                  f"acc={correct/max(total,1):.3f}  \"{instruction[:40]}\"")

    accuracy = correct / total if total > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"\n  {label}: {correct}/{total} correct  |  accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  avg latency: {avg_lat:.2f}s/sample  |  skipped: {skipped}")

    # Breakdown by split
    if len(by_split) > 1:
        print(f"\n  By split:")
        for s, d in sorted(by_split.items()):
            acc = d["correct"] / d["total"] if d["total"] > 0 else 0
            print(f"    {s:10s}: {d['correct']:4d}/{d['total']:4d}  ({acc*100:.2f}%)")

    # Breakdown by type
    print(f"\n  By element type:")
    for t, d in sorted(by_type.items()):
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0
        print(f"    {t:10s}: {d['correct']:4d}/{d['total']:4d}  ({acc*100:.2f}%)")

    return {
        "label": label, "accuracy": accuracy, "correct": correct, "total": total,
        "avg_latency_s": avg_lat, "skipped": skipped,
        "by_split": by_split, "by_type": by_type,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fara-7B on ScreenSpot-v2 GUI grounding benchmark")
    parser.add_argument("--model_path", default="cpu_and_mobile/models",
                        help="Path to ONNX model dir (default: cpu_and_mobile/models/)")
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing ScreenSpot-v2 images (extracted from screenspotv2_image.zip)")
    parser.add_argument("--data_dir", default=None,
                        help="Directory containing ScreenSpot-v2 JSON files. Defaults to image_dir parent.")
    parser.add_argument("--split", default="desktop", choices=["desktop", "mobile", "web", "all"],
                        help="Which split to evaluate (default: desktop)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of samples (default: all)")
    parser.add_argument("--pytorch_model", default=None,
                        help="HF model ID for PyTorch comparison (e.g. microsoft/Fara-7B)")
    parser.add_argument("--skip_onnx", action="store_true",
                        help="Skip ONNX evaluation")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt for the model")
    args = parser.parse_args()

    data_dir = args.data_dir or str(Path(args.image_dir).parent)

    print(f"ScreenSpot-v2 Evaluation")
    print(f"  Data dir:  {data_dir}")
    print(f"  Image dir: {args.image_dir}")
    print(f"  Split:     {args.split}")

    samples = load_screenspot(data_dir, args.split, args.num_samples)
    if not samples:
        print("No samples loaded. Check --data_dir and --image_dir paths.")
        sys.exit(1)

    results = []
    sys_prompt = args.system_prompt

    # ---- ONNX ----
    if not args.skip_onnx:
        onnx_model, onnx_proc, onnx_tok = build_onnx_runner(args.model_path)

        def onnx_runner(pil_image, instruction):
            return run_onnx(onnx_model, onnx_proc, onnx_tok, pil_image, instruction, sys_prompt)

        results.append(evaluate(samples, args.image_dir, onnx_runner, f"ONNX @ {args.model_path}"))

    # ---- PyTorch (optional) ----
    if args.pytorch_model:
        pt_model, pt_proc, device = build_pytorch_runner(args.pytorch_model)

        def pt_runner(pil_image, instruction):
            return run_pytorch(pt_model, pt_proc, pil_image, instruction, device, sys_prompt)

        results.append(evaluate(samples, args.image_dir, pt_runner, f"PyTorch @ {args.pytorch_model}"))

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Benchmark  : ScreenSpot-v2 (GUI grounding)")
    print(f"  Split      : {args.split}")
    print(f"  Samples    : {len(samples)}")
    print()
    for r in results:
        print(f"  {r['label']}")
        print(f"    Accuracy : {r['accuracy']*100:.2f}%  ({r['correct']}/{r['total']})")
        print(f"    Avg lat  : {r['avg_latency_s']:.2f}s/sample")
        print()

    if len(results) == 2:
        delta = results[1]["accuracy"] - results[0]["accuracy"]
        print(f"  Accuracy delta (PyTorch - ONNX): {delta*100:+.2f} pp")
        print(f"  Speedup (PyTorch lat / ONNX lat): "
              f"{results[1]['avg_latency_s'] / max(results[0]['avg_latency_s'], 1e-9):.2f}x")


if __name__ == "__main__":
    main()
