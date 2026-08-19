"""Evaluate the Qwen3.6-35B-A3B ORT GenAI package on AI2D diagram QA."""

import argparse
import io
import json
import re
import tempfile
import time
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from inference_vl import generate_response, load_runtime

DEFAULT_MODEL_PATH = "cuda/vl_kquant_fp16/models"
DEFAULT_SYSTEM_PROMPT = "Answer concisely with just the option number. /no_think"
OPTION_NUMBERS = ("1", "2", "3", "4")


def build_prompt(question: str, options: list[str]) -> str:
    option_text = "\n".join(f"{number}. {option}" for number, option in zip(OPTION_NUMBERS, options))
    prompt = (
        "Look at the diagram and answer the multiple-choice question.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_text}\n\n"
        "Reply with the number only (1, 2, 3, or 4)."
    )
    return prompt


def parse_answer(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    matches = re.findall(r"\b([1-4])\b", text)
    return matches[-1] if matches else None


def ground_truth_number(sample: dict) -> str | None:
    try:
        answer_index = int(sample.get("answer", ""))
    except (TypeError, ValueError):
        return None
    return OPTION_NUMBERS[answer_index] if 0 <= answer_index < len(OPTION_NUMBERS) else None


def sample_image(sample: dict) -> Image.Image | None:
    image = sample.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, dict) and image.get("bytes"):
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.6 VL KQuant on AI2D")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, type=Path, help="Model directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of AI2D test samples")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum total sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens per answer")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Instruction before each question")
    args = parser.parse_args()

    if not args.model_path.is_dir():
        parser.error(f"model directory not found: {args.model_path}")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")

    dataset = load_dataset("lmms-lab/ai2d", split="test")
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    runtime = load_runtime(args.model_path)

    correct = 0
    evaluated = 0
    latencies = []
    for index, sample in enumerate(dataset, start=1):
        expected = ground_truth_number(sample)
        image = sample_image(sample)
        options = sample.get("options", [])
        if expected is None or image is None or len(options) != 4:
            print(f"[{index}/{len(dataset)}] skipped malformed sample")
            continue

        prompt = build_prompt(sample.get("question", ""), options)
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            image.save(image_file, format="PNG")
            image_file.flush()
            start = time.perf_counter()
            output, _ = generate_response(
                *runtime,
                prompt,
                Path(image_file.name),
                args.max_length,
                args.max_new_tokens,
                stream=False,
                system_prompt=args.system_prompt,
            )
            latencies.append(time.perf_counter() - start)

        predicted = parse_answer(output)
        evaluated += 1
        correct += predicted == expected
        print(
            f"[{index}/{len(dataset)}] expected={expected} predicted={predicted} "
            f"running_accuracy={correct / evaluated:.3f} output={output.strip()!r}"
        )

    if not evaluated:
        raise RuntimeError("AI2D evaluation produced no valid samples")

    result = {
        "dataset": "lmms-lab/ai2d",
        "samples_requested": args.num_samples,
        "samples_evaluated": evaluated,
        "correct": correct,
        "accuracy": correct / evaluated,
        "average_latency_s": sum(latencies) / len(latencies),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
