"""Evaluate Nemotron Speech Streaming ASR model with WER metric.

Loads a HuggingFace ASR dataset, runs streaming inference using
onnxruntime-genai, and computes Word Error Rate (WER).

Sampling follows the same convention as Olive's LLM benchmark:
  --limit N       → use first N samples
  --limit 0.1     → use 10% of the dataset (randomly sampled with seed)

Compatible with both:
  - Pre-built ONNX model directories (from cpu/optimize.py)
  - onnxruntime-genai StreamingProcessor API

Usage:
    # Evaluate on 50 samples from LibriSpeech test.clean
    python cpu/evaluate.py --model_dir build/onnx_models_int4 \
        --dataset librispeech --split test.clean --limit 50

    # Evaluate on 10% of the dataset (randomly sampled)
    python cpu/evaluate.py --model_dir build/onnx_models_int4 \
        --dataset librispeech --split test.clean --limit 0.1

    # Full evaluation (all samples)
    python cpu/evaluate.py --model_dir build/onnx_models_int4 \
        --dataset librispeech --split test.clean --limit 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from random import Random

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_RECIPE_ROOT = _SCRIPT_DIR.parent
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))


SAMPLE_RATE = 16000
DEFAULT_LIMIT = 64  # Default sample count, same as Olive benchmark default


def _load_model_config(model_dir: str):
    """Read sample_rate and chunk_samples from genai_config.json."""
    config_path = os.path.join(model_dir, "genai_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    sample_rate = config["model"]["sample_rate"]
    chunk_samples = config["model"]["chunk_samples"]
    return sample_rate, chunk_samples


def _transcribe_streaming(model, tokenizer, audio_array: np.ndarray, chunk_samples: int) -> str:
    """Run streaming transcription on a single audio sample using onnxruntime-genai."""
    import onnxruntime_genai as og

    audio = audio_array.astype(np.float32)
    processor = og.StreamingProcessor(model)
    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(model)
    generator = og.Generator(model, params)

    transcript = ""

    def decode_chunk():
        nonlocal transcript
        while not generator.is_done():
            generator.generate_next_token()
            tokens = generator.get_next_tokens()
            if len(tokens) > 0:
                text = tokenizer_stream.decode(tokens[0])
                if text:
                    transcript += text

    # Process audio in streaming chunks
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples].astype(np.float32)
        inputs = processor.process(chunk)
        if inputs is not None:
            generator.set_inputs(inputs)
            decode_chunk()

    # Flush remaining audio
    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
        decode_chunk()

    # Feed silence for right context
    for _ in range(4):
        silence = np.zeros(chunk_samples, dtype=np.float32)
        inputs = processor.process(silence)
        if inputs is not None:
            generator.set_inputs(inputs)
            decode_chunk()

    return transcript


def _normalize_text(text: str) -> str:
    """Normalize text for WER computation using the Open ASR Leaderboard normalizer."""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        return normalizer(text)
    except ImportError:
        # Fallback: basic normalization
        import re

        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text


def _apply_limit(dataset, limit: float, seed: int = 42):
    """Apply sampling limit to dataset following Olive's convention.

    Args:
        dataset: HuggingFace dataset
        limit: If >= 1, treated as sample count. If 0 < limit < 1, treated as
               percentage. If 0, use all samples.
        seed: Random seed for percentage-based sampling.

    Returns:
        Subsampled dataset.
    """
    if limit == 0:
        return dataset  # Use all samples

    total = len(dataset)

    if 0 < limit < 1:
        # Percentage-based: randomly sample limit% of the dataset
        n = max(1, int(total * limit))
        rng = Random(seed)
        indices = sorted(rng.sample(range(total), min(n, total)))
        dataset = dataset.select(indices)
        print(f"  Sampled {len(indices)}/{total} samples ({limit*100:.0f}%, seed={seed})")
    elif limit >= 1:
        # Count-based: take first N samples
        n = min(int(limit), total)
        dataset = dataset.select(range(n))
        print(f"  Using first {n}/{total} samples")

    return dataset


def load_dataset_with_limit(args, sample_rate: int = SAMPLE_RATE):
    """Load HuggingFace ASR dataset and apply sampling limit."""
    from datasets import Audio, load_dataset

    print(f"Loading dataset: {args.dataset_path}/{args.dataset} split={args.split} ...")
    dataset = load_dataset(
        args.dataset_path,
        args.dataset,
        split=args.split,
        streaming=False,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))

    # Apply limit (same semantics as Olive's --limit)
    dataset = _apply_limit(dataset, args.limit, args.seed)
    return dataset


def main(args):
    import jiwer
    import onnxruntime_genai as og

    model_dir = args.model_dir
    print(f"Loading model from {model_dir} ...")
    sample_rate, chunk_samples = _load_model_config(model_dir)
    print(f"  Sample rate: {sample_rate}, Chunk samples: {chunk_samples}")

    config = og.Config(model_dir)
    if args.execution_provider != "cpu":
        config.clear_providers()
        config.append_provider(args.execution_provider)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)

    dataset = load_dataset_with_limit(args, sample_rate=sample_rate)

    predictions = []
    references = []
    total_audio_s = 0.0
    total_inference_s = 0.0

    print("Running evaluation...")
    count = 0
    for sample in dataset:
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        audio_dur = len(audio_array) / sr

        # Reference text
        ref_text = sample.get("text", sample.get("sentence", ""))
        ref_norm = _normalize_text(ref_text)

        # Skip empty references
        if not ref_norm.strip():
            continue

        # Run inference
        t0 = time.time()
        pred_text = _transcribe_streaming(model, tokenizer, audio_array, chunk_samples)
        elapsed = time.time() - t0

        pred_norm = _normalize_text(pred_text)

        predictions.append(pred_norm)
        references.append(ref_norm)
        total_audio_s += audio_dur
        total_inference_s += elapsed
        count += 1

        if args.verbose:
            sample_wer = jiwer.wer(ref_norm, pred_norm) if ref_norm.strip() else 0.0
            rtfx = audio_dur / max(elapsed, 1e-9)
            print(
                f"  [{count:>4d}] dur={audio_dur:.1f}s WER={100*sample_wer:.1f}% RTFx={rtfx:.1f}\n"
                f"    HYP: {pred_text}\n"
                f"    REF: {ref_text}"
            )

        if count % 50 == 0:
            running_wer = jiwer.wer(references, predictions)
            print(f"  ... {count} samples, running WER: {100*running_wer:.2f}%")

    # Final metrics
    if not predictions:
        print("No samples evaluated!")
        return

    wer = jiwer.wer(references, predictions)
    rtfx = total_audio_s / max(total_inference_s, 1e-9)

    print()
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}/{args.dataset} ({args.split})")
    print(f"Samples: {len(predictions)}")
    print(f"WER:     {100*wer:.2f}%")
    print(f"RTFx:    {rtfx:.2f}")
    print(f"Total audio:     {total_audio_s:.1f}s")
    print(f"Total inference: {total_inference_s:.1f}s")
    print("=" * 60)

    # Save results
    results = {
        "model_dir": str(model_dir),
        "dataset": f"{args.dataset_path}/{args.dataset}",
        "split": args.split,
        "num_samples": len(predictions),
        "limit": args.limit,
        "seed": args.seed,
        "wer": round(wer, 4),
        "rtfx": round(rtfx, 2),
        "total_audio_seconds": round(total_audio_s, 1),
        "total_inference_seconds": round(total_inference_s, 1),
    }

    results_file = Path(args.output_file) if args.output_file else Path(model_dir) / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Nemotron Speech Streaming ASR model with WER"
    )

    # Model
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to the optimized ONNX model directory (output of cpu/optimize.py)",
    )
    parser.add_argument(
        "--execution_provider", type=str, default="cpu",
        help="ORT execution provider (cpu, cuda, dml)",
    )

    # Dataset
    parser.add_argument(
        "--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted",
        help="HuggingFace dataset path",
    )
    parser.add_argument(
        "--dataset", type=str, default="librispeech",
        help="Dataset subset name (e.g., librispeech, ami, earnings22, gigaspeech, ...)",
    )
    parser.add_argument(
        "--split", type=str, default="test.clean",
        help="Dataset split",
    )

    # Sampling (follows Olive benchmark convention)
    parser.add_argument(
        "--limit", type=float, default=DEFAULT_LIMIT,
        help=(
            "Number or percentage of samples to evaluate. "
            "If >= 1: use first N samples. "
            "If 0 < limit < 1: randomly sample that percentage. "
            "If 0: use all samples. "
            f"(default: {DEFAULT_LIMIT})"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for percentage-based sampling (default: 42)",
    )

    # Output
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Path to save results JSON (default: <model_dir>/eval_results.json)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample details",
    )

    args = parser.parse_args()
    main(args)
