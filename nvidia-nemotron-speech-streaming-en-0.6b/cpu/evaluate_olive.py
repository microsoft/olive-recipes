"""Olive-compatible evaluation function for Nemotron Speech Streaming.

Used by the Olive CUSTOM metric evaluator when running:
    python -m olive run --config cpu/nemotron_eval_wer.json

The evaluate_func receives (model, device, execution_providers) and returns
a dict of metric values {"wer": float, "rtfx": float}.
"""

import json
import os

import jiwer
import numpy as np


SAMPLE_RATE = 16000


def _normalize_text(text: str) -> str:
    """Normalize text for WER computation."""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        return normalizer(text)
    except ImportError:
        import re

        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text


def _transcribe_streaming(og_model, tokenizer, audio_array: np.ndarray, chunk_samples: int) -> str:
    """Run streaming transcription using onnxruntime-genai."""
    import onnxruntime_genai as og

    audio = audio_array.astype(np.float32)
    processor = og.StreamingProcessor(og_model)
    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(og_model)
    generator = og.Generator(og_model, params)

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

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples].astype(np.float32)
        inputs = processor.process(chunk)
        if inputs is not None:
            generator.set_inputs(inputs)
            decode_chunk()

    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
        decode_chunk()

    # Silence for right context
    for _ in range(4):
        silence = np.zeros(chunk_samples, dtype=np.float32)
        inputs = processor.process(silence)
        if inputs is not None:
            generator.set_inputs(inputs)
            decode_chunk()

    return transcript


def evaluate_nemotron_wer(
    model,
    device,
    execution_providers,
    dataset_path="hf-audio/esb-datasets-test-only-sorted",
    dataset_name="librispeech",
    split="test.clean",
    limit=64,
    seed=42,
    max_samples=None,
):
    """Evaluate Nemotron ASR model and return WER/RTFx metrics.

    This function is called by Olive's CUSTOM metric evaluator.

    Args:
        model: OliveModelHandler
        device: Device enum
        execution_providers: List of execution providers
        dataset_path: HuggingFace dataset path
        dataset_name: Dataset subset name
        split: Dataset split
        limit: Sampling limit (>=1: count, 0<x<1: percentage, 0: all). Default 64.
        seed: Random seed for percentage-based sampling
        max_samples: Deprecated alias for limit (for backward compat)

    Returns:
        dict with "wer" and "rtfx" float values

    """
    import onnxruntime_genai as og
    from datasets import Audio, load_dataset
    from random import Random

    # Backward compat: max_samples overrides limit
    if max_samples is not None:
        limit = max_samples

    # Get model directory path
    model_dir = str(model.model_path)

    # Load model config
    config_path = os.path.join(model_dir, "genai_config.json")
    with open(config_path, "r") as f:
        genai_config = json.load(f)
    sample_rate = genai_config["model"]["sample_rate"]
    chunk_samples = genai_config["model"]["chunk_samples"]

    # Load onnxruntime-genai model
    og_config = og.Config(model_dir)
    og_model = og.Model(og_config)
    tokenizer = og.Tokenizer(og_model)

    # Load dataset
    dataset = load_dataset(
        dataset_path, dataset_name, split=split, streaming=False, trust_remote_code=True
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))

    # Apply sampling limit (same semantics as Olive's --limit)
    if limit != 0:
        total = len(dataset)
        if 0 < limit < 1:
            n = max(1, int(total * limit))
            rng = Random(seed)
            indices = sorted(rng.sample(range(total), min(n, total)))
            dataset = dataset.select(indices)
        elif limit >= 1:
            n = min(int(limit), total)
            dataset = dataset.select(range(n))

    predictions = []
    references = []

    for sample in dataset:
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        ref_text = sample.get("text", sample.get("sentence", ""))
        ref_norm = _normalize_text(ref_text)

        if not ref_norm.strip():
            continue

        pred_text = _transcribe_streaming(og_model, tokenizer, audio_array, chunk_samples)
        pred_norm = _normalize_text(pred_text)

        predictions.append(pred_norm)
        references.append(ref_norm)

    if not predictions:
        return {"wer": 1.0, "rtfx": 0.0}

    wer = jiwer.wer(references, predictions)

    print(f"Nemotron WER: {100*wer:.2f}% ({len(predictions)} samples)")
    return {"wer": wer}
