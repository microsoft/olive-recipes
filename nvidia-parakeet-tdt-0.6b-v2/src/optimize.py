"""Optimized export of Parakeet TDT 0.6B v2: encoder INT4 + decoder/joint FP32.

  Encoder: OnnxConversion -> OrtTransformersOptimization(conformer) -> INT4 k-quant
  Decoder: OnnxConversion (FP32)
  Joint:   OnnxConversion (FP32)

After the Olive pipelines, the tokenizer files, genai_config.json, and
audio_processor_config.json are generated inline (loading the NeMo model
to extract vocab + scores from the SentencePiece tokenizer).

Usage:
    python src/optimize.py
    python src/optimize.py --output-dir /some/other/path

    # Or run individual Olive configs:
    python -m olive run --config src/parakeet_encoder_int4.json
    python -m olive run --config src/parakeet_decoder_fp32.json
    python -m olive run --config src/parakeet_joint_fp32.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_RECIPE_ROOT = _SCRIPT_DIR.parent
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _SCRIPT_DIR / p


def _run_olive_pipeline(config_name: str, output_dir: str, output_subdir: str):
    """Run an Olive pipeline from a JSON config, overriding output_dir."""
    from olive import run as olive_run

    config_path = _SCRIPT_DIR / config_name
    with open(config_path) as f:
        config = json.load(f)

    config["output_dir"] = str(_resolve(output_dir) / output_subdir)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(_SCRIPT_DIR), delete=False
    ) as tmp:
        json.dump(config, tmp, indent=4)
        tmp_path = tmp.name

    try:
        olive_run(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_olive_pipelines(output_dir: str):
    print("=== Stage 1: Olive Encoder (Convert -> Fusion -> INT4 k-quant) ===")
    _run_olive_pipeline("parakeet_encoder_int4.json", output_dir, "encoder.onnx")
    print()

    print("=== Stage 2: Olive Decoder (OnnxConversion, FP32) ===")
    _run_olive_pipeline("parakeet_decoder_fp32.json", output_dir, "decoder.onnx")
    print()

    print("=== Stage 3: Olive Joint (OnnxConversion, FP32) ===")
    _run_olive_pipeline("parakeet_joint_fp32.json", output_dir, "joint.onnx")
    print()


def run_config_generation(output_dir: str, model_name: str, device: str):
    """Generate tokenizer files, genai_config.json, and audio_processor_config.json.

    Loads the NeMo model once and extracts the SentencePiece vocab + scores
    so the Unigram tokenizer matches what was used during training.
    """
    print("=== Stage 4: Generating tokenizer + genai_config.json + audio_processor_config.json ===")
    import nemo.collections.asr as nemo_asr

    out_dir = _resolve(output_dir)

    print(f"  Loading {model_name} ...")
    if model_name.endswith(".nemo"):
        asr_model = nemo_asr.models.ASRModel.restore_from(model_name)
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    asr_model.eval()

    blank_id, vocab_size = _build_tokenizer_files(asr_model, out_dir)
    print(f"  [OK] tokenizer.json / tokenizer_config.json / vocab.txt"
          f"  (vocab_size={vocab_size}, blank_id={blank_id})")

    _build_genai_config(
        out_dir,
        asr_model=asr_model,
        blank_id=blank_id,
        vocab_size=vocab_size,
        device=device,
    )
    print(f"  [OK] genai_config.json (device={device})")
    print("  [OK] audio_processor_config.json")
    print()


def _build_tokenizer_files(asr_model, out_dir: Path):
    """Extract Unigram vocab + scores from the NeMo SentencePiece tokenizer."""
    sp = asr_model.tokenizer.tokenizer  # sentencepiece.SentencePieceProcessor
    vocab_size = sp.get_piece_size()  # 1024 for v2

    vocab = [[sp.id_to_piece(i), float(sp.get_score(i))] for i in range(vocab_size)]
    blank_id = vocab_size
    vocab.append(["<blank>", 0.0])

    unk_id = sp.unk_id() if sp.unk_id() >= 0 else 0

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": unk_id,
                "content": vocab[unk_id][0],
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": blank_id,
                "content": "<blank>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "add_prefix_space": True,
            "prepend_scheme": "always",
        },
        "model": {
            "type": "Unigram",
            "unk_id": unk_id,
            "vocab": vocab,
        },
    }
    (out_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer_json, indent=2, ensure_ascii=False)
    )

    tokenizer_config = {
        "tokenizer_class": "T5Tokenizer",
        "unk_token": vocab[unk_id][0],
        "eos_token": "<blank>",
        "pad_token": "<blank>",
        "model_max_length": 8192,
        "sp_model_kwargs": {},
        "added_tokens_decoder": {
            str(unk_id): {
                "content": vocab[unk_id][0],
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            str(blank_id): {
                "content": "<blank>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
        },
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False)
    )

    with open(out_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for piece, _score in vocab:
            f.write(f"{piece}\n")

    return blank_id, len(vocab)


def _build_genai_config(out_dir: Path, asr_model, blank_id: int, vocab_size: int, device: str = "cpu"):
    from src.parakeet_model_load import (
        D_MODEL, N_LAYERS, DECODER_HIDDEN, DECODER_LSTM_LAYERS, SUBSAMPLING_FACTOR,
    )

    # Pull preprocessor params straight from the NeMo cfg so we never drift
    # from how the model was trained.
    preprocessor_cfg = asr_model.cfg.get("preprocessor", {})
    sample_rate = preprocessor_cfg.get("sample_rate", 16000)
    n_mels = preprocessor_cfg.get("features", preprocessor_cfg.get("nfilt", 128))
    n_fft = preprocessor_cfg.get("n_fft", 512)
    preemph = preprocessor_cfg.get("preemph", 0.97)
    dither = preprocessor_cfg.get("dither", 1e-5)
    normalize = preprocessor_cfg.get("normalize", "per_feature")

    window_size = preprocessor_cfg.get("window_size", 0.025)
    window_stride = preprocessor_cfg.get("window_stride", 0.01)
    win_length = preprocessor_cfg.get("win_length")
    if win_length is None:
        win_length = int(window_size * sample_rate) if isinstance(window_size, float) and window_size < 1.0 else int(window_size)
    hop_length = preprocessor_cfg.get("hop_length")
    if hop_length is None:
        hop_length = int(window_stride * sample_rate) if isinstance(window_stride, float) and window_stride < 1.0 else int(window_stride)

    max_symbols = asr_model.cfg.get("decoding", {}).get("greedy", {}).get("max_symbols", 10)

    if device == "cuda":
        cuda_opts = {"provider_options": [{"cuda": {}}]}
        encoder_session_options = cuda_opts
        decoder_session_options = cuda_opts
        joiner_session_options = cuda_opts
    else:
        encoder_session_options = None
        decoder_session_options = None
        joiner_session_options = None

    encoder_section = {
        "filename": "encoder.onnx",
        "hidden_size": D_MODEL,
        "num_hidden_layers": N_LAYERS,
        "inputs": {
            "audio_features": "audio_signal",
            "input_lengths": "length",
        },
        "outputs": {
            "encoder_outputs": "outputs",
            "output_lengths": "encoded_lengths",
        },
    }
    decoder_section = {
        "filename": "decoder.onnx",
        "hidden_size": DECODER_HIDDEN,
        "num_hidden_layers": DECODER_LSTM_LAYERS,
        "inputs": {
            "targets": "targets",
            "targets_length": "target_length_orig",
            "lstm_hidden_state": "h_in",
            "lstm_cell_state": "c_in",
        },
        "outputs": {
            "outputs": "decoder_output",
            "outputs_length": "target_length",
            "lstm_hidden_state": "h_out",
            "lstm_cell_state": "c_out",
        },
    }
    joiner_section = {
        "filename": "joint.onnx",
        "inputs": {
            "encoder_outputs": "encoder_output",
            "decoder_outputs": "decoder_output",
        },
        "outputs": {
            "logits": "joint_output",
        },
    }
    if encoder_session_options is not None:
        encoder_section = {"session_options": encoder_session_options, **encoder_section}
    if decoder_session_options is not None:
        decoder_section = {"session_options": decoder_session_options, **decoder_section}
    if joiner_session_options is not None:
        joiner_section = {"session_options": joiner_session_options, **joiner_section}

    # Chunk = 96000 samples / 16 kHz = 6 s of audio per encoder call.
    # Pairs with left_context_samples=160000 (10 s) and right_context_samples=32000
    # (2 s lookahead), giving an 18 s receptive window per chunk.
    chunk_samples = 96000
    cfg = {
        "model": {
            "type": "parakeet_tdt",
            "vocab_size": vocab_size,
            # 131072 = 2^17 output tokens. At ~80 ms per encoder frame and 1 token
            # per step in typical speech, this caps decoding at ~2.9 h of audio.
            "context_length": 131072,
            "decoder_start_token_id": blank_id,
            "eos_token_id": blank_id,
            "encoder": encoder_section,
            "decoder": decoder_section,
            "joiner": joiner_section,
            "speech": {"config_filename": "audio_processor_config.json"},
            "sample_rate": sample_rate,
            "num_mels": n_mels,
            "fft_size": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "preemph": preemph,
            "log_eps": 5.9604645e-08,
            "subsampling_factor": SUBSAMPLING_FACTOR,
            "chunk_samples": chunk_samples,
            "left_context_samples": 160000,
            "right_context_samples": 32000,
            "blank_id": blank_id,
            "max_symbols_per_step": max_symbols,
            "tdt_durations": [0, 1, 2, 3, 4],
            "norm_eps": 1e-05,
        },
        "search": {
            "max_length": 131072,
            "min_length": 0,
        },
    }
    (out_dir / "genai_config.json").write_text(json.dumps(cfg, indent=2))

    apc = {
        "model_type": "speech_features",
        "audio_params": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "window_length": win_length,
            "window_type": "hann",
            "fmin": 0,
            "fmax": sample_rate // 2,
            "dither": dither,
            "preemphasis": preemph,
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 1e-10,
            "normalize": normalize,
            "center": True,
            "mag_power": 2.0,
        },
    }
    (out_dir / "audio_processor_config.json").write_text(json.dumps(apc, indent=2))


def main():
    from src.parakeet_model_load import MODEL_NAME

    parser = argparse.ArgumentParser(
        description="Optimized ONNX export of Parakeet TDT 0.6B v2 (INT4 encoder + FP32 decoder/joint)"
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="HuggingFace model name or path to a local .nemo file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--skip-configs",
        action="store_true",
        help="Skip stage 4 (only export ONNX files)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Target execution provider for the generated genai_config.json. "
             "Only affects the encoder session_options; the ONNX graphs themselves "
             "are identical for both targets (default: cpu).",
    )
    args = parser.parse_args()

    if not args.model_name.endswith(".nemo") and args.model_name != MODEL_NAME:
        raise ValueError(
            f"This recipe targets '{MODEL_NAME}' (or a .nemo file with the same architecture). "
            f"Got: '{args.model_name}'"
        )

    run_olive_pipelines(output_dir=args.output_dir)

    if not args.skip_configs:
        run_config_generation(
            output_dir=args.output_dir,
            model_name=args.model_name,
            device=args.device,
        )

    output_path = Path(args.output_dir)
    files = sorted(f for f in output_path.rglob("*") if f.is_file())
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print(f"=== Done! ONNX models -> {output_path} ===")
    print(f"    Total size: {total_mb:.1f} MB")
    for f in files:
        rel = f.relative_to(output_path)
        tag = " <- INT4 k-quant" if f.name.startswith("encoder") and f.suffix in (".onnx", ".data") else ""
        print(f"    {rel} ({f.stat().st_size / (1024 * 1024):.1f} MB){tag}")


if __name__ == "__main__":
    main()
