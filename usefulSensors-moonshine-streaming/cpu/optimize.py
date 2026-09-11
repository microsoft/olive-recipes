"""End-to-end Olive optimization pipeline for Moonshine Streaming ASR.

Exports the HuggingFace ``MoonshineStreamingForConditionalGeneration`` model
into the five stateful ONNX graphs consumed by onnxruntime-genai's
``streaming_enc_dec_asr`` model type, then generates the runtime config files
and fetches the tokenizer + Silero VAD.  Everything is reproducible from the
Torch checkpoint -- no pre-built ``.ort`` graphs are used.

  frontend / encoder / adapter / cross_kv / decoder_kv
      -> OnnxConversion (FP32, dynamo exporter, dynamic sequence axes)

Usage:
    # Full pipeline (tiny model -> build/onnx)
    python cpu/optimize.py

    # Small model
    python cpu/optimize.py --model-name usefulsensors/moonshine-streaming-small \
        --output-dir build/onnx-small

    # Or run a single component directly through the Olive CLI:
    python -m olive run --config cpu/moonshine_frontend_fp32_cpu.json
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure the recipe root is importable so the Olive model_script resolves.
_SCRIPT_DIR = Path(__file__).resolve().parent
_RECIPE_ROOT = _SCRIPT_DIR.parent
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

MODEL_NAME = "usefulsensors/moonshine-streaming-tiny"
DEFAULT_OUTPUT_DIR = "build/onnx"

# chunk_samples is the streaming window fed to the frontend each step. It must
# be a multiple of frame_len (80) so the stateful conv phase stays aligned and
# the carried sample_buffer is empty between chunks. 8000 = 0.5s @ 16kHz.
CHUNK_SAMPLES = 8000

# Ordered (config file, output basename) for the five component graphs.
COMPONENTS = [
    ("moonshine_frontend_fp32_cpu.json", "frontend.onnx"),
    ("moonshine_encoder_fp32_cpu.json", "encoder.onnx"),
    ("moonshine_adapter_fp32_cpu.json", "adapter.onnx"),
    ("moonshine_cross_kv_fp32_cpu.json", "cross_kv.onnx"),
    ("moonshine_decoder_kv_fp32_cpu.json", "decoder_kv.onnx"),
]

# With --quantize, swap the encoder + decoder_kv to a quantized config
# (frontend/adapter/cross_kv always stay FP32). --quant-method picks the
# algorithm:
#   "kquant8" -> INT8 weight-only k-quant (MatMulNBits, bits=8); uses
#               least-squares refinement for higher accuracy than plain RTN
#               at a similar disk size.
#   "kquant8-enc" -> same INT8 weight-only k-quant as "kquant8" but applied to
#               the ENCODER ONLY; decoder_kv stays FP32 (use when decoder
#               quantization degrades transcription quality).
QUANTIZED_CONFIGS = {
    "kquant8": {
        "encoder.onnx": "moonshine_encoder_kquant8_cpu.json",
        "decoder_kv.onnx": "moonshine_decoder_kv_kquant8_cpu.json",
    },
    "kquant8-enc": {
        # Encoder only -> INT8 k-quant; decoder_kv is omitted so it falls
        # through to the FP32 config in COMPONENTS.
        "encoder.onnx": "moonshine_encoder_kquant8_cpu.json",
    },
}
_QUANT_LABELS = {
    "kquant8": "OnnxConversion -> INT8 k-quant (MatMulNBits)",
    "kquant8-enc": "OnnxConversion -> INT8 k-quant, encoder only (MatMulNBits)",
}

# Streaming pipeline hyper-parameters that are NOT stored in the model weights.
# They describe the onnxruntime-genai runtime contract and match the published
# usefulsensors/moonshine-streaming model card.
PIPELINE = {
    "max_seq_len": 448,
    "tokens_per_second": 6.5,
    "max_segment_memory_frames": 500,
    "min_segment_memory_frames": 250,
    "left_context_frames": 160,
}
VAD = {
    "filename": "silero_vad.onnx",
    "threshold": 0.5,
    "silence_duration_ms": 500,
    "prefix_padding_ms": 200,
}

# genai tokenizer_config.json for the TokenizersBackend (token strings only;
# no model weights). Mirrors the published streaming model's runtime tokenizer.
TOKENIZER_CONFIG = {
    "backend": "tokenizers",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "is_local": True,
    "model_max_length": 4096,
    "pad_token": "<unk>",
    "processor_class": "MoonshineStreamingProcessor",
    "tokenizer_class": "TokenizersBackend",
    "unk_token": "<unk>",
}


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _SCRIPT_DIR / p


def _run_olive_pipeline(config_name, model_name, output_dir, output_subdir):
    """Run one Olive pipeline from a JSON config, overriding the model path and
    output directory so a single set of configs works for any checkpoint."""
    from olive import run as olive_run

    with open(_SCRIPT_DIR / config_name) as f:
        config = json.load(f)

    config["input_model"]["model_path"] = model_name
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


def run_olive_pipelines(model_name, output_dir, quantize=False, quant_method="kquant8"):
    configs = QUANTIZED_CONFIGS.get(quant_method, {}) if quantize else {}
    for i, (config_name, subdir) in enumerate(COMPONENTS, 1):
        if subdir in configs:
            config_name = configs[subdir]
            label = _QUANT_LABELS[quant_method]
        else:
            label = "OnnxConversion, FP32"
        print(f"=== Stage 1.{i}: Olive {subdir} ({label}) ===")
        _run_olive_pipeline(config_name, model_name, output_dir, subdir)
        print()


def _derive_params(model_name):
    """Read every architecture value the config files need from the Torch model."""
    from cpu.moonshine_model_load import load_full_model, model_dims

    model = load_full_model(model_name)
    d = model_dims(model)
    enc = model.model.encoder
    ec = model.config.encoder_config
    emb = enc.embedder

    stride = int(emb.conv1.stride[0]) * int(emb.conv2.stride[0])  # subsampling = 4
    sample_rate = int(ec.sample_rate)
    frame_len = d["frame_len"]
    sliding_windows = [list(w) for w in ec.sliding_windows]

    return {
        **d,
        "sample_rate": sample_rate,
        "conv1_out": int(emb.conv1.out_channels),   # 1240 (small)
        "conv2_out": int(emb.conv2.out_channels),   # 620  (== encoder_dim)
        "num_heads": int(ec.num_attention_heads),
        "bos_token_id": int(model.config.bos_token_id),
        "eos_token_id": int(model.config.eos_token_id),
        "pad_token_id": int(model.config.pad_token_id),
        "decoder_start_token_id": int(model.config.decoder_start_token_id),
        # subsampling maps one memory frame back to `stride` input frames
        "seconds_per_memory_frame": frame_len / sample_rate * stride,   # 0.02
        # total future frames the encoder depends on = sum of per-layer right ctx
        "total_lookahead": int(sum(int(w[1]) for w in sliding_windows)),  # 16
    }


def generate_configs(model_name, output_dir):
    """Write genai_config.json and streaming_config.json from derived params."""
    print("=== Stage 2: Generating config files ===")
    p = _derive_params(model_name)
    dst = _resolve(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    genai_config = {
        "model": {
            "type": "streaming_enc_dec_asr",
            "bos_token_id": p["bos_token_id"],
            "eos_token_id": p["eos_token_id"],
            "pad_token_id": p["pad_token_id"],
            "decoder_start_token_id": p["decoder_start_token_id"],
            "vocab_size": p["vocab_size"],
            "sample_rate": p["sample_rate"],
            "chunk_samples": CHUNK_SAMPLES,
            "encoder": {
                "hidden_size": p["encoder_dim"],
                "num_attention_heads": p["num_heads"],
                "num_hidden_layers": p["num_encoder_layers"],
                "head_size": p["head_dim"],
            },
            "decoder": {
                "hidden_size": p["decoder_dim"],
                "num_attention_heads": p["num_heads"],
                "num_hidden_layers": p["num_decoder_layers"],
                "head_size": p["head_dim"],
            },
            "vad": dict(VAD),
            "moonshine": {
                "frontend_filename": "frontend.onnx",
                "encoder_filename": "encoder.onnx",
                "adapter_filename": "adapter.onnx",
                "cross_kv_filename": "cross_kv.onnx",
                "decoder_kv_filename": "decoder_kv.onnx",
                "sample_buffer_size": p["sample_buffer_size"],
                "conv1_buffer_size": p["left_pad1"],
                "conv2_buffer_size": p["left_pad2"],
                "total_lookahead": p["total_lookahead"],
                "max_seq_len": PIPELINE["max_seq_len"],
                "tokens_per_second": PIPELINE["tokens_per_second"],
                "seconds_per_memory_frame": p["seconds_per_memory_frame"],
                "max_segment_memory_frames": PIPELINE["max_segment_memory_frames"],
                "min_segment_memory_frames": PIPELINE["min_segment_memory_frames"],
                "left_context_frames": PIPELINE["left_context_frames"],
            },
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": False,
            "max_length": PIPELINE["max_seq_len"],
            "min_length": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    with open(dst / "genai_config.json", "w") as f:
        json.dump(genai_config, f, indent=2)
    print("  [OK] genai_config.json")

    streaming_config = {
        "encoder_dim": p["encoder_dim"],
        "decoder_dim": p["decoder_dim"],
        "depth": p["num_encoder_layers"],
        "nheads": p["num_heads"],
        "head_dim": p["head_dim"],
        "vocab_size": p["vocab_size"],
        "bos_id": p["bos_token_id"],
        "eos_id": p["eos_token_id"],
        "frame_len": p["frame_len"],
        "total_lookahead": p["total_lookahead"],
        "d_model_frontend": p["encoder_dim"],
        "c1": p["conv1_out"],
        "c2": p["conv2_out"],
        "frontend_state_shapes": {
            "sample_buffer": [1, p["sample_buffer_size"]],
            "sample_len": [1],
            "conv1_buffer": [1, p["conv1_channels"], p["left_pad1"]],
            "conv2_buffer": [1, p["conv2_channels"], p["left_pad2"]],
            "frame_count": [1],
        },
    }
    with open(dst / "streaming_config.json", "w") as f:
        json.dump(streaming_config, f, indent=2)
    print("  [OK] streaming_config.json")
    print()


def export_tokenizer(model_name, output_dir):
    """Fetch tokenizer.json from the HF repo and write the genai tokenizer_config."""
    from huggingface_hub import hf_hub_download

    print("=== Stage 3: Exporting tokenizer ===")
    dst = _resolve(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(model_name, "tokenizer.json")
    shutil.copy2(cached, dst / "tokenizer.json")
    with open(dst / "tokenizer_config.json", "w") as f:
        json.dump(TOKENIZER_CONFIG, f, indent=2)
    print(f"  [OK] tokenizer.json + tokenizer_config.json")
    print()


def download_silero_vad(output_dir):
    """Download the Silero VAD ONNX model from onnx-community/silero-vad."""
    from huggingface_hub import hf_hub_download

    print("=== Stage 4: Downloading Silero VAD ===")
    dst = _resolve(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id="onnx-community/silero-vad", filename="onnx/model.onnx")
    shutil.copy2(cached, dst / "silero_vad.onnx")
    size_mb = (dst / "silero_vad.onnx").stat().st_size / (1024 * 1024)
    print(f"  [OK] silero_vad.onnx ({size_mb:.1f} MB)")
    print()


def cleanup(output_dir):
    """Remove Olive's per-run model_config.json (not used by genai)."""
    stray = _resolve(output_dir) / "model_config.json"
    if stray.exists():
        stray.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Export & optimize Moonshine Streaming ASR for CPU (onnxruntime-genai)."
    )
    parser.add_argument("--model-name", default=MODEL_NAME,
                        help="HuggingFace model id (small or tiny).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output model directory (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--skip-vad", action="store_true",
                        help="Skip downloading Silero VAD (VAD is off by default at runtime).")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize the encoder + decoder_kv MatMuls "
                             "(frontend/adapter/cross_kv stay FP32).")
    parser.add_argument("--quant-method", choices=["kquant8", "kquant8-enc"], default="kquant8",
                        help="Algorithm used when --quantize is set: "
                             "'kquant8' = INT8 weight-only k-quant (MatMulNBits, bits=8); "
                             "'kquant8-enc' = INT8 k-quant on the encoder only (decoder_kv stays FP32).")
    args = parser.parse_args()

    # This recipe is architecture-locked to the usefulsensors/moonshine-streaming
    # family (tiny / small). Other checkpoints may have different frontend
    # buffer sizes, encoder/decoder dims, or tokenizer configs and won't
    # produce a runnable genai model. Warn but don't hard-fail so local
    # forks can still opt in.
    if not args.model_name.startswith("usefulsensors/moonshine-streaming-"):
        print(
            f"  Warning: --model-name '{args.model_name}' is outside the "
            f"'usefulsensors/moonshine-streaming-*' family this recipe was "
            f"built for; export may fail or produce an unusable model."
        )

    run_olive_pipelines(args.model_name, args.output_dir,
                        quantize=args.quantize, quant_method=args.quant_method)
    generate_configs(args.model_name, args.output_dir)
    export_tokenizer(args.model_name, args.output_dir)
    if not args.skip_vad:
        vad_dest = _resolve(args.output_dir) / "silero_vad.onnx"
        try:
            download_silero_vad(args.output_dir)
        except Exception as exc:
            print(f"  Warning: Silero VAD download failed ({exc}).\n"
                  f"  Download manually from https://huggingface.co/onnx-community/silero-vad\n"
                  f"  and place silero_vad.onnx at: {vad_dest}")
    cleanup(args.output_dir)

    out = _resolve(args.output_dir)
    files = sorted(f for f in out.iterdir() if f.is_file())
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print(f"=== Done! Model -> {out} ===")
    print(f"    Total size: {total_mb:.1f} MB")
    for f in files:
        print(f"    {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
