"""End-to-end optimization pipeline for Nemotron Speech Streaming.

All model components (encoder, decoder, joint) are exported and optimized
using Olive's declarative pass system:

  - Encoder: OnnxConversion → OnnxKQuantQuantization for CPU, or FP16
    OnnxConversion with native Attention for NvTensorRtRtx
  - Decoder: FP16 OnnxConversion for NvTensorRtRtx, FP32 for CPU
  - Joint:   FP16 OnnxConversion for NvTensorRtRtx, FP32 for CPU

After the Olive pipelines, tokenizer and config files are generated. Silero
VAD is downloaded for CPU and omitted for NvTensorRtRtx.

Usage:
    # Full pipeline
    python src/optimize.py

    # Or use Olive CLI directly for individual components:
    python -m olive run --config src/nemotron_encoder_int4_cpu.json
    python -m olive run --config src/nemotron_decoder_fp32_cpu.json
    python -m olive run --config src/nemotron_joint_fp32_cpu.json
"""

import argparse
from dataclasses import dataclass
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure the recipe root is on sys.path so `from src.nemotron_model_load import ...` works
# regardless of where the script is invoked from.
_SCRIPT_DIR = Path(__file__).resolve().parent
_RECIPE_ROOT = _SCRIPT_DIR.parent
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

_TOKENIZER_SCRIPT = _RECIPE_ROOT / "scripts" / "export_tokenizer.py"

DEFAULT_OUTPUT_DIR = "build/onnx_models_int4"
DEFAULT_TRT_RTX_OUTPUT_DIR = "build/onnx_models_trtrtx_fp16"
TRT_RTX_ALIASES = {
    "nvtensorrtrtx",
    "nvtensorrtx",
    "trt-rtx",
    "trtrtx",
    "nvtensorrtxexecutionprovider",
    "nvtensorrtrtxexecutionprovider",
}


@dataclass(frozen=True)
class EncoderExportPlan:
    config_name: str
    encoder_precision: str


def _is_trt_rtx_execution_provider(execution_provider: str) -> bool:
    normalized = execution_provider.replace("_", "").replace("-", "").lower()
    return normalized in TRT_RTX_ALIASES


def resolve_encoder_export_plan(execution_provider: str, encoder_precision: str) -> EncoderExportPlan:
    """Select the encoder Olive config for the requested backend."""
    if _is_trt_rtx_execution_provider(execution_provider):
        return EncoderExportPlan("nemotron_encoder_fp16_trtrtx.json", "fp16")
    if encoder_precision == "fp16":
        raise ValueError("encoder_precision=fp16 is only supported with NvTensorRtRtx.")
    if encoder_precision == "int8":
        return EncoderExportPlan("nemotron_encoder_int8_cpu.json", "int8")
    return EncoderExportPlan("nemotron_encoder_int4_cpu.json", "int4")


def _decoder_config_name(execution_provider: str) -> str:
    return "nemotron_decoder_fp16_trtrtx.json" if _is_trt_rtx_execution_provider(execution_provider) else "nemotron_decoder_fp32_cpu.json"


def _joint_config_name(execution_provider: str) -> str:
    return "nemotron_joint_fp16_trtrtx.json" if _is_trt_rtx_execution_provider(execution_provider) else "nemotron_joint_fp32_cpu.json"


def should_include_vad(execution_provider: str) -> bool:
    return not _is_trt_rtx_execution_provider(execution_provider)


def _resolve(path: str) -> Path:
    """Resolve a path relative to the src/ directory."""
    p = Path(path)
    return p if p.is_absolute() else _SCRIPT_DIR / p


def _run_olive_pipeline(config_name: str, output_dir: str, output_subdir: str, model_path: str = None):
    """Run an Olive pipeline from a JSON config, overriding output_dir."""
    from olive import run as olive_run

    config_path = _SCRIPT_DIR / config_name
    with open(config_path) as f:
        config = json.load(f)

    config["output_dir"] = str(_resolve(output_dir) / output_subdir)
    if model_path is not None:
        config["input_model"]["model_path"] = model_path

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(_SCRIPT_DIR), delete=False
    ) as tmp:
        json.dump(config, tmp, indent=4)
        tmp_path = tmp.name

    try:
        olive_run(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_olive_pipelines(output_dir: str, model_path: str = None, encoder_precision: str = "int4", execution_provider: str = "cpu"):
    """Run all Olive pipelines for the selected backend."""
    encoder_plan = resolve_encoder_export_plan(execution_provider, encoder_precision)
    if encoder_plan.encoder_precision != encoder_precision:
        print(f"=== Stage 1: Olive Encoder (requested {encoder_precision}, using FP16 for {execution_provider}) ===")
    elif encoder_plan.encoder_precision == "fp16":
        print("=== Stage 1: Olive Encoder (OnnxConversion → FP16, opset 23) ===")
    elif encoder_plan.encoder_precision == "int8":
        print("=== Stage 1: Olive Encoder (OnnxConversion → INT8 k-quant) ===")
    else:
        print("=== Stage 1: Olive Encoder (OnnxConversion → INT4 quant) ===")
    _run_olive_pipeline(encoder_plan.config_name, output_dir, "encoder.onnx", model_path)
    print()

    component_precision = "FP16, opset 23" if _is_trt_rtx_execution_provider(execution_provider) else "FP32"
    print(f"=== Stage 2: Olive Decoder (OnnxConversion, {component_precision}) ===")
    _run_olive_pipeline(_decoder_config_name(execution_provider), output_dir, "decoder.onnx", model_path)
    print()

    print(f"=== Stage 3: Olive Joint (OnnxConversion, {component_precision}) ===")
    _run_olive_pipeline(_joint_config_name(execution_provider), output_dir, "joint.onnx", model_path)
    print()


def run_tokenizer_export(model_name: str, output_dir: str):
    """Export tokenizer files to the output directory."""
    print("=== Stage 4: Exporting tokenizer ===")
    cmd = [
        sys.executable,
        str(_TOKENIZER_SCRIPT),
        "--model_name", model_name,
        "--output_dir", str(_resolve(output_dir)),
    ]
    result = subprocess.run(cmd, cwd=str(_SCRIPT_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"Tokenizer export failed (exit code {result.returncode})")
    print()


def generate_configs(model_name: str, output_dir: str, chunk_size: float, include_vad: bool = True):
    """Generate genai_config.json and audio_processor_config.json.

    Loads the NeMo model to extract architecture parameters, then writes
    the config files needed by onnxruntime-genai for inference.
    """
    print("=== Stage 5: Generating config files ===")
    from src.nemotron_model_load import _load_nemo_model, get_att_context_size, D_MODEL, N_LAYERS, DECODER_HIDDEN, DECODER_LSTM_LAYERS

    asr_model = _load_nemo_model(model_name)
    asr_model.eval()

    dst = _resolve(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    encoder = asr_model.encoder
    joint = asr_model.joint

    vocab_size = joint.num_classes_with_blank
    blank_id = vocab_size - 1

    preprocessor_cfg = asr_model.cfg.get('preprocessor', {})
    sample_rate = preprocessor_cfg.get('sample_rate', 16000)
    n_mels = preprocessor_cfg.get('features', preprocessor_cfg.get('nfilt', 128))
    n_fft = preprocessor_cfg.get('n_fft', 512)
    hop_length = preprocessor_cfg.get('hop_length', 160)
    win_length = preprocessor_cfg.get('win_length', 400)
    preemph = preprocessor_cfg.get('preemph', 0.97)

    subsampling_factor = getattr(encoder, 'subsampling_factor', 8)
    att_context_size = get_att_context_size(chunk_size)
    left_context = att_context_size[0]

    conv_context = 8
    if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
        layer = encoder.layers[0]
        if hasattr(layer, 'conv') and hasattr(layer.conv, 'conv'):
            conv = layer.conv.conv
            if hasattr(conv, 'kernel_size'):
                ks = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
                conv_context = ks - 1

    pre_encode_cache_size = getattr(encoder, 'pre_encode_cache_size', 9)
    if isinstance(pre_encode_cache_size, (list, tuple)):
        pre_encode_cache_size = pre_encode_cache_size[-1]

    chunk_samples = int(chunk_size * sample_rate)
    max_symbols = asr_model.cfg.get('decoding', {}).get('greedy', {}).get('max_symbols', 10)

    genai_config = {
        "model": {
            "type": "nemotron_speech",
            "vocab_size": vocab_size,
            "num_mels": n_mels,
            "fft_size": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "preemph": preemph,
            "log_eps": 5.96046448e-08,
            "subsampling_factor": subsampling_factor,
            "left_context": left_context,
            "conv_context": conv_context,
            "pre_encode_cache_size": pre_encode_cache_size,
            "sample_rate": sample_rate,
            "chunk_samples": chunk_samples,
            "blank_id": blank_id,
            "max_symbols_per_step": max_symbols,
            "encoder": {
                "filename": "encoder.onnx",
                "hidden_size": D_MODEL,
                "num_hidden_layers": N_LAYERS,
                "inputs": {
                    "audio_features": "audio_signal",
                    "input_lengths": "length",
                    "cache_last_channel": "cache_last_channel",
                    "cache_last_time": "cache_last_time",
                    "cache_last_channel_len": "cache_last_channel_len",
                    "lang_id": "lang_id",
                },
                "outputs": {
                    "encoder_outputs": "outputs",
                    "output_lengths": "encoded_lengths",
                    "cache_last_channel_next": "cache_last_channel_next",
                    "cache_last_time_next": "cache_last_time_next",
                    "cache_last_channel_len_next": "cache_last_channel_len_next",
                },
            },
            "decoder": {
                "filename": "decoder.onnx",
                "hidden_size": DECODER_HIDDEN,
                "num_hidden_layers": DECODER_LSTM_LAYERS,
                "inputs": {
                    "targets": "targets",
                    "lstm_hidden_state": "h_in",
                    "lstm_cell_state": "c_in",
                },
                "outputs": {
                    "outputs": "decoder_output",
                    "lstm_hidden_state": "h_out",
                    "lstm_cell_state": "c_out",
                },
            },
            "joiner": {
                "filename": "joint.onnx",
                "inputs": {
                    "encoder_outputs": "encoder_output",
                    "decoder_outputs": "decoder_output",
                },
                "outputs": {
                    "logits": "joint_output",
                },
            },
        },
    }
    if include_vad:
        genai_config["model"]["vad"] = {
            "filename": "silero_vad.onnx",
            "threshold": 0.3,
            "silence_duration_ms": 3360,
            "prefix_padding_ms": 560,
        }

    with open(dst / "genai_config.json", "w") as f:
        json.dump(genai_config, f, indent=2)
    print(f"  [OK] genai_config.json")

    # Audio processor config
    window_size = preprocessor_cfg.get('window_size', preprocessor_cfg.get('n_window_size', 0.025))
    window_stride = preprocessor_cfg.get('window_stride', preprocessor_cfg.get('n_window_stride', 0.01))
    if isinstance(window_size, float) and window_size < 1.0:
        window_length_samples = int(window_size * sample_rate)
    elif isinstance(window_size, int):
        window_length_samples = window_size
    else:
        window_length_samples = 400
    if isinstance(window_stride, float) and window_stride < 1.0:
        hop_length_samples = int(window_stride * sample_rate)
    elif isinstance(window_stride, int):
        hop_length_samples = window_stride
    else:
        hop_length_samples = 160

    audio_config = {
        "model_type": "speech_features",
        "audio_params": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length_samples,
            "n_mels": n_mels,
            "window_length": window_length_samples,
            "window_type": "hann",
            "fmin": 0,
            "fmax": sample_rate // 2,
            "dither": preprocessor_cfg.get('dither', 0.0),
            "preemphasis": preemph,
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 1e-10,
            "normalize": preprocessor_cfg.get('normalize', 'none'),
            "center": True,
            "mag_power": 2.0,
        },
    }

    with open(dst / "audio_processor_config.json", "w") as f:
        json.dump(audio_config, f, indent=2)
    print(f"  [OK] audio_processor_config.json")
    print()


def download_silero_vad(output_dir: str):
    """Download the Silero VAD ONNX model from onnx-community/silero-vad."""
    from huggingface_hub import hf_hub_download

    print("=== Stage 6: Downloading Silero VAD ===")
    dst_dir = _resolve(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "silero_vad.onnx"

    cached = hf_hub_download(
        repo_id="onnx-community/silero-vad",
        filename="onnx/model.onnx",
        revision="e71cae966052b992a7eca6b17738916ce0eca4ec",
    )
    shutil.copy2(cached, str(dst))
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  Saved Silero VAD model to: {dst} ({size_mb:.1f} MB)")
    print()


def main():
    from src.nemotron_model_load import MODEL_NAME, CHUNK_SIZE

    parser = argparse.ArgumentParser(
        description="Optimize Nemotron Speech Streaming for CPU or NvTensorRtRtx inference"
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="HuggingFace model name or path to a local .nemo file",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for optimized models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--encoder-precision",
        choices=["int4", "int8", "fp16"],
        default="int4",
        help="Encoder precision: int4/int8 for CPU, fp16 for NvTensorRtRtx. Default: int4.",
    )
    parser.add_argument(
        "--execution-provider",
        choices=[
            "cpu",
            "cuda",
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "NvTensorRtRtx",
            "NvTensorRTRTXExecutionProvider",
        ],
        default="cpu",
        help="Target execution provider. NvTensorRtRtx forces FP16 export with opset 23.",
    )
    args = parser.parse_args()

    if _is_trt_rtx_execution_provider(args.execution_provider) and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_TRT_RTX_OUTPUT_DIR

    # Validate model name — the Olive configs and model_load.py constants are
    # specific to the 0.6B model architecture.
    if not args.model_name.endswith(".nemo") and args.model_name != MODEL_NAME:
        raise ValueError(
            f"This recipe only supports '{MODEL_NAME}' (or a .nemo file with the same architecture). "
            f"Got: '{args.model_name}'"
        )

    # Stages 1-3: Run Olive pipelines for encoder, decoder, joint
    run_olive_pipelines(
        output_dir=args.output_dir,
        model_path=args.model_name,
        encoder_precision=args.encoder_precision,
        execution_provider=args.execution_provider,
    )

    # Stage 4: Export tokenizer
    run_tokenizer_export(model_name=args.model_name, output_dir=args.output_dir)

    # Stage 5: Generate config files (chunk_size matches the hardcoded export shapes)
    include_vad = should_include_vad(args.execution_provider)
    generate_configs(
        model_name=args.model_name,
        output_dir=args.output_dir,
        chunk_size=CHUNK_SIZE,
        include_vad=include_vad,
    )

    # Stage 6: Download Silero VAD
    if include_vad:
        vad_dest = _resolve(args.output_dir) / "silero_vad.onnx"
        try:
            download_silero_vad(output_dir=args.output_dir)
        except Exception as exc:
            print(
                f"  Warning: Silero VAD download failed ({exc}).\n"
                f"  Download manually from https://huggingface.co/onnx-community/silero-vad\n"
                f"  and place silero_vad.onnx at: {vad_dest}"
            )
    else:
        print("=== Stage 6: Skipping Silero VAD for NvTensorRtRtx ===")
        print("  VAD is omitted from genai_config.json for TRT-RTX compatibility.")
        print()

    # Summary
    output_path = _resolve(args.output_dir)
    if output_path.exists():
        files = sorted(f for f in output_path.iterdir() if f.is_file())
        total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"=== Done! Optimized models → {output_path} ===")
        print(f"    Total size: {total_mb:.1f} MB")
        resolved_precision = resolve_encoder_export_plan(args.execution_provider, args.encoder_precision).encoder_precision
        enc_label = {
            "int4": "INT4 k-quant (Olive)",
            "int8": "INT8 dynamic (Olive)",
            "fp16": "FP16 (Olive, opset 23)",
        }.get(resolved_precision, "")
        for f in files:
            tag = f" ← {enc_label}" if f.name.startswith("encoder") and enc_label else ""
            print(f"    {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB){tag}")


if __name__ == "__main__":
    main()
