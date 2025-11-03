# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

import tempfile

import onnx
from optimum.onnx import merge_decoders
from optimum.exporters.onnx.model_configs import WhisperOnnxConfig
from optimum.exporters.onnx.base import ConfigBehavior
from transformers import WhisperConfig


def run_olive_workflow(config_path: Path):
    from olive.workflows import run

    with open(config_path, "r") as f:
        cfg = json.load(f)

    run(config_path)


def _ensure_decoder_with_past_outputs(model: onnx.ModelProto, num_layers: int):
    """Ensure decoder_with_past outputs ONLY decoder present KVs (NOT encoder).

    - Outputs: logits + decoder present KVs only (65 total)
    - No encoder present KVs (encoder cache is constant)
    """
    desired_outputs = ["logits"]
    for i in range(num_layers):
        desired_outputs.extend(
            [
                f"present.{i}.decoder.key",
                f"present.{i}.decoder.value",
            ]
        )

    outputs_map = {out.name: out for out in model.graph.output}
    model.graph.ClearField("output")
    for name in desired_outputs:
        if name in outputs_map:
            model.graph.output.append(outputs_map[name])


def merge_decoder_outputs(root_dir: Path):
    decoder_path = root_dir / "decoder" / "model.onnx"
    decoder_with_past_path = root_dir / "decoder_with_past" / "model.onnx"
    merged_path = root_dir / "decoder_merged" / "model.onnx"

    if not decoder_path.exists() or not decoder_with_past_path.exists():
        raise FileNotFoundError("Decoder ONNX models not found.")

    merged_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_merged = Path(temp_dir) / "merged.onnx"

        merge_decoders(
            decoder=decoder_path.as_posix(),
            decoder_with_past=decoder_with_past_path.as_posix(),
            save_path=temp_merged.as_posix(),
            strict=False,
        )

        # Load merged model and fix ir_version if needed
        merged_model = onnx.load(temp_merged.as_posix())

        if not merged_model.ir_version or merged_model.ir_version == 0:
            base_model = onnx.load(decoder_path.as_posix(), load_external_data=False)
            merged_model.ir_version = (
                base_model.ir_version if base_model.ir_version else onnx.IR_VERSION
            )

        onnx.save_model(
            merged_model,
            merged_path.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=merged_path.name + "_data",
            convert_attribute=False,
        )


def fix_dynamic_axes(models_dir: Path, model_id: str):
    cfg = WhisperConfig.from_pretrained(model_id)

    encoder_path = models_dir / "encoder" / "model.onnx"
    decoder_path = models_dir / "decoder" / "model.onnx"
    decoder_with_past_path = models_dir / "decoder_with_past" / "model.onnx"

    if encoder_path.exists():
        encoder_cfg = WhisperOnnxConfig(
            cfg,
            task="automatic-speech-recognition",
            behavior=ConfigBehavior.ENCODER,
        )
        encoder_cfg.fix_dynamic_axes(encoder_path)

    if decoder_path.exists():
        decoder_cfg = WhisperOnnxConfig(
            cfg,
            task="automatic-speech-recognition",
            behavior=ConfigBehavior.DECODER,
            use_past=False,
        )
        decoder_cfg.fix_dynamic_axes(decoder_path)

    if decoder_with_past_path.exists():
        decoder_past_cfg = WhisperOnnxConfig(
            cfg,
            task="automatic-speech-recognition",
            behavior=ConfigBehavior.DECODER,
            use_past=True,
            use_past_in_inputs=True,
        )
        decoder_past_cfg.fix_dynamic_axes(decoder_with_past_path)


def postprocess_decoder_models(models_dir: Path, config: WhisperConfig):
    num_layers = config.decoder_layers

    decoder_path = models_dir / "decoder" / "model.onnx"
    if decoder_path.exists():
        decoder_model = onnx.load(decoder_path.as_posix(), load_external_data=False)
        desired_outputs = ["logits"]
        for i in range(num_layers):
            desired_outputs.extend(
                [
                    f"present.{i}.decoder.key",
                    f"present.{i}.decoder.value",
                    f"present.{i}.encoder.key",
                    f"present.{i}.encoder.value",
                ]
            )
        outputs_map = {out.name: out for out in decoder_model.graph.output}
        decoder_model.graph.ClearField("output")
        for name in desired_outputs:
            if name in outputs_map:
                decoder_model.graph.output.append(outputs_map[name])
        onnx.save(decoder_model, decoder_path.as_posix(), convert_attribute=True)

    decoder_with_past_path = models_dir / "decoder_with_past" / "model.onnx"
    if decoder_with_past_path.exists():
        decoder_with_past_model = onnx.load(
            decoder_with_past_path.as_posix(), load_external_data=False
        )
        _ensure_decoder_with_past_outputs(decoder_with_past_model, num_layers)
        onnx.save(
            decoder_with_past_model,
            decoder_with_past_path.as_posix(),
            convert_attribute=True,
        )


def add_encoder_past_inputs_to_decoder_with_past(
    model_path: Path, config: WhisperConfig
):
    model = onnx.load(model_path.as_posix(), load_external_data=False)
    num_layers = config.decoder_layers

    # Check if encoder inputs are missing
    existing_inputs = {inp.name for inp in model.graph.input}
    missing_encoder_inputs = []

    for i in range(num_layers):
        enc_key = f"past_key_values.{i}.encoder.key"
        enc_val = f"past_key_values.{i}.encoder.value"
        if enc_key not in existing_inputs:
            missing_encoder_inputs.append((i, "key", enc_key))
        if enc_val not in existing_inputs:
            missing_encoder_inputs.append((i, "value", enc_val))

    if not missing_encoder_inputs:
        print("  ✓ All encoder past inputs already present")
        return

    print(f"  Adding {len(missing_encoder_inputs)} missing encoder past inputs...")

    # Get reference shape from decoder inputs
    num_heads = config.decoder_attention_heads
    head_dim = config.d_model // num_heads

    # Add missing inputs to the graph
    for layer_idx, kv_type, input_name in missing_encoder_inputs:
        # Create input tensor info
        input_tensor = onnx.helper.make_tensor_value_info(
            input_name,
            onnx.TensorProto.FLOAT,
            ["batch_size", num_heads, "encoder_sequence_length", head_dim],
        )
        model.graph.input.append(input_tensor)

    onnx.save(model, model_path.as_posix())
    print(f"  ✓ Added {len(missing_encoder_inputs)} encoder past inputs")


def optimize_whisper_model():
    run_olive_workflow("encoder.json")
    run_olive_workflow("decoder.json")
    run_olive_workflow("decoder_with_past.json")


def main():
    model_id = "openai/whisper-large-v3"
    base = Path(__file__).parent
    models_dir = base / "models"

    parser = argparse.ArgumentParser("Whisper arguments")
    parser.add_argument(
        "--merge_decoder", action="store_true", help="Merge decoder models"
    )

    args = parser.parse_args()
    optimize_whisper_model()

    if args.merge_decoder:
        cfg = WhisperConfig.from_pretrained(model_id)

        print("\n[1/3] Post-processing decoder models...")
        postprocess_decoder_models(models_dir, cfg)

        print("\n[2/3] Fixing dynamic axes...")
        fix_dynamic_axes(models_dir, model_id)

        print("\n[3/3] Merging decoders...")
        merge_decoder_outputs(models_dir)


if __name__ == "__main__":
    main()
