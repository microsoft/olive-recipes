# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path

import onnx
from transformers import WhisperConfig


def run_olive_workflow(config_path: Path):
    from olive.workflows import run

    run(config_path)


def _ensure_decoder_outputs(model: onnx.ModelProto, num_layers: int):
    """Ensure decoder outputs ONLY decoder present KVs (NOT encoder).

    - Outputs: logits + decoder present KVs only
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


def postprocess_decoder_model(models_dir: Path, config: WhisperConfig):
    """Post-process decoder model to ensure correct outputs."""
    num_layers = config.decoder_layers

    decoder_path = models_dir / "decoder" / "model.onnx"
    if decoder_path.exists():
        print(f"  Post-processing {decoder_path}...")
        decoder_model = onnx.load(decoder_path.as_posix(), load_external_data=False)
        _ensure_decoder_outputs(decoder_model, num_layers)
        onnx.save(decoder_model, decoder_path.as_posix(), convert_attribute=True)
        print(f"  ✓ Saved {decoder_path}")
    else:
        print(f"  ⚠ Decoder model not found at {decoder_path}")


def add_encoder_past_inputs_to_decoder(model_path: Path, config: WhisperConfig):
    """Add encoder cross-attention KV cache inputs to decoder model if missing."""
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


def optimize_whisper_model(models_dir: Path):
    """Run Olive workflows for all Whisper components."""
    from user_script import post_process_encoder_decoder_init, post_process_decoder

    # Run encoder_decoder_init (which combines encoder + cross-attention cache init)
    print("\n[1/2] Running encoder_decoder_init workflow...")
    run_olive_workflow("encoder_decoder_init.json")

    # Post-process encoder_decoder_init model
    encoder_decoder_init_path = models_dir / "encoder_decoder_init" / "model.onnx"
    if encoder_decoder_init_path.exists():
        print("  Post-processing encoder_decoder_init model...")
        post_process_encoder_decoder_init(str(encoder_decoder_init_path))

    # Run decoder (with past key values)
    print("\n[2/2] Running decoder workflow...")
    run_olive_workflow("decoder.json")

    # Post-process decoder model
    decoder_path = models_dir / "decoder" / "model.onnx"
    if decoder_path.exists():
        print("  Post-processing decoder model...")
        post_process_decoder(str(decoder_path))


def main():
    model_id = "openai/whisper-large-v3-turbo"
    base = Path(__file__).parent
    models_dir = base / "models"

    # Run Olive optimization workflows
    optimize_whisper_model(models_dir)

    # Post-processing steps
    cfg = WhisperConfig.from_pretrained(model_id)

    print("\n[Post-processing] Ensuring correct decoder outputs...")
    postprocess_decoder_model(models_dir, cfg)

    decoder_path = models_dir / "decoder" / "model.onnx"
    if decoder_path.exists():
        print("\n[Post-processing] Adding encoder inputs to decoder...")
        add_encoder_past_inputs_to_decoder(decoder_path, cfg)

    # Cleanup temporary folders
    print("\n[Cleanup] Removing temporary folders...")
    encoder_decoder_init_dir = models_dir / "encoder_decoder_init"
    decoder_dir = models_dir / "decoder"

    if encoder_decoder_init_dir.exists():
        shutil.rmtree(encoder_decoder_init_dir)
        print(f"  ✓ Removed {encoder_decoder_init_dir}")

    if decoder_dir.exists():
        shutil.rmtree(decoder_dir)
        print(f"  ✓ Removed {decoder_dir}")


if __name__ == "__main__":
    main()
