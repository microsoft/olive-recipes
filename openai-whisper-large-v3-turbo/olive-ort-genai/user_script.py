# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from itertools import chain

import onnx
import torch
from transformers import AutoModelForSpeechSeq2Seq, WhisperConfig


class WhisperEncoder(torch.nn.Module):
    """Whisper encoder component"""

    def __init__(self, model, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.encoder = model.model.encoder

    def forward(self, audio_features: torch.Tensor):
        outputs = self.encoder(audio_features)
        return outputs.last_hidden_state

    def input_names(self):
        """Get input names for encoder model"""
        return ["audio_features"]

    def output_names(self):
        """Get output names for encoder model"""
        return ["encoder_hidden_states"]

    def dynamic_axes(self, input_names, output_names):
        """Get dynamic axes for encoder model"""
        return get_model_dynamic_axes(self.config, input_names, output_names)


class WhisperEncoderDecoderInit(torch.nn.Module):
    """Whisper encoder component + first pass through Whisper decoder component to initialize KV caches"""

    def __init__(self, model, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.encoder = model.model.encoder
        self.decoder = model.model.decoder

        self.max_source_positions = self.config.max_source_positions
        self.num_heads = self.config.decoder_attention_heads
        self.head_size = self.config.d_model // self.num_heads

    def forward(self, audio_features: torch.Tensor):
        encoder_hidden_states = self.encoder(audio_features).last_hidden_state

        # Get cross attention KV caches and return them for this model
        # We do this because these MatMuls are only run once before their outputs are being re-used in the decoder
        present_cross_attention_key_value_caches = []
        for layer in self.decoder.layers:
            cross_attn_key_cache = (
                layer.encoder_attn.k_proj(encoder_hidden_states)
                .view(-1, self.max_source_positions, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            cross_attn_value_cache = (
                layer.encoder_attn.v_proj(encoder_hidden_states)
                .view(-1, self.max_source_positions, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            present_cross_attention_key_value_caches.append(cross_attn_key_cache)
            present_cross_attention_key_value_caches.append(cross_attn_value_cache)

        return encoder_hidden_states, present_cross_attention_key_value_caches

    def input_names(self):
        """Get input names for encoder_decoder_init model"""
        return ["audio_features"]

    def output_names(self):
        """Get output names for encoder_decoder_init model"""
        output_names = [
            "encoder_hidden_states",
            *list(
                chain.from_iterable(
                    (f"present_key_cross_{i}", f"present_value_cross_{i}")
                    for i in range(self.config.decoder_layers)
                )
            ),
        ]
        return output_names

    def dynamic_axes(self, input_names, output_names):
        """Get dynamic axes for encoder_decoder_init model"""
        return get_model_dynamic_axes(self.config, input_names, output_names)

    def fix_key_value_cache_dims(self, output, is_cross: bool = False):
        """Fix dimensions for key/value cache tensors.

        Shape should be (batch_size, num_heads, sequence_length, head_size) for self attention KV caches
        and (batch_size, num_heads, num_frames // 2, head_size) for cross attention KV caches
        """
        num_heads = output.type.tensor_type.shape.dim[1]
        if "_dim_" in num_heads.dim_param:
            num_heads.Clear()
            num_heads.dim_value = self.num_heads
        sequence_length = output.type.tensor_type.shape.dim[2]
        if "_dim_" in sequence_length.dim_param:
            sequence_length.Clear()
            if is_cross:
                sequence_length.dim_value = self.max_source_positions
            else:
                sequence_length.dim_param = "total_sequence_length"
        head_size = output.type.tensor_type.shape.dim[3]
        if "_dim_" in head_size.dim_param:
            head_size.Clear()
            head_size.dim_value = self.head_size
        return output

    def fix_outputs(self, model):
        """Fix output dimensions and reorder outputs.

        ONNX exporter might mark dimensions like 'Transposepresent_value_self_1_dim_2' in shape inference.
        We now change the dim_values to the correct one.
        """
        reordered_outputs = []
        self_attn_kv_caches = []
        cross_attn_kv_caches = []

        for output in model.graph.output:
            if "present" not in output.name:
                reordered_outputs.append(output)

            elif "self" in output.name:
                # Self attention KV caches
                new_output = self.fix_key_value_cache_dims(output, is_cross=False)
                # For encoder_decoder_init without beam search, we only have cross attention
                # So self attention caches should not exist
                self_attn_kv_caches.append(new_output)
            else:
                # Cross attention KV caches
                new_output = self.fix_key_value_cache_dims(output, is_cross=True)
                reordered_outputs.append(new_output)

        # For encoder_decoder_init, we typically only have cross attention caches
        # But just in case, handle self attention caches too
        reordered_outputs += self_attn_kv_caches + cross_attn_kv_caches

        while len(model.graph.output) > 0:
            model.graph.output.pop()
        model.graph.output.extend(reordered_outputs)
        return model


class WhisperDecoder(torch.nn.Module):
    """Whisper decoder with past key values"""

    def __init__(self, model, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *past_key_values,
    ):
        # Restructure flattened past_key_values into list of tuples
        # Input: past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0, ...
        # Output: [(past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0), ...]
        # Each layer has 4 caches: self_key, self_value, cross_key, cross_value
        num_layers = len(past_key_values) // 4
        past_kv_list = []
        for i in range(num_layers):
            layer_past = (
                past_key_values[i * 4],      # past_key_self
                past_key_values[i * 4 + 1],  # past_value_self
                past_key_values[i * 4 + 2],  # past_key_cross
                past_key_values[i * 4 + 3],  # past_value_cross
            )
            past_kv_list.append(layer_past)

        outputs = self.decoder(
            encoder_hidden_states=encoder_hidden_states,
            input_ids=decoder_input_ids,
            past_key_values=past_kv_list,
            use_cache=True,
        )
        logits = self.proj_out(outputs.last_hidden_state)
        present_key_values = outputs.past_key_values

        # Before: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0),
        #         (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1),
        # After:  (past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1), ...
        present_self = []
        for layer_past in present_key_values:
            present_self.append(layer_past[0])  # key
            present_self.append(layer_past[1])  # value

        # Return present_self_* for decoder-with-past since past_cross_* and present_cross_* are identical
        return logits, present_self

    def input_names(self):
        """Get input names for decoder model (decoder with past)"""
        input_names = ["input_ids", "encoder_hidden_states"]
        # Add past key values for all layers
        input_names.extend(
            list(
                chain.from_iterable(
                    (f"past_key_self_{i}", f"past_value_self_{i}", f"past_key_cross_{i}", f"past_value_cross_{i}")
                    for i in range(self.config.decoder_layers)
                )
            )
        )
        return input_names

    def output_names(self):
        """Get output names for decoder model (decoder with past)"""
        output_names = ["logits"]
        # Add present self key values (cross KV caches are not outputs as they're unchanged)
        output_names.extend(
            list(
                chain.from_iterable(
                    (f"present_key_self_{i}", f"present_value_self_{i}")
                    for i in range(self.config.decoder_layers)
                )
            )
        )
        return output_names

    def dynamic_axes(self, input_names, output_names):
        """Get dynamic axes for decoder model"""
        return get_model_dynamic_axes(self.config, input_names, output_names)


# ============================================================================
# Dynamic Axes Helper Function
# ============================================================================


def get_model_dynamic_axes(
    config: WhisperConfig,
    input_names: list[str],
    output_names: list[str],
):
    """Generate dynamic_axes dict for ONNX export based on input/output names.

    This function maps each input/output name to its dynamic dimensions based on
    the tensor's shape and semantic meaning in the Whisper model.

    Args:
        config: WhisperConfig object
        input_names: List of input tensor names
        output_names: List of output tensor names

    Returns:
        Dictionary mapping tensor names to dynamic axes specifications
    """
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"audio_features", "encoder_input_ids"}:
            # shape is (batch_size, num_mels, num_frames)
            dynamic_axes[name] = {0: "batch_size"}
        elif name in {"input_ids", "decoder_input_ids"}:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "alignment_heads":
            # shape is (num_alignment_heads, 2)
            dynamic_axes[name] = {0: "num_alignment_heads"}
        elif name in {"sot_sequence_length", "segment_length"}:
            # shape is (1)
            pass
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "encoder_hidden_states":
            # shape is (batch_size, num_frames // 2, hidden_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "past_key_self" in name or "past_value_self" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif "present_key_self" in name or "present_value_self" in name:
            # shape is (batch_size, num_heads, past_sequence_length + sequence_length, head_size),
            # which is equal to (batch_size, num_heads, total_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}
        elif (
            "past_key_cross" in name
            or "past_value_cross" in name
            or "present_key_cross" in name
            or "present_value_cross" in name
        ):
            # shape is (batch_size, num_heads, num_frames // 2, head_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "cross_qk" in name:
            # shape is (batch_size, num_heads, source_sequence_length, target_sequence_length)
            dynamic_axes[name] = {0: "batch_size", 2: "sequence_length"}
        elif "jump_times" in name:
            # shape is (batch_size, max_length)
            dynamic_axes[name] = {0: "batch_size", 1: "max_length"}
        else:
            raise Exception(f"Unknown input or output name found: {name}")
    return dynamic_axes


# ============================================================================
# ONNX Model Fixing Functions
# ============================================================================


def fix_key_value_cache_dims(io, config: WhisperConfig, is_cross: bool = False, is_output: bool = False):
    """Fix dimensions for key/value cache tensors.

    Shape should be (batch_size, num_heads, sequence_length, head_size) for self attention KV caches
    and (batch_size, num_heads, num_frames // 2, head_size) for cross attention KV caches
    """
    num_heads = config.decoder_attention_heads
    head_size = config.d_model // num_heads
    max_source_positions = config.max_source_positions

    # Fix num_heads dimension
    num_heads_dim = io.type.tensor_type.shape.dim[1]
    if "_dim_" in num_heads_dim.dim_param:
        num_heads_dim.Clear()
        num_heads_dim.dim_value = num_heads

    # Fix sequence_length dimension
    sequence_length = io.type.tensor_type.shape.dim[2]
    if "_dim_" in sequence_length.dim_param:
        sequence_length.Clear()
        if is_cross:
            sequence_length.dim_value = max_source_positions
        else:
            sequence_length.dim_param = "total_sequence_length" if is_output else "past_sequence_length"

    # Fix head_size dimension
    head_size_dim = io.type.tensor_type.shape.dim[3]
    if "_dim_" in head_size_dim.dim_param:
        head_size_dim.Clear()
        head_size_dim.dim_value = head_size

    return io


def fix_encoder_decoder_init_outputs(model: onnx.ModelProto, config: WhisperConfig):
    """Fix outputs for encoder_decoder_init model.

    Uses the same logic as WhisperEncoderDecoderInit.fix_outputs method.
    """
    # Calculate needed values from config
    max_source_positions = config.max_source_positions
    num_heads = config.decoder_attention_heads
    head_size = config.d_model // num_heads

    def fix_kv_cache_dims(output, is_cross: bool = False):
        """Fix dimensions for key/value cache tensors."""
        num_heads_dim = output.type.tensor_type.shape.dim[1]
        if "_dim_" in num_heads_dim.dim_param:
            num_heads_dim.Clear()
            num_heads_dim.dim_value = num_heads
        sequence_length = output.type.tensor_type.shape.dim[2]
        if "_dim_" in sequence_length.dim_param:
            sequence_length.Clear()
            if is_cross:
                sequence_length.dim_value = max_source_positions
            else:
                sequence_length.dim_param = "total_sequence_length"
        head_size_dim = output.type.tensor_type.shape.dim[3]
        if "_dim_" in head_size_dim.dim_param:
            head_size_dim.Clear()
            head_size_dim.dim_value = head_size
        return output

    # ONNX exporter might mark dimensions like 'Transposepresent_value_cross_1_dim_2' in shape inference.
    # We now change the dim_values to the correct one.
    reordered_outputs = []
    self_attn_kv_caches = []
    cross_attn_kv_caches = []

    for output in model.graph.output:
        if "present" not in output.name:
            reordered_outputs.append(output)

        elif "self" in output.name:
            # Self attention KV caches (shouldn't exist for encoder_decoder_init, but handle anyway)
            new_output = fix_kv_cache_dims(output, is_cross=False)
            self_attn_kv_caches.append(new_output)
        else:
            # Cross attention KV caches
            new_output = fix_kv_cache_dims(output, is_cross=True)
            reordered_outputs.append(new_output)

    # For encoder_decoder_init, we typically only have cross attention caches
    reordered_outputs += self_attn_kv_caches + cross_attn_kv_caches

    while len(model.graph.output) > 0:
        model.graph.output.pop()
    model.graph.output.extend(reordered_outputs)
    return model


def fix_decoder_io(model: onnx.ModelProto, config: WhisperConfig, is_inputs: bool = True):
    """Fix order and dimensions of decoder inputs/outputs."""
    io_list = model.graph.input if is_inputs else model.graph.output
    reordered_io = []
    self_attn_kv_caches = []
    cross_attn_kv_caches = []

    for io in io_list:
        if "past" not in io.name and "present" not in io.name:
            reordered_io.append(io)
        elif "self" in io.name or "decoder" in io.name:
            # Self attention KV caches
            new_io = fix_key_value_cache_dims(io, config, is_cross=False, is_output=not is_inputs)
            self_attn_kv_caches.append(new_io)
        else:
            # Cross attention KV caches (encoder)
            new_io = fix_key_value_cache_dims(io, config, is_cross=True, is_output=not is_inputs)
            cross_attn_kv_caches.append(new_io)

    # Sort KV caches by layer index to ensure correct order
    def extract_layer_index(io_tensor):
        import re
        match = re.search(r'_(\d+)$', io_tensor.name)
        return int(match.group(1)) if match else 0

    self_attn_kv_caches.sort(key=extract_layer_index)
    cross_attn_kv_caches.sort(key=extract_layer_index)

    reordered_io += self_attn_kv_caches + cross_attn_kv_caches
    return reordered_io


def fix_decoder_inputs_and_outputs(model: onnx.ModelProto, config: WhisperConfig):
    """Fix inputs and outputs for decoder model."""
    # Fix inputs
    reordered_inputs = fix_decoder_io(model, config, is_inputs=True)
    while len(model.graph.input) > 0:
        model.graph.input.pop()
    model.graph.input.extend(reordered_inputs)

    # Fix outputs
    reordered_outputs = fix_decoder_io(model, config, is_inputs=False)
    while len(model.graph.output) > 0:
        model.graph.output.pop()
    model.graph.output.extend(reordered_outputs)

    return model


# ============================================================================
# Model loading
# ============================================================================


def load_whisper_model(model_id="openai/whisper-large-v3-turbo"):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    config = model.config
    return model, config


# ============================================================================
# Model component getters
# ============================================================================


def get_encoder(model_id="openai/whisper-large-v3-turbo"):
    model, config = load_whisper_model(model_id)
    return WhisperEncoder(model, config)


def get_encoder_decoder_init(model_id="openai/whisper-large-v3-turbo"):
    model, config = load_whisper_model(model_id)
    return WhisperEncoderDecoderInit(model, config)


def get_decoder(model_id="openai/whisper-large-v3-turbo"):
    model, config = load_whisper_model(model_id)
    return WhisperDecoder(model, config)


# ============================================================================
# I/O config
# ============================================================================


config = WhisperConfig.from_pretrained("openai/whisper-large-v3-turbo")


def get_encoder_io_config(model_id="openai/whisper-large-v3-turbo"):
    """IO config for encoder ONNX model.

    Uses the same pattern as onnxruntime transformers:
    1. Get input/output names from lists
    2. Generate dynamic_axes using get_model_dynamic_axes helper
    """
    input_names = ["audio_features"]
    output_names = ["encoder_hidden_states"]

    # Use the helper function to generate dynamic_axes
    dynamic_axes = get_model_dynamic_axes(config, input_names, output_names)

    return {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "dynamic_shapes": None,
    }


def get_encoder_decoder_init_io_config(model_id="openai/whisper-large-v3-turbo"):
    """IO config for encoder_decoder_init ONNX model.

    Uses the same pattern as onnxruntime transformers:
    1. Get input/output names from lists
    2. Generate dynamic_axes using get_model_dynamic_axes helper
    """
    num_layers = config.decoder_layers

    input_names = ["audio_features"]

    output_names = [
        "encoder_hidden_states",
        *list(
            chain.from_iterable(
                (f"present_key_cross_{i}", f"present_value_cross_{i}")
                for i in range(num_layers)
            )
        ),
    ]

    # Use the helper function to generate dynamic_axes
    dynamic_axes = get_model_dynamic_axes(config, input_names, output_names)

    return {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "dynamic_shapes": None,
    }


def get_decoder_io_config(model_id="openai/whisper-large-v3-turbo"):
    """IO config for decoder ONNX model with past key values.

    Uses the same pattern as onnxruntime transformers:
    1. Get input/output names from lists
    2. Generate dynamic_axes using get_model_dynamic_axes helper

    INPUTS: input_ids + encoder_hidden_states + all past_key_values (decoder self + encoder cross)
    OUTPUTS: logits + decoder self present (NOT encoder cross - it's unchanged)
    """
    num_layers = config.decoder_layers

    input_names = ["input_ids", "encoder_hidden_states"]
    for i in range(num_layers):
        # Inputs include BOTH decoder and encoder past (all 4 per layer)
        input_names.extend(
            [
                f"past_key_self_{i}",
                f"past_value_self_{i}",
                f"past_key_cross_{i}",
                f"past_value_cross_{i}",
            ]
        )

    output_names = ["logits"]
    for i in range(num_layers):
        # Outputs include ONLY decoder present (encoder KV cache is constant)
        output_names.extend(
            [
                f"present_key_self_{i}",
                f"present_value_self_{i}",
            ]
        )

    # Use the helper function to generate dynamic_axes
    dynamic_axes = get_model_dynamic_axes(config, input_names, output_names)

    return {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "dynamic_shapes": None,
    }


# ============================================================================
# Dummy input helper functions
# ============================================================================


def get_sample_audio_features(
    batch_size: int,
    sequence_length: int = 3000,
    use_fp16: bool = False,
):
    """Create sample audio features input.

    Shape: (batch_size, num_mel_bins, sequence_length)
    """
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    audio_features = torch.randn(batch_size, config.num_mel_bins, sequence_length, dtype=torch_dtype)
    return audio_features


def get_sample_decoder_input_ids(
    batch_size: int,
    sequence_length: int,
    use_int32: bool = True,
):
    """Create sample decoder input IDs.

    Shape: (batch_size, sequence_length)
    """
    torch_dtype = torch.int32 if use_int32 else torch.int64
    decoder_input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(batch_size, sequence_length), dtype=torch_dtype
    )
    return decoder_input_ids


def get_sample_encoder_hidden_states(
    batch_size: int,
    use_fp16: bool = False,
):
    """Create sample encoder hidden states.

    Shape: (batch_size, num_frames // 2, hidden_size)
    num_frames // 2 = max_source_positions = 1500
    """
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    encoder_hidden_states = torch.randn(
        batch_size, config.max_source_positions, config.d_model, dtype=torch_dtype
    )
    return encoder_hidden_states


# ============================================================================
# Dummy input
# ============================================================================


def get_encoder_dummy_inputs(model=None):
    """Generate dummy inputs for encoder model.

    Returns:
        Tuple containing audio_features tensor
    """
    batch_size = 2
    sequence_length = 3000
    audio_features = get_sample_audio_features(batch_size, sequence_length, use_fp16=False)
    return (audio_features,)


def get_encoder_decoder_init_dummy_inputs(model=None):
    """Generate dummy inputs for encoder_decoder_init model.

    Based on get_sample_encoder_decoder_init_inputs from onnxruntime.
    Returns both audio_features and decoder_input_ids to match reference signature.
    Note: For encoder_decoder_init without beam search, only audio_features is used.

    Returns:
        Dict containing audio_features and decoder_input_ids
    """
    batch_size = 2
    encoder_sequence_length = 3000
    decoder_sequence_length = 1  # Initial decoder sequence for first token
    audio_features = get_sample_audio_features(batch_size, encoder_sequence_length, use_fp16=False)
    decoder_input_ids = get_sample_decoder_input_ids(batch_size, decoder_sequence_length, use_int32=True)
    return {"audio_features": audio_features, "decoder_input_ids": decoder_input_ids}


def get_decoder_dummy_inputs(model=None):
    """Generate dummy inputs for decoder model (decoder with past).

    Returns:
        Tuple containing (input_ids, encoder_hidden_states, past_key_values...)
    """
    batch_size = 2
    decoder_seq = 1  # For decoder with past, typically 1 token at a time
    past_seq = 16    # Past sequence length for self-attention
    encoder_seq = config.max_source_positions  # 1500
    num_heads = config.decoder_attention_heads
    head_dim = config.d_model // num_heads
    num_layers = config.decoder_layers

    # Create input_ids
    input_ids = get_sample_decoder_input_ids(batch_size, decoder_seq, use_int32=True)

    # Create encoder_hidden_states
    encoder_hidden_states = get_sample_encoder_hidden_states(batch_size, use_fp16=False)

    # Create past key values (4 per layer: self_key, self_value, cross_key, cross_value)
    past_key_values = []
    for _ in range(num_layers):
        # Decoder self-attention cache
        past_key_values.append(torch.randn(batch_size, num_heads, past_seq, head_dim))
        past_key_values.append(torch.randn(batch_size, num_heads, past_seq, head_dim))
        # Encoder cross-attention cache
        past_key_values.append(torch.randn(batch_size, num_heads, encoder_seq, head_dim))
        past_key_values.append(torch.randn(batch_size, num_heads, encoder_seq, head_dim))

    return tuple([input_ids, encoder_hidden_states] + past_key_values)


# ============================================================================
# Post-processing functions (called by Olive after ONNX conversion)
# ============================================================================


def post_process_encoder_decoder_init(model_path: str):
    """Post-process encoder_decoder_init ONNX model after conversion."""
    import shutil
    from pathlib import Path

    model = onnx.load(model_path, load_external_data=False)
    config = WhisperConfig.from_pretrained("openai/whisper-large-v3-turbo")

    # Fix outputs
    model = fix_encoder_decoder_init_outputs(model, config)

    # Determine new output path
    model_dir = Path(model_path).parent.parent # Go up to models/
    new_model_path = model_dir / "whisper-large-v3-turbo_encoder_fp16.onnx"

    # Save the fixed model with new name
    onnx.save(
        model,
        str(new_model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{new_model_path.name}.data",
    )
    print(f"Post-processed encoder_decoder_init model saved to {new_model_path}")


def post_process_decoder(model_path: str):
    """Post-process decoder ONNX model after conversion."""
    from pathlib import Path

    model = onnx.load(model_path, load_external_data=False)
    config = WhisperConfig.from_pretrained("openai/whisper-large-v3-turbo")

    # Only fix dimensions, don't reorder inputs/outputs
    # The names from torch.onnx.export should already be correct
    for io in model.graph.input:
        if "past" in io.name or "present" in io.name:
            is_cross = "cross" in io.name or "encoder" in io.name
            fix_key_value_cache_dims(io, config, is_cross=is_cross, is_output=False)

    for io in model.graph.output:
        if "present" in io.name:
            is_cross = "cross" in io.name or "encoder" in io.name
            fix_key_value_cache_dims(io, config, is_cross=is_cross, is_output=True)

    # Determine new output path
    model_dir = Path(model_path).parent.parent  # Go up to models/
    new_model_path = model_dir / "whisper-large-v3-turbo_decoder_fp16.onnx"

    # Save the fixed model with new name
    onnx.save(
        model,
        str(new_model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{new_model_path.name}.data",
    )
    print(f"Post-processed decoder model saved to {new_model_path}")
