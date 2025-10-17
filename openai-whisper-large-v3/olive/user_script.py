# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from transformers import AutoModelForSpeechSeq2Seq, WhisperConfig
from transformers.cache_utils import EncoderDecoderCache


class WhisperEncoder(torch.nn.Module):
    """Return encoder last hidden state."""

    def __init__(self, model, config: WhisperConfig):
        super().__init__()
        self.encoder = model.model.encoder
        self.config = config

    def forward(self, input_features):
        return self.encoder(input_features)[0]


class DecoderWithoutPast(torch.nn.Module):
    """Decoder wrapper for the first decoding step.

    ONNX input names: input_ids, encoder_hidden_states
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, input_ids, encoder_hidden_states):
        """
        Args:
            input_ids: Input IDs for decoder
            encoder_hidden_states: Encoder hidden states
        """
        outputs = self.model.model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        logits = self.model.proj_out(outputs.last_hidden_state)
        present = outputs.past_key_values
        if isinstance(present, EncoderDecoderCache):
            present = present.to_legacy_cache()
        else:
            # tuple of tuples already in legacy format
            present = tuple(present)

        flat = [logits]
        for layer in present:
            flat.extend(layer)
        return tuple(flat)


class DecoderWithPast(torch.nn.Module):
    """Decoder wrapper for subsequent decoding steps with KV cache."""

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, input_ids, *past_key_values):
        """
        Args:
            input_ids: shape (batch, 1) for next token
            *past_key_values: Flattened past key values
                - For each layer: decoder.key, decoder.value, encoder.key, encoder.value
                - Total: num_layers * 4 tensors

        Returns:
            Tuple of (logits, present.0.decoder.key, present.0.decoder.value, ...)
            Only decoder present outputs
        """
        num_layers = self.config.decoder_layers
        legacy = []
        idx = 0

        # Extract first encoder key to get shape for dummy encoder_hidden_states
        first_encoder_key = past_key_values[2]  # past_key_values.0.encoder.key
        batch_size = first_encoder_key.shape[0]
        encoder_seq_len = first_encoder_key.shape[2]
        hidden_size = self.config.d_model

        # Create dummy encoder_hidden_states to force torch.onnx.export to keep encoder inputs
        # The actual encoder_hidden_states won't be used because we have encoder cache
        # But this dummy operation creates a dependency on encoder cache inputs
        encoder_hidden_states = torch.zeros(
            batch_size,
            encoder_seq_len,
            hidden_size,
            dtype=first_encoder_key.dtype,
            device=first_encoder_key.device,
        )

        for _ in range(num_layers):
            decoder_key = past_key_values[idx]
            decoder_value = past_key_values[idx + 1]
            encoder_key = past_key_values[idx + 2]
            encoder_value = past_key_values[idx + 3]

            # Use encoder inputs to create encoder_hidden_states shape
            # This creates dependency so ONNX won't optimize away encoder inputs
            encoder_hidden_states = (
                encoder_hidden_states
                + encoder_key.sum() * 0.0
                + encoder_value.sum() * 0.0
            )

            legacy.append((decoder_key, decoder_value, encoder_key, encoder_value))
            idx += 4

        cache = EncoderDecoderCache.from_legacy_cache(tuple(legacy))

        outputs = self.model.model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,  # Pass dummy (forces encoder cache usage)
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        logits = self.model.proj_out(outputs.last_hidden_state)

        present = outputs.past_key_values
        if isinstance(present, EncoderDecoderCache):
            present = present.to_legacy_cache()
        else:
            present = tuple(present)

        # Filter outputs: only return decoder present, NOT encoder present
        flat = [logits]
        for layer in present:
            flat.append(layer[0])  # decoder key
            flat.append(layer[1])  # decoder value
        return tuple(flat)


# ============================================================================
# Model loader
# ============================================================================

model_id = "openai/whisper-large-v3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)


def get_encoder(model_id="openai/whisper-large-v3"):
    return WhisperEncoder(model, model.config)


def get_decoder(model_id="openai/whisper-large-v3"):
    return DecoderWithoutPast(model, model.config)


def get_decoder_with_past(model_id="openai/whisper-large-v3"):
    return DecoderWithPast(model, model.config)


# ============================================================================
# IO configuration
# ============================================================================

config = WhisperConfig.from_pretrained(model_id)


def get_encoder_io_config(model_id="openai/whisper-large-v3"):
    input_names = ["input_features"]
    output_names = ["last_hidden_state"]

    dynamic_axes = {
        "input_features": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"},
    }

    dynamic_shapes = None

    return {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "dynamic_shapes": dynamic_shapes,
    }


def get_decoder_io_config(model_id="openai/whisper-large-v3"):
    num_layers = config.decoder_layers
    input_names = ["input_ids", "encoder_hidden_states"]
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.extend(
            [
                f"present.{i}.decoder.key",
                f"present.{i}.decoder.value",
                f"present.{i}.encoder.key",
                f"present.{i}.encoder.value",
            ]
        )

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
        "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length / 2"},
        "logits": {0: "batch_size", 1: "decoder_sequence_length"},
    }
    for i in range(num_layers):
        dynamic_axes[f"present.{i}.decoder.key"] = {
            0: "batch_size",
            2: "decoder_sequence_length",
        }
        dynamic_axes[f"present.{i}.decoder.value"] = {
            0: "batch_size",
            2: "decoder_sequence_length",
        }
        dynamic_axes[f"present.{i}.encoder.key"] = {
            0: "batch_size",
            2: "encoder_sequence_length / 2",
        }
        dynamic_axes[f"present.{i}.encoder.value"] = {
            0: "batch_size",
            2: "encoder_sequence_length / 2",
        }

    dynamic_shapes = None

    return {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "dynamic_shapes": dynamic_shapes,
    }


def get_decoder_with_past_io_config(model_id="openai/whisper-large-v3"):
    """IO config decoder_with_past ONNX model.

    INPUTS: All past_key_values (decoder + encoder) - 129 total
    OUTPUTS: Only decoder present (NOT encoder) - 65 total
    """
    num_layers = config.decoder_layers
    input_names = ["input_ids"]
    for i in range(num_layers):
        # Inputs include BOTH decoder and encoder past (all 4 per layer)
        input_names.extend(
            [
                f"past_key_values.{i}.decoder.key",
                f"past_key_values.{i}.decoder.value",
                f"past_key_values.{i}.encoder.key",
                f"past_key_values.{i}.encoder.value",
            ]
        )

    output_names = ["logits"]
    for i in range(num_layers):
        # Outputs include ONLY decoder present (encoder KV cache is constant)
        output_names.extend(
            [
                f"present.{i}.decoder.key",
                f"present.{i}.decoder.value",
            ]
        )

    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }
    for i in range(num_layers):
        # Input axes
        dynamic_axes[f"past_key_values.{i}.decoder.key"] = {
            0: "batch_size",
            2: "past_decoder_sequence_length",
        }
        dynamic_axes[f"past_key_values.{i}.decoder.value"] = {
            0: "batch_size",
            2: "past_decoder_sequence_length",
        }
        dynamic_axes[f"past_key_values.{i}.encoder.key"] = {
            0: "batch_size",
            2: "encoder_sequence_length_out",
        }
        dynamic_axes[f"past_key_values.{i}.encoder.value"] = {
            0: "batch_size",
            2: "encoder_sequence_length_out",
        }

        # Output axes - ONLY decoder (with +1 for new token)
        dynamic_axes[f"present.{i}.decoder.key"] = {
            0: "batch_size",
            2: "past_decoder_sequence_length + 1",
        }
        dynamic_axes[f"present.{i}.decoder.value"] = {
            0: "batch_size",
            2: "past_decoder_sequence_length + 1",
        }

    dynamic_shapes = None

    return {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "dynamic_shapes": dynamic_shapes,
    }


# ============================================================================
# Dummy input
# ============================================================================


def get_encoder_dummy_inputs(model=None):
    batch = 2
    features = config.num_mel_bins
    seq = 3000
    input_features = torch.randn(batch, features, seq)
    return (input_features,)


def get_decoder_dummy_inputs(model=None):
    batch = 2  # DEFAULT_DUMMY_SHAPES['batch_size']
    decoder_seq = 16  # DEFAULT_DUMMY_SHAPES['sequence_length']
    encoder_seq = config.max_source_positions // 2  # 750
    hidden = config.d_model
    input_ids = torch.randint(
        0, config.vocab_size, (batch, decoder_seq), dtype=torch.int64
    )
    encoder_hidden_states = torch.randn(batch, encoder_seq, hidden)
    return (input_ids, encoder_hidden_states)


def get_decoder_with_past_dummy_inputs(model=None):
    batch = 2  # DEFAULT_DUMMY_SHAPES['batch_size']
    past_seq = 16  # DEFAULT_DUMMY_SHAPES['sequence_length']
    encoder_seq = config.max_source_positions  # 1500 (encoder_sequence_length)
    num_heads = config.decoder_attention_heads
    head_dim = config.d_model // num_heads
    num_layers = config.decoder_layers

    input_ids = torch.randint(0, config.vocab_size, (batch, 1), dtype=torch.int64)

    past_key_values = []
    for _ in range(num_layers):
        # Decoder self-attention cache
        past_key_values.append(torch.randn(batch, num_heads, past_seq, head_dim))
        past_key_values.append(torch.randn(batch, num_heads, past_seq, head_dim))
        # Encoder cross-attention cache
        past_key_values.append(torch.randn(batch, num_heads, encoder_seq, head_dim))
        past_key_values.append(torch.randn(batch, num_heads, encoder_seq, head_dim))

    return tuple([input_ids] + past_key_values)
