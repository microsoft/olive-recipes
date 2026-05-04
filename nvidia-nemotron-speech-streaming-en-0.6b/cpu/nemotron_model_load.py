# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model loaders and dummy input generators for Nemotron Speech Streaming components.

Used by Olive's OnnxConversion pass via the ``model_script`` / ``model_loader``
mechanism. Each component (encoder, decoder, joint) has its own loader and
dummy inputs function, referenced from separate Olive JSON configs.

Streaming defaults (chunk_size=0.56s, left_chunks=10) match the values in
the Olive JSON configs.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Shared streaming constants
# ---------------------------------------------------------------------------
# CHUNK_SIZE is hardcoded because it determines the static ONNX input shapes
# at export time. The NeMo model supports multiple chunk sizes (0.08, 0.16,
# 0.56, 1.12s) at runtime, but once exported to ONNX with static shapes the
# encoder is locked to a single chunk size. 0.56s is the recommended default
# per NVIDIA's documentation (best latency/accuracy trade-off). The value is
# not available from a HuggingFace config — it lives inside the .nemo archive
# as encoder.att_context_size and requires loading the full model to read.
CHUNK_SIZE = 0.56  # seconds
LEFT_CHUNKS = 10
MEL_FEATURES = 128
SUBSAMPLING_FACTOR = 8

# Model architecture constants (0.6B model)
N_LAYERS = 24
D_MODEL = 1024
CONV_CONTEXT = 8  # conv_kernel_size(9) - 1
DECODER_HIDDEN = 640
DECODER_LSTM_LAYERS = 2

MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"


def get_att_context_size(chunk_size: float = CHUNK_SIZE, left_chunks: int = LEFT_CHUNKS):
    """Get attention context size based on chunk size and left chunks."""
    right_context = {0.08: 0, 0.16: 1, 0.56: 6, 1.12: 13}.get(chunk_size, 13)
    chunk_encoded_frames = int(chunk_size * 100) // SUBSAMPLING_FACTOR
    left_context = left_chunks * chunk_encoded_frames
    return [left_context, right_context]


def _get_streaming_shapes():
    """Compute static streaming tensor shapes from the shared constants."""
    chunk_encoded_frames = int(CHUNK_SIZE * 100) // SUBSAMPLING_FACTOR
    left_context = LEFT_CHUNKS * chunk_encoded_frames
    pre_encode_cache = 9
    chunk_mel_frames = int(CHUNK_SIZE * 100)  # 56 for 0.56s
    static_mel_frames = chunk_mel_frames + pre_encode_cache  # 65

    return {
        "last_channel_cache_size": left_context,
        "static_mel_frames": static_mel_frames,
    }


def _load_nemo_model(model_name=MODEL_NAME):
    """Load the NeMo ASR model (shared across loaders)."""
    import nemo.collections.asr as nemo_asr

    if model_name.endswith(".nemo"):
        asr_model = nemo_asr.models.ASRModel.restore_from(model_name)
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    asr_model = asr_model.cpu()
    asr_model.eval()
    return asr_model


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class StreamingEncoderWrapper(nn.Module):
    """Wrap the NeMo CacheAware encoder for streaming ONNX export.

    Provides an untyped forward() that torch.onnx.export can trace.
    Cache tensors use [B, n_layers, ...] layout for ONNX I/O consistency.
    """

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, audio_signal, length,
                cache_last_channel, cache_last_time, cache_last_channel_len):
        audio_signal = audio_signal.transpose(1, 2)  # [B, T, mel] -> [B, mel, T]
        encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next = \
            self.enc.forward_for_export(
                audio_signal=audio_signal,
                length=length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )
        encoded = encoded.transpose(1, 2)  # [B, D, T] -> [B, T, D]
        return encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next


def encoder_model_loader(model_name):
    """Load the NeMo model and return the streaming encoder wrapper."""
    asr_model = _load_nemo_model(model_name)
    encoder = asr_model.encoder
    encoder.eval()

    att_context_size = get_att_context_size()
    if hasattr(encoder, "set_default_att_context_size"):
        encoder.set_default_att_context_size(att_context_size)

    wrapper = StreamingEncoderWrapper(encoder)
    wrapper.eval()
    return wrapper


def encoder_dummy_inputs(model):
    """Generate dummy inputs for ONNX export of the streaming encoder."""
    shapes = _get_streaming_shapes()
    static_mel_frames = shapes["static_mel_frames"]
    last_channel_cache_size = shapes["last_channel_cache_size"]

    batch = 1
    return (
        torch.randn(batch, static_mel_frames, MEL_FEATURES),
        torch.tensor([static_mel_frames], dtype=torch.int64),
        torch.zeros(batch, N_LAYERS, last_channel_cache_size, D_MODEL),
        torch.zeros(batch, N_LAYERS, D_MODEL, CONV_CONTEXT),
        torch.zeros(batch, dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# Decoder (stateful LSTM)
# ---------------------------------------------------------------------------

class StatefulDecoderWrapper(nn.Module):
    """Wrap the NeMo decoder to expose LSTM states as explicit I/O."""

    def __init__(self, dec):
        super().__init__()
        self.decoder = dec
        self.decoder._rnnt_export = True

    def forward(self, targets, h_in, c_in):
        g, states = self.decoder.predict(
            y=targets, state=(h_in, c_in), add_sos=False
        )
        h_out, c_out = states
        g = g.transpose(1, 2)  # [B, 1, D] -> [B, D, 1]
        return g, h_out, c_out


def decoder_model_loader(model_name):
    """Load the NeMo model and return the stateful decoder wrapper."""
    asr_model = _load_nemo_model(model_name)
    decoder = asr_model.decoder
    decoder.eval()

    wrapper = StatefulDecoderWrapper(decoder)
    wrapper.eval()
    return wrapper


def decoder_dummy_inputs(model):
    """Generate dummy inputs for ONNX export of the stateful decoder."""
    batch = 1
    return (
        torch.zeros(batch, 1, dtype=torch.int64),
        torch.zeros(DECODER_LSTM_LAYERS, batch, DECODER_HIDDEN, dtype=torch.float32),
        torch.zeros(DECODER_LSTM_LAYERS, batch, DECODER_HIDDEN, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Joint network
# ---------------------------------------------------------------------------

class JointWrapper(nn.Module):
    """Wrap the NeMo RNNTJoint so torch.onnx.export can trace it."""

    def __init__(self, j):
        super().__init__()
        self.joint = j

    def forward(self, encoder_output, decoder_output):
        return self.joint.joint(encoder_output, decoder_output)


def joint_model_loader(model_name):
    """Load the NeMo model and return the joint network wrapper."""
    asr_model = _load_nemo_model(model_name)
    joint = asr_model.joint
    joint.eval()

    wrapper = JointWrapper(joint)
    wrapper.eval()
    return wrapper


def joint_dummy_inputs(model):
    """Generate dummy inputs for ONNX export of the joint network."""
    batch = 1
    return (
        torch.randn(batch, 1, D_MODEL),
        torch.randn(batch, 1, DECODER_HIDDEN),
    )
