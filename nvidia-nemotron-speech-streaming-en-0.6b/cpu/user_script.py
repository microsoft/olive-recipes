# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Olive user_script for Nemotron Speech Streaming RNNT model.

Provides model loaders, IO configs, and dummy input generators for the
three RNNT components (encoder, decoder, joint) following the same pattern
as the Whisper recipe.

The NeMo model is loaded once and cached; wrapper classes provide clean
forward() signatures that torch.onnx.export can trace.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model loading (singleton)
# ---------------------------------------------------------------------------

_asr_model = None
_model_name = "nvidia/nemotron-speech-streaming-en-0.6b"

# Streaming defaults
CHUNK_SIZE = 0.56
LEFT_CHUNKS = 10


def _load_nemo_model():
    global _asr_model
    if _asr_model is None:
        import nemo.collections.asr as nemo_asr

        _asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=_model_name)
        _asr_model.cpu().eval()
    return _asr_model


# ---------------------------------------------------------------------------
# Streaming cache shape helpers
# ---------------------------------------------------------------------------


def _get_att_context_size(chunk_size=CHUNK_SIZE, left_chunks=LEFT_CHUNKS, subsampling_factor=8):
    right_context = {0.08: 0, 0.16: 1, 0.56: 6, 1.12: 13}.get(chunk_size, 13)
    chunk_encoded_frames = int(chunk_size * 100) // subsampling_factor
    left_context = left_chunks * chunk_encoded_frames
    return [left_context, right_context]


def _get_streaming_cache_shapes(encoder, att_context_size, chunk_size=CHUNK_SIZE):
    n_layers = getattr(encoder, "num_layers", 24)
    d_model = getattr(encoder, "d_model", 1024)

    if hasattr(encoder, "get_streaming_config"):
        cfg = encoder.get_streaming_config()
        last_channel_cache_size = cfg.get("last_channel_cache_size", att_context_size[0])
        shift_size = cfg.get("shift_size", [105, 112])
        chunk_frames = shift_size[1] if len(shift_size) > 1 else shift_size[0]
        pre_encode_cache = cfg.get("pre_encode_cache_size", [0, 9])
    else:
        last_channel_cache_size = att_context_size[0]
        chunk_frames = int(chunk_size * 100)
        pre_encode_cache = [0, 9]

    conv_context = 8
    if hasattr(encoder, "layers") and len(encoder.layers) > 0:
        layer = encoder.layers[0]
        if hasattr(layer, "conv") and hasattr(layer.conv, "conv"):
            conv = layer.conv.conv
            if hasattr(conv, "kernel_size"):
                ks = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
                conv_context = ks - 1

    return {
        "n_layers": n_layers,
        "d_model": d_model,
        "last_channel_cache_size": last_channel_cache_size,
        "conv_context": conv_context,
        "chunk_mel_frames": chunk_frames,
        "pre_encode_cache_size": pre_encode_cache[-1] if isinstance(pre_encode_cache, (list, tuple)) else pre_encode_cache,
    }


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class StreamingEncoderWrapper(nn.Module):
    """Wrap the NeMo CacheAware encoder for streaming ONNX export.

    Cache tensors use [B, n_layers, ...] layout for ONNX I/O.
    """

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len):
        audio_signal = audio_signal.transpose(1, 2)
        encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next = self.enc.forward_for_export(
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        encoded = encoded.transpose(1, 2)
        return encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next


class StatefulDecoderWrapper(nn.Module):
    """Wrap the NeMo decoder to expose LSTM h/c states as explicit I/O."""

    def __init__(self, dec):
        super().__init__()
        self.decoder = dec
        self.decoder._rnnt_export = True

    def forward(self, targets, h_in, c_in):
        g, states = self.decoder.predict(y=targets, state=(h_in, c_in), add_sos=False)
        h_out, c_out = states
        g = g.transpose(1, 2)
        return g, h_out, c_out


class JointWrapper(nn.Module):
    """Wrap the NeMo RNNTJoint for ONNX export."""

    def __init__(self, j):
        super().__init__()
        self.joint = j

    def forward(self, encoder_output, decoder_output):
        return self.joint.joint(encoder_output, decoder_output)


# ---------------------------------------------------------------------------
# Model loaders (called by Olive via PyTorchModelHandler.model_loader)
# ---------------------------------------------------------------------------


def get_streaming_encoder(model_name=_model_name):
    asr_model = _load_nemo_model()
    encoder = asr_model.encoder
    encoder.eval()
    att_context_size = _get_att_context_size()
    if hasattr(encoder, "set_default_att_context_size"):
        encoder.set_default_att_context_size(att_context_size)
    return StreamingEncoderWrapper(encoder)


def get_decoder(model_name=_model_name):
    asr_model = _load_nemo_model()
    decoder = asr_model.decoder
    decoder.eval()
    return StatefulDecoderWrapper(decoder)


def get_joint(model_name=_model_name):
    asr_model = _load_nemo_model()
    joint = asr_model.joint
    joint.eval()
    return JointWrapper(joint)


# ---------------------------------------------------------------------------
# IO configs (called by Olive via PyTorchModelHandler.io_config)
# ---------------------------------------------------------------------------


def get_encoder_io_config(model_name=_model_name):
    """Static-shape IO config for the streaming encoder."""
    return {
        "input_names": [
            "audio_signal",
            "length",
            "cache_last_channel",
            "cache_last_time",
            "cache_last_channel_len",
        ],
        "output_names": [
            "outputs",
            "encoded_lengths",
            "cache_last_channel_next",
            "cache_last_time_next",
            "cache_last_channel_len_next",
        ],
        "dynamic_axes": None,
        "dynamic_shapes": None,
    }


def get_decoder_io_config(model_name=_model_name):
    return {
        "input_names": ["targets", "h_in", "c_in"],
        "output_names": ["decoder_output", "h_out", "c_out"],
        "dynamic_axes": {
            "targets": {0: "batch", 1: "target_len"},
            "h_in": {1: "batch"},
            "c_in": {1: "batch"},
            "decoder_output": {0: "batch", 2: "target_len"},
            "h_out": {1: "batch"},
            "c_out": {1: "batch"},
        },
        "dynamic_shapes": None,
    }


def get_joint_io_config(model_name=_model_name):
    return {
        "input_names": ["encoder_output", "decoder_output"],
        "output_names": ["joint_output"],
        "dynamic_axes": {
            "encoder_output": {0: "batch", 1: "time"},
            "decoder_output": {0: "batch", 1: "target_len"},
            "joint_output": {0: "batch", 1: "time", 2: "target_len"},
        },
        "dynamic_shapes": None,
    }


# ---------------------------------------------------------------------------
# Dummy inputs (called by Olive via PyTorchModelHandler.dummy_inputs_func)
# ---------------------------------------------------------------------------


def get_encoder_dummy_inputs(model=None):
    """Generate dummy inputs matching static streaming shapes."""
    asr_model = _load_nemo_model()
    encoder = asr_model.encoder
    att_context_size = _get_att_context_size()
    cache_cfg = _get_streaming_cache_shapes(encoder, att_context_size)

    batch_size = 1
    mel_features = 128
    static_mel_frames = cache_cfg["chunk_mel_frames"] + cache_cfg["pre_encode_cache_size"]
    n_layers = cache_cfg["n_layers"]
    d_model = cache_cfg["d_model"]
    cache_len = cache_cfg["last_channel_cache_size"]
    conv_context = cache_cfg["conv_context"]

    return (
        torch.randn(batch_size, static_mel_frames, mel_features),
        torch.tensor([static_mel_frames], dtype=torch.int64),
        torch.zeros(batch_size, n_layers, cache_len, d_model),
        torch.zeros(batch_size, n_layers, d_model, conv_context),
        torch.zeros(batch_size, dtype=torch.int64),
    )


def get_decoder_dummy_inputs(model=None):
    batch_size = 1
    hidden_size = 640
    num_layers = 2
    return (
        torch.zeros(batch_size, 1, dtype=torch.int64),
        torch.zeros(num_layers, batch_size, hidden_size),
        torch.zeros(num_layers, batch_size, hidden_size),
    )


def get_joint_dummy_inputs(model=None):
    asr_model = _load_nemo_model()
    encoder_dim = asr_model.cfg.encoder.d_model
    decoder_dim = getattr(asr_model.decoder, "pred_hidden", None) or getattr(asr_model.decoder, "d_model", 640)
    batch_size = 1
    return (
        torch.randn(batch_size, 1, encoder_dim),
        torch.randn(batch_size, 1, decoder_dim),
    )
