# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model loaders and dummy input generators for Parakeet TDT 0.6B v2.

Used by Olive's OnnxConversion pass via the ``model_script`` /
``model_loader`` mechanism. Each component (encoder, decoder, joint) has
its own loader and dummy inputs function, referenced from separate Olive
JSON configs.

Unlike Nemotron Speech Streaming, Parakeet v2 is an **offline** (non
cache-aware) FastConformer + TDT model. The encoder takes variable-length
audio with no cache tensors. Long-form audio is handled at the runtime
layer (onnxruntime-genai parakeet runner) by feeding overlapping windows.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Architecture constants (Parakeet TDT 0.6B v2)
# ---------------------------------------------------------------------------
MEL_FEATURES = 128
SUBSAMPLING_FACTOR = 8

N_LAYERS = 24
D_MODEL = 1024
DECODER_HIDDEN = 640
DECODER_LSTM_LAYERS = 2

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


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
# Encoder (offline FastConformer)
# ---------------------------------------------------------------------------

class OfflineEncoderWrapper(nn.Module):
    """Wrap the offline FastConformer encoder for ONNX export.

    Forward takes mel features in NeMo's native ``[B, mel, T]`` layout
    (matching what the genai parakeet runner feeds at inference) and
    returns encoded features ``[B, D, T']`` (channel-first time at dim 2,
    matching what the genai parakeet runner reads as ``enc_shape[2]``)
    plus encoded lengths ``[B]``. Time axis is dynamic.
    """

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, audio_signal, length):
        encoded, encoded_len = self.enc(audio_signal=audio_signal, length=length)
        # Keep NeMo's native [B, D, T'] layout; the genai parakeet runner
        # indexes encoder output as enc_data[d * enc_time + t].
        return encoded, encoded_len


def encoder_model_loader(model_name):
    asr_model = _load_nemo_model(model_name)
    encoder = asr_model.encoder
    encoder.eval()
    wrapper = OfflineEncoderWrapper(encoder)
    wrapper.eval()
    return wrapper


def encoder_dummy_inputs(model):
    """Dummy inputs: 6s of audio at 100 fps mel = 600 frames in [B, mel, T]."""
    batch = 1
    dummy_frames = 600
    return (
        torch.randn(batch, MEL_FEATURES, dummy_frames),
        torch.tensor([dummy_frames], dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# Decoder (stateful LSTM prediction network)
# ---------------------------------------------------------------------------

class StatefulDecoderWrapper(nn.Module):
    """Wrap the NeMo decoder to expose LSTM states as explicit I/O.

    Also passes through the targets length so the genai parakeet runner can
    bind ``target_length_orig`` -> ``target_length`` for stateful stepping.
    """

    def __init__(self, dec):
        super().__init__()
        self.decoder = dec
        self.decoder._rnnt_export = True

    def forward(self, targets, target_length_orig, h_in, c_in):
        g, states = self.decoder.predict(
            y=targets, state=(h_in, c_in), add_sos=False
        )
        h_out, c_out = states
        g = g.transpose(1, 2)  # [B, 1, D] -> [B, D, 1]
        return g, target_length_orig, h_out, c_out


def decoder_model_loader(model_name):
    asr_model = _load_nemo_model(model_name)
    decoder = asr_model.decoder
    decoder.eval()
    wrapper = StatefulDecoderWrapper(decoder)
    wrapper.eval()
    return wrapper


def decoder_dummy_inputs(model):
    batch = 1
    return (
        torch.zeros(batch, 1, dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        torch.zeros(DECODER_LSTM_LAYERS, batch, DECODER_HIDDEN, dtype=torch.float32),
        torch.zeros(DECODER_LSTM_LAYERS, batch, DECODER_HIDDEN, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Joint network (TDT: vocab logits + duration logits)
# ---------------------------------------------------------------------------

class JointWrapper(nn.Module):
    """Wrap the NeMo TDT joint so torch.onnx.export can trace it."""

    def __init__(self, j):
        super().__init__()
        self.joint = j

    def forward(self, encoder_output, decoder_output):
        return self.joint.joint(encoder_output, decoder_output)


def joint_model_loader(model_name):
    asr_model = _load_nemo_model(model_name)
    joint = asr_model.joint
    joint.eval()
    wrapper = JointWrapper(joint)
    wrapper.eval()
    return wrapper


def joint_dummy_inputs(model):
    batch = 1
    return (
        torch.randn(batch, 1, D_MODEL),
        torch.randn(batch, 1, DECODER_HIDDEN),
    )
