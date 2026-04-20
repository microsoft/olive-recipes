# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model loader and dummy input generator for the Nemotron Speech Streaming encoder.

Used by Olive's OnnxConversion pass via the ``model_script`` / ``model_loader``
mechanism (same pattern as the openai-whisper-large-v3-turbo recipe).

Streaming defaults (chunk_size=0.56 s, left_chunks=10) match the values in
nemotron_speech_int4_cpu.json and optimize.py.  If you change the
streaming configuration, update CHUNK_SIZE and LEFT_CHUNKS here **and** in
the other files so the encoder ONNX and genai_config.json stay in sync.
"""

import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Shared streaming constants — single source of truth for this recipe.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 0.56  # seconds
LEFT_CHUNKS = 10
MEL_FEATURES = 128
SUBSAMPLING_FACTOR = 8

# Derived from the 0.6 B model architecture (24 Conformer layers, d_model=1024,
# conv_kernel_size=9).  These are fixed for the model and do not change with
# streaming config.
N_LAYERS = 24
D_MODEL = 1024
CONV_CONTEXT = 8  # conv_kernel_size(9) - 1

_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")


def _get_streaming_shapes():
    """Compute static streaming tensor shapes from the shared constants."""
    right_context = {0.08: 0, 0.16: 1, 0.56: 6, 1.12: 13}.get(CHUNK_SIZE, 13)
    chunk_encoded_frames = int(CHUNK_SIZE * 100) // SUBSAMPLING_FACTOR
    left_context = LEFT_CHUNKS * chunk_encoded_frames

    # The pre-encode cache size is model-dependent; 9 is the default for this
    # model (matches export script fallback and verified model config).
    pre_encode_cache = 9
    chunk_mel_frames = int(CHUNK_SIZE * 100)  # 56 for 0.56 s
    static_mel_frames = chunk_mel_frames + pre_encode_cache  # 65

    return {
        "last_channel_cache_size": left_context,
        "static_mel_frames": static_mel_frames,
    }


def model_loader(model_name):
    """Load the NeMo model and return the streaming encoder wrapper.

    Called by Olive's PyTorchModelHandler when ``model_loader`` is specified
    in the JSON config.  Returns a vanilla ``nn.Module`` that Olive feeds to
    ``torch.onnx.export`` via the OnnxConversion pass.
    """
    sys.path.insert(0, _SCRIPTS_DIR)
    try:
        from export_nemotron_to_onnx_static_shape import (
            _make_streaming_encoder_wrapper,
            get_att_context_size,
        )
    finally:
        sys.path.pop(0)

    import nemo.collections.asr as nemo_asr

    if model_name.endswith(".nemo"):
        asr_model = nemo_asr.models.ASRModel.restore_from(model_name)
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    asr_model = asr_model.cpu()
    asr_model.eval()

    # Configure streaming attention context.
    encoder = asr_model.encoder
    encoder.eval()
    att_context_size = get_att_context_size(CHUNK_SIZE, LEFT_CHUNKS)
    if hasattr(encoder, "set_default_att_context_size"):
        encoder.set_default_att_context_size(att_context_size)

    wrapper = _make_streaming_encoder_wrapper(encoder)
    wrapper.eval()
    return wrapper


def generate_dummy_inputs(model):
    """Generate dummy inputs for ONNX export of the streaming encoder.

    Called by Olive's OnnxConversion pass.  ``model`` is the PyTorchModelHandler
    (not the nn.Module).  Shapes are computed from shared constants to avoid
    reloading the NeMo model.
    """
    shapes = _get_streaming_shapes()
    static_mel_frames = shapes["static_mel_frames"]
    last_channel_cache_size = shapes["last_channel_cache_size"]

    batch = 1
    dummy_audio = torch.randn(batch, static_mel_frames, MEL_FEATURES)
    dummy_length = torch.tensor([static_mel_frames], dtype=torch.int64)
    dummy_cache_ch = torch.zeros(batch, N_LAYERS, last_channel_cache_size, D_MODEL)
    dummy_cache_tm = torch.zeros(batch, N_LAYERS, D_MODEL, CONV_CONTEXT)
    dummy_cache_len = torch.zeros(batch, dtype=torch.int64)

    return (dummy_audio, dummy_length, dummy_cache_ch, dummy_cache_tm, dummy_cache_len)
