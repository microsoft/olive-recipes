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

import numpy as np
import torch
import torch.nn as nn

from olive.common.quant.state_dict import install_quant_tensor_param
from olive.common.quant.tensor import QuantTensor

try:
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, PreTrainedModel, PretrainedConfig
    from transformers.models.auto import configuration_auto, modeling_auto
except Exception:  # pragma: no cover
    AutoConfig = None
    AutoModelForSpeechSeq2Seq = None
    PreTrainedModel = None
    PretrainedConfig = None
    configuration_auto = None
    modeling_auto = None

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


def _restore_quantized_state_dict(model, state_dict):
    """Rebind QuantTensor checkpoint entries into the model's live Parameters."""
    if not isinstance(state_dict, dict):
        return False

    quantized_entries = {}
    for name, value in state_dict.items():
        if isinstance(value, QuantTensor):
            quantized_entries[name] = value
        elif isinstance(value, torch.nn.Parameter) and isinstance(getattr(value, "data", None), QuantTensor):
            quantized_entries[name] = value.data

    if not quantized_entries:
        return False

    filtered_state = {k: v for k, v in state_dict.items() if k not in quantized_entries}
    for name, qt in quantized_entries.items():
        module_name, _, pname = name.rpartition(".")
        if not module_name or not pname:
            continue
        module = model
        if module_name:
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                module = None
        if module is None:
            continue
        if pname not in module._parameters:
            continue
        install_quant_tensor_param(module, pname, qt)

    if filtered_state:
        model.load_state_dict(filtered_state, strict=False)
    return True


def _load_nemo_model(model_name=MODEL_NAME):
    """Load the NeMo ASR model (shared across loaders).

    Local paths can point at saved checkpoint files or model directories, which are not HF repo IDs.
    In that case we detect them explicitly and overlay the checkpoint weights on a fresh base NeMo model.
    """
    import nemo.collections.asr as nemo_asr

    local_model_path = None
    if isinstance(model_name, str):
        candidate = model_name.strip()
        if candidate and candidate.endswith((".pt", ".pth", ".ckpt")):
            local_model_path = candidate
        else:
            local_path = __import__("pathlib").Path(candidate)
            if local_path.exists():
                if local_path.is_dir():
                    for suffix in ("model.pt", "model.pth", "model.ckpt"):
                        check = local_path / suffix
                        if check.exists():
                            local_model_path = str(check)
                            break
                    if local_model_path is None:
                        for candidate_file in sorted(local_path.iterdir()):
                            if candidate_file.is_file() and candidate_file.suffix.lower() in {".pt", ".pth", ".ckpt"}:
                                local_model_path = str(candidate_file)
                                break
                elif local_path.is_file() and local_path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
                    local_model_path = str(local_path)

    if local_model_path is not None:
        base_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        state_dict = torch.load(local_model_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            if not _restore_quantized_state_dict(base_model, state_dict):
                base_model.load_state_dict(state_dict, strict=False)
        else:
            raise TypeError(f"Unsupported NeMo checkpoint format at {local_model_path!r}: expected state_dict")
        base_model = base_model.cpu()
        base_model.eval()
        return base_model

    if model_name.endswith(".nemo"):
        asr_model = nemo_asr.models.ASRModel.restore_from(model_name)
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    asr_model = asr_model.cpu()
    asr_model.eval()
    return asr_model


def register_nemotron_hf_bridge():
    """Register the real Transformers Nemotron ASR classes when available."""
    if AutoConfig is None or AutoModelForSpeechSeq2Seq is None:
        return None

    try:
        from transformers.models.nemotron_asr_streaming import NemotronAsrStreamingConfig, NemotronAsrStreamingModel

        try:
            AutoConfig.register("nemotron_asr_streaming", NemotronAsrStreamingConfig, exist_ok=True)
            AutoModelForSpeechSeq2Seq.register(NemotronAsrStreamingConfig, NemotronAsrStreamingModel, exist_ok=True)
        except ValueError:
            pass
        if configuration_auto is not None:
            mapping = getattr(configuration_auto, "CONFIG_MAPPING", None)
            if mapping is not None:
                mapping["nemotron_asr_streaming"] = NemotronAsrStreamingConfig
        if modeling_auto is not None:
            seq2seq_mapping = getattr(modeling_auto, "MODEL_FOR_SPEECH_SEQ_TO_SEQ_MAPPING", None)
            if seq2seq_mapping is not None:
                seq2seq_mapping["nemotron_asr_streaming"] = NemotronAsrStreamingModel
            model_mapping = getattr(modeling_auto, "MODEL_MAPPING_NAMES", None)
            if model_mapping is not None:
                model_mapping["nemotron_asr_streaming"] = "NemotronAsrStreamingModel"
        return NemotronAsrStreamingModel
    except Exception:
        pass

    if PretrainedConfig is None or PreTrainedModel is None:
        return None

    class NemotronAsrStreamingConfig(PretrainedConfig):
        model_type = "nemotron_asr_streaming"

        def __init__(self, vocab_size=1024, hidden_size=1024, encoder_hidden_size=1024, decoder_hidden_size=640,
                     num_hidden_layers=24, **kwargs):
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.encoder_hidden_size = encoder_hidden_size
            self.decoder_hidden_size = decoder_hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.architectures = ["NemotronAsrStreamingModel"]

    class NemotronAsrStreamingModel(PreTrainedModel):
        config_class = NemotronAsrStreamingConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = None
            self.config = config

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            asr_model = _load_nemo_model(pretrained_model_name_or_path)
            instance = cls(NemotronAsrStreamingConfig())
            instance.model = asr_model
            instance.eval()
            return instance

        def forward(self, *args, **kwargs):
            if self.model is None:
                raise ValueError("Nemotron ASR model is not loaded.")
            return self.model(*args, **kwargs)

    try:
        AutoConfig.register("nemotron_asr_streaming", NemotronAsrStreamingConfig, exist_ok=True)
        AutoModelForSpeechSeq2Seq.register(NemotronAsrStreamingConfig, NemotronAsrStreamingModel, exist_ok=True)
    except ValueError:
        pass

    if configuration_auto is not None:
        mapping = getattr(configuration_auto, "CONFIG_MAPPING", None)
        if mapping is not None:
            mapping["nemotron_asr_streaming"] = NemotronAsrStreamingConfig

    if modeling_auto is not None:
        seq2seq_mapping = getattr(modeling_auto, "MODEL_FOR_SPEECH_SEQ_TO_SEQ_MAPPING", None)
        if seq2seq_mapping is not None:
            seq2seq_mapping["nemotron_asr_streaming"] = NemotronAsrStreamingModel
        model_mapping = getattr(modeling_auto, "MODEL_MAPPING_NAMES", None)
        if model_mapping is not None:
            model_mapping["nemotron_asr_streaming"] = "NemotronAsrStreamingModel"

    return NemotronAsrStreamingModel


register_nemotron_hf_bridge()


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