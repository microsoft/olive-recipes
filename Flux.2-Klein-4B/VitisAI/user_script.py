# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Olive user_script for FLUX.2-klein-4B ONNX export.

import torch
import torch.nn as nn
from torch.onnx import symbolic_helper

try:
    from torch.onnx._type_utils import JitScalarType
except (ImportError, ModuleNotFoundError):
    from torch.onnx import JitScalarType


# =============================================================================
# Transformer
# =============================================================================

# ---------------------------------------------------------------------------
# RMSNorm decomposition for opset 17
#
# aten::rms_norm has no native ONNX op at opset 17. Decompose it as:
#   Cast → Pow(2) → ReduceMean(axes_i, keepdims=1) → Add(eps) → Sqrt
#        → Div → Cast → Mul(weight)
#
# keepdims=1 preserves the reduced dim as size-1, so Div broadcasts without
# an extra Unsqueeze. The resulting 8-node subgraph is fused into
# SimplifiedLayerNormalization by the onnxruntime.transformers mmdit optimizer.
# ---------------------------------------------------------------------------

@symbolic_helper.parse_args("v", "is", "v", "v")
def _rms_norm_symbolic(g, input, normalized_shape, weight, eps):
    eps_val = symbolic_helper._maybe_get_const(eps, "f")
    if eps_val is None or not isinstance(eps_val, (int, float)):
        eps_val = 1e-6

    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    input_dtype = JitScalarType.from_value(input, JitScalarType.FLOAT)
    fp32_onnx   = JitScalarType.FLOAT.onnx_type()

    input_fp32     = g.op("Cast",     input,     to_i=fp32_onnx)
    pow_two        = g.op("Constant", value_t=torch.tensor(2.0, dtype=torch.float32))
    x_squared      = g.op("Pow",      input_fp32, pow_two)
    x_squared_mean = g.op("ReduceMean", x_squared, axes_i=axes)
    eps_const      = g.op("Constant", value_t=torch.tensor(eps_val, dtype=torch.float32))
    rms            = g.op("Sqrt", g.op("Add", x_squared_mean, eps_const))
    normalized     = g.op("Cast", g.op("Div", input_fp32, rms), to_i=input_dtype.onnx_type())

    if weight is not None and not symbolic_helper._is_none(weight):
        normalized = g.op("Mul", normalized, weight)

    normalized.setType(input.type())
    return normalized


torch.onnx.register_custom_op_symbolic("aten::rms_norm", _rms_norm_symbolic, 17)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class FluxTransformerWrapper(nn.Module):
    """Flux2Transformer2DModel wrapper for ONNX export.

    guidance is passed as None internally — not an ONNX input.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def transformer_load(model_path: str) -> FluxTransformerWrapper:
    from diffusers import Flux2Transformer2DModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    transformer.eval()
    transformer.to(device=device)
    return FluxTransformerWrapper(transformer)


# ---------------------------------------------------------------------------
# Dummy inputs
#
# Model config: in_channels=128, joint_attention_dim=7680,
#               axes_dims_rope=[32,32,32,32] → rope last dim = 4
# Resolution:   1024×1024 → img_seq_len = (1024/16)² = 4096
#
# img_ids / txt_ids are INT64 so Cast(INT64→FLOAT32) nodes are traced
# into the graph (the transformer converts position indices to float internally).
# ---------------------------------------------------------------------------

_BATCH       = 1
_IMG_SEQ_LEN = 4096
_TXT_SEQ_LEN = 256
_HIDDEN_DIM  = 128
_TXT_DIM     = 7680
_ROPE_DIMS   = 4


def transformer_conversion_inputs(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "hidden_states": torch.randn(
            _BATCH, _IMG_SEQ_LEN, _HIDDEN_DIM, dtype=torch.float32, device=device
        ),
        "encoder_hidden_states": torch.randn(
            _BATCH, _TXT_SEQ_LEN, _TXT_DIM, dtype=torch.float32, device=device
        ),
        "timestep": torch.tensor([0.5] * _BATCH, dtype=torch.float32, device=device),
        "img_ids": torch.zeros(
            _BATCH, _IMG_SEQ_LEN, _ROPE_DIMS, dtype=torch.int64, device=device
        ),
        "txt_ids": torch.zeros(
            _BATCH, _TXT_SEQ_LEN, _ROPE_DIMS, dtype=torch.int64, device=device
        ),
    }


# =============================================================================
# VAE
# =============================================================================

try:
    from diffusers import AutoencoderKLFlux2  # type: ignore
except Exception:
    AutoencoderKLFlux2 = None


# =============================================================================
# VAE Encoder
# =============================================================================

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class VaeEncoderWrapper(nn.Module):
    """AutoencoderKL encoder wrapper for ONNX export.

    Uses latent_dist.mode() for deterministic, traceable output.
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(images).latent_dist.mode()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def vae_encoder_load(model_path: str) -> VaeEncoderWrapper:
    from diffusers import AutoencoderKL

    vae_cls = AutoencoderKLFlux2 if AutoencoderKLFlux2 is not None else AutoencoderKL
    print(f"[INFO] using VAE class: {vae_cls.__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae_cls.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.eval()
    vae.to(device=device)
    return VaeEncoderWrapper(vae)


# ---------------------------------------------------------------------------
# Dummy inputs
#
# 1024×1024 RGB image input
# ---------------------------------------------------------------------------

_VAE_ENC_BATCH = 1
_VAE_ENC_C     = 3      # RGB
_VAE_ENC_H     = 1024
_VAE_ENC_W     = 1024


def vae_encoder_conversion_inputs(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "images": torch.randn(
            _VAE_ENC_BATCH, _VAE_ENC_C, _VAE_ENC_H, _VAE_ENC_W,
            dtype=torch.float32,
            device=device,
        )
    }


# =============================================================================
# VAE Decoder
# =============================================================================

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class VaeDecoderWrapper(nn.Module):
    """AutoencoderKL decoder wrapper for ONNX export."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def vae_decoder_load(model_path: str) -> VaeDecoderWrapper:
    from diffusers import AutoencoderKL

    vae_cls = AutoencoderKLFlux2 if AutoencoderKLFlux2 is not None else AutoencoderKL
    print(f"[INFO] using VAE class: {vae_cls.__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae_cls.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.eval()
    vae.to(device=device)
    return VaeDecoderWrapper(vae)


# ---------------------------------------------------------------------------
# Dummy inputs
#
# 1024×1024 output → latent spatial 128×128 (VAE 8× factor), 32 channels
# 32ch × 2×2 spatial pack = 128ch → transformer in_channels=128
# ---------------------------------------------------------------------------

_VAE_BATCH    = 1
_VAE_LATENT_C = 32   # AutoencoderKLFlux2 latent channels
_VAE_LATENT_H = 128
_VAE_LATENT_W = 128


def vae_decoder_conversion_inputs(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "latents": torch.randn(
            _VAE_BATCH, _VAE_LATENT_C, _VAE_LATENT_H, _VAE_LATENT_W,
            dtype=torch.float32,
            device=device,
        )
    }
