# -------------------------------------------------------------------------
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------


"""User script for Olive ONNX export of VideoChat-Flash VLM.

Follows the same approach as internVideo2_builder.py:
  - Uses AutoModel.from_pretrained() to load the full model
  - Extracts vision_tower and mm_projector, frees LLM backbone
  - Fixes meta-device parameters (gamma→weight naming mismatch)
  - Wraps components for ONNX export (image + video modes)
  - Uses cumsum+where in embedding merger (ONNX-compatible)

The text decoder uses ModelBuilder (no user script needed).

Architecture:
  images [B, T, 3, 224, 224]
      → InternVideo2 ViT (39 blocks, 1408-dim, 3D sincos pos-embed)
      → mm_projector (ToMe token compression + MLP 1408→3584)
      → visual_tokens [B, num_visual_tokens, 3584]

  input_ids [B, seq]  +  image_features [1, N, 3584]
      → embed_tokens(input_ids) → cumsum+where merge at <|image_pad|> positions
      → inputs_embeds [B, seq, 3584]
"""
import os
import sys
import gc
import types

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Mock video-processing libraries that modeling_videochat_flash.py imports
# via mm_utils but are not needed for ONNX export.  Force-override even if
# installed, because decord on Windows lacks VideoReader.
for _mod in ("av", "cv2", "decord", "imageio"):
    sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["decord"].VideoReader = type("VideoReader", (), {})

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Compat fix: transformers 5.x folded rope_theta into rope_parameters dict.
# The model's custom code (modeling_qwen2_flash.py) still accesses
# config.rope_theta as a top-level attribute, causing AttributeError.
# Patch __post_init__ to expose rope_theta from rope_parameters.
# In transformers <5.x, rope_theta is already a top-level attr — skip.
# ---------------------------------------------------------------------------
from transformers import Qwen2Config as _Qwen2Config

if hasattr(_Qwen2Config, "__post_init__"):
    _orig_qwen2_post_init = _Qwen2Config.__post_init__

    def _qwen2_post_init_compat(self, **kwargs):
        _orig_qwen2_post_init(self, **kwargs)
        rp = getattr(self, "rope_parameters", None)
        if isinstance(rp, dict) and "rope_theta" in rp:
            self.rope_theta = rp["rope_theta"]

    _Qwen2Config.__post_init__ = _qwen2_post_init_compat

MODEL_ID = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
IMAGE_PAD_TOKEN_ID = 151655   # <|image_pad|>
HIDDEN_SIZE = 3584
IMAGE_SIZE = 224
TOKENS_PER_IMAGE = 64
LOCAL_NUM_FRAMES = 4


# ---------------------------------------------------------------------------
# Vision wrapper (matching internVideo2_builder.py — video mode)
# ---------------------------------------------------------------------------

class VisionWithProjectorImage(nn.Module):
    """Image mode: [B, 1, 3, H, W] → [B, 64, 3584]

    compress=False → ToMe produces 64 tokens per image (no per-frame merging).
    This is what OGA's QwenImageProcessor sends (single frame per image).
    """
    def __init__(self, vision_tower, mm_projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector

    def forward(self, images):
        visual_features = self.vision_tower(images)
        projected = self.mm_projector(visual_features, compress=False)
        return projected


class VisionWithProjectorOGA(nn.Module):
    """OGA-compatible wrapper: [B, 1, 3, H, W] → [B, 64, 3584]

    Accepts single-frame input (what OGA's preprocessor sends), internally
    duplicates the frame to LOCAL_NUM_FRAMES, runs the vision encoder once
    on all frames, then applies temporal compression (ToMe16 + MLP).

    This produces the same 64-token output as VisionWithProjectorVideo
    (which the text decoder was trained on) while accepting the single-frame
    input that OGA expects.
    """
    def __init__(self, vision_tower, mm_projector, local_num_frames):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.local_num_frames = local_num_frames

    def forward(self, images):
        # images: [B, 1, 3, H, W] — single frame from OGA preprocessor
        # Duplicate to [B, T, 3, H, W] to match video-mode expectations
        images = images.repeat(1, self.local_num_frames, 1, 1, 1)
        T = self.local_num_frames
        visual_features = self.vision_tower(images)
        B = visual_features.shape[0]
        visual_features = visual_features.reshape(B * T, -1, visual_features.shape[-1])
        projected = self.mm_projector(visual_features, compress=True, local_num_frames=T)
        return projected


class VisionWithProjectorVideo(nn.Module):
    """Video mode: [B, T, 3, H, W] → [B, T*16, 3584]

    compress=True → ToMe merges 256→16 tokens per frame (16× reduction).
    For T=4: 4 frames × 16 tokens = 64 tokens total.

    For single images, repeat the frame T times at inference time.
    """
    def __init__(self, vision_tower, mm_projector, local_num_frames):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.local_num_frames = local_num_frames

    def forward(self, images):
        T = self.local_num_frames
        visual_features = self.vision_tower(images)
        B = visual_features.shape[0]
        visual_features = visual_features.reshape(B * T, -1, visual_features.shape[-1])
        projected = self.mm_projector(visual_features, compress=True, local_num_frames=T)
        return projected


# ---------------------------------------------------------------------------
# Embedding wrapper (ONNX-compatible cumsum+where approach)
# ---------------------------------------------------------------------------

class EmbeddingWithMerge(nn.Module):
    """Embedding lookup + visual feature injection at <|image_pad|> positions.

    Uses cumsum+where instead of masked_scatter for full ONNX compatibility.
    A safety row of zeros is appended to image_features so indexing never
    goes out-of-bounds (handles text-only prompts where mask is all-False).
    """
    def __init__(self, embed_weight, image_pad_id=IMAGE_PAD_TOKEN_ID):
        super().__init__()
        vocab_size, embed_dim = embed_weight.shape
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.embed_tokens.weight = nn.Parameter(embed_weight)
        self.image_pad_id = image_pad_id

    def forward(self, input_ids, image_features):
        text_embeds = self.embed_tokens(input_ids)
        hidden_size = text_embeds.shape[-1]

        mask = (input_ids == self.image_pad_id)
        indices = mask.long().cumsum(dim=-1) - 1
        indices = indices.clamp(min=0)

        flat_features = image_features.reshape(-1, hidden_size)
        safe_features = torch.cat([
            flat_features,
            torch.zeros(1, hidden_size, dtype=flat_features.dtype, device=flat_features.device)
        ], dim=0)

        visual_at_positions = torch.nn.functional.embedding(indices, safe_features)
        mask_3d = mask.unsqueeze(-1).expand_as(text_embeds)
        inputs_embeds = torch.where(mask_3d, visual_at_positions, text_embeds)
        return inputs_embeds


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _fix_meta_params(vision_tower, model_id):
    """Fix LayerScale parameters with gamma->weight name mismatch.

    The checkpoint stores these as ``ls1.gamma`` but the model code
    expects ``ls1.weight``.  ``from_pretrained()`` can't match them,
    leaving those parameters on meta device, with NaN values, or with
    default-initialized values (depending on transformers version).
    This function loads the correct tensors from safetensors shards
    for ALL LayerScale params that have a gamma->weight mismatch.
    """
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(model_id)

    vt_prefix = "model.vision_tower."
    ckpt_lookup = {}
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        shard_path = os.path.join(model_dir, fname)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(vt_prefix):
                    model_key = key[len(vt_prefix):]
                    ckpt_lookup[model_key] = (shard_path, key)

    needed = {}
    for param_name, _ in vision_tower.named_parameters():
        if param_name in ckpt_lookup:
            continue
        gamma_name = param_name.replace(".weight", ".gamma")
        if gamma_name != param_name and gamma_name in ckpt_lookup:
            shard_path, ckpt_key = ckpt_lookup[gamma_name]
            needed.setdefault(shard_path, []).append((param_name, ckpt_key))

    if not needed:
        print("  All vision tower parameters OK (no gamma->weight mismatch)")
        return

    total = sum(len(v) for v in needed.values())
    print(f"  Fixing {total} params with gamma->weight mismatch...")

    fixed = 0
    for shard_path, items in needed.items():
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for param_name, ckpt_key in items:
                tensor = f.get_tensor(ckpt_key)
                parts = param_name.rsplit(".", 1)
                parent = vision_tower
                for part in parts[0].split("."):
                    parent = getattr(parent, part)
                setattr(parent, parts[1], nn.Parameter(tensor.to(torch.float16)))
                fixed += 1

    print(f"  Fixed {fixed}/{total} params")


def _load_vision_and_projector(model_path):
    """Load the full HF model, extract vision_tower + mm_projector, free LLM backbone."""
    from transformers import AutoModel

    # transformers 5.x + torch 2.11 constructs models on meta device first,
    # but the vision tower's __init__ calls torch.linspace().item() which
    # fails on meta tensors.  Temporarily make .item() return 0.0 for meta
    # tensors — the actual values are overwritten when weights are loaded.
    _orig_item = torch.Tensor.item

    def _meta_safe_item(self):
        if self.device.type == "meta":
            return 0.0
        return _orig_item(self)

    torch.Tensor.item = _meta_safe_item

    print("  Loading full model in float16...")
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
    finally:
        torch.Tensor.item = _orig_item

    vision_tower = model.get_vision_tower()
    mm_projector = model.model.mm_projector

    # Free the LLM backbone (~7GB)
    del model.model.layers, model.model.embed_tokens, model.lm_head
    del model
    gc.collect()

    _fix_meta_params(vision_tower, model_path)
    return vision_tower, mm_projector


# ===================================================================
# Vision callbacks — IMAGE mode (1 frame, compress=False)
# ===================================================================

def get_vision_image_model(model_path=None):
    model_path = model_path or MODEL_ID
    vision_tower, mm_projector = _load_vision_and_projector(model_path)
    model = VisionWithProjectorImage(vision_tower, mm_projector)
    model.float().eval()
    return model


def get_vision_image_io_config(model_path=None):
    return {
        "input_names": ["images"],
        "output_names": ["visual_tokens"],
        "dynamic_axes": {
            "images": {0: "batch", 1: "num_frames"},
            "visual_tokens": {0: "batch", 1: "num_visual_tokens"},
        },
    }


def get_vision_image_dummy_inputs(model=None):
    return {
        "images": torch.randn(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32),
    }


# ===================================================================
# Vision callbacks — OGA mode (1 frame in, internally duplicated to 4,
# compressed via ToMe → 64 tokens, matching video-mode quality)
# ===================================================================

def get_vision_oga_model(model_path=None):
    model_path = model_path or MODEL_ID
    vision_tower, mm_projector = _load_vision_and_projector(model_path)
    model = VisionWithProjectorOGA(vision_tower, mm_projector, LOCAL_NUM_FRAMES)
    model.float().eval()
    return model


def get_vision_oga_io_config(model_path=None):
    return {
        "input_names": ["images"],
        "output_names": ["visual_tokens"],
        "dynamic_axes": {
            "images": {0: "batch"},
            "visual_tokens": {0: "batch", 1: "num_visual_tokens"},
        },
    }


def get_vision_oga_dummy_inputs(model=None):
    return {
        "images": torch.randn(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32),
    }


# ===================================================================
# Vision callbacks — VIDEO mode (T=4 frames, compress=True)
# ===================================================================

def get_vision_model(model_path=None):
    model_path = model_path or MODEL_ID
    vision_tower, mm_projector = _load_vision_and_projector(model_path)
    model = VisionWithProjectorVideo(vision_tower, mm_projector, LOCAL_NUM_FRAMES)
    model.float().eval()
    return model


def get_vision_io_config(model_path=None):
    return {
        "input_names": ["images"],
        "output_names": ["visual_tokens"],
        "dynamic_axes": {
            "images": {0: "batch", 1: "num_frames"},
            "visual_tokens": {0: "batch", 1: "num_visual_tokens"},
        },
    }


def get_vision_dummy_inputs(model=None):
    return {
        "images": torch.randn(1, LOCAL_NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32),
    }


# ===================================================================
# Embedding callbacks
# ===================================================================

def get_embedding_model(model_path=None):
    model_path = model_path or MODEL_ID
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    print("  Loading embed_tokens.weight from safetensors (fp32)...")
    model_dir = snapshot_download(model_path)

    embed_key = "model.embed_tokens.weight"
    embed_weight = None
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        shard_path = os.path.join(model_dir, fname)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if embed_key in f.keys():
                embed_weight = f.get_tensor(embed_key).float()
                print(f"  Loaded {embed_key} from {fname}: {embed_weight.shape}")
                break

    if embed_weight is None:
        raise ValueError(f"{embed_key} not found in safetensors shards")

    model = EmbeddingWithMerge(embed_weight, image_pad_id=IMAGE_PAD_TOKEN_ID)
    model.eval()
    return model


def get_embedding_io_config(model_path=None):
    return {
        "input_names": ["input_ids", "image_features"],
        "output_names": ["inputs_embeds"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "seq_len"},
            "image_features": {0: "num_images", 1: "num_image_tokens"},
            "inputs_embeds": {0: "batch", 1: "seq_len"},
        },
    }


def get_embedding_dummy_inputs(model=None):
    NUM_VISUAL_TOKENS = 64
    dummy_ids = torch.ones(1, 10 + NUM_VISUAL_TOKENS, dtype=torch.long) * 100
    dummy_ids[0, 5:5 + NUM_VISUAL_TOKENS] = IMAGE_PAD_TOKEN_ID
    dummy_features = torch.randn(1, NUM_VISUAL_TOKENS, HIDDEN_SIZE)
    return {"input_ids": dummy_ids, "image_features": dummy_features}
