import math
import os
import glob

import torch
import torch.nn as nn
from transformers import AutoConfig

_default_model_id = "google/translategemma-4b-it"

config = AutoConfig.from_pretrained(_default_model_id, trust_remote_code=True)

IMAGE_TOKEN_ID = config.image_token_index  # 262144
HIDDEN_SIZE = config.text_config.hidden_size  # 2560
VISION_HIDDEN_SIZE = config.vision_config.hidden_size  # 1152
IMAGE_SIZE = config.vision_config.image_size  # 896
PATCH_SIZE = config.vision_config.patch_size  # 14
MM_TOKENS_PER_IMAGE = config.mm_tokens_per_image  # 256
VOCAB_SIZE = config.text_config.vocab_size  # 262208


# ── Vision wrapper ────────────────────────────────────────────────────────


class Gemma3VisionModel(nn.Module):
    """Wraps vision_tower + multi_modal_projector into a single exportable module.

    pixel_values [batch, 3, 896, 896] -> image_features [batch*256, 2560]
    Output is flattened to 2D so it can be directly consumed by the embedding model.
    """

    def __init__(self, vision_tower, projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True)
        last_hidden_state = vision_outputs.last_hidden_state
        image_features = self.projector(last_hidden_state)
        # Flatten [batch, tokens_per_image, hidden] -> [batch*tokens_per_image, hidden]
        return image_features.reshape(-1, image_features.shape[-1])


def get_vision_model(model_path=None):
    from transformers import AutoModel

    model_id = model_path or _default_model_id
    full_model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    wrapper = Gemma3VisionModel(full_model.vision_tower, full_model.multi_modal_projector)
    wrapper = wrapper.to(torch.float32).eval()
    del full_model
    return wrapper


def get_vision_io_config(model_path=None):
    return {
        "input_names": ["pixel_values"],
        "output_names": ["image_features"],
        "dynamic_axes": {
            "pixel_values": {0: "num_images"},
            "image_features": {0: "num_image_tokens"},
        },
    }


def get_vision_dummy_inputs(model=None):
    return {
        "pixel_values": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32),
    }


# ── Embedding wrapper ────────────────────────────────────────────────────


class Gemma3EmbeddingModel(nn.Module):
    """Wraps embed_tokens (with scale) + image-feature scattering.

    input_ids [batch, seq_len] + image_features [total_image_tokens, hidden_size]
    -> inputs_embeds [batch, seq_len, hidden_size]
    """

    def __init__(self, embed_tokens: nn.Module, image_token_id: int):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.image_token_id = image_token_id

    def forward(
        self, input_ids: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        # Replace image_token_id with 0 to avoid OOB in embedding lookup
        mask = input_ids == self.image_token_id
        safe_ids = torch.where(mask, torch.zeros_like(input_ids), input_ids)

        # embed_tokens already applies * sqrt(hidden_size) via Gemma3TextScaledWordEmbedding
        inputs_embeds = self.embed_tokens(safe_ids)

        # Scatter image features into image-token positions
        mask_3d = mask.unsqueeze(-1).expand_as(inputs_embeds)
        image_features = image_features.to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(mask_3d, image_features)

        return inputs_embeds


def get_embedding_model(model_path=None):
    from transformers import AutoModel

    model_id = model_path or _default_model_id
    full_model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    embed_tokens = full_model.language_model.embed_tokens
    wrapper = Gemma3EmbeddingModel(embed_tokens, IMAGE_TOKEN_ID)
    wrapper = wrapper.to(torch.float32).eval()
    del full_model
    return wrapper


def get_embedding_io_config(model_path=None):
    return {
        "input_names": ["input_ids", "image_features"],
        "output_names": ["inputs_embeds"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "image_features": {0: "num_image_tokens"},
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        },
    }


def get_embedding_dummy_inputs(model=None):
    batch_size = 1
    seq_len = 1 + 10 + 1 + MM_TOKENS_PER_IMAGE + 1 + 5  # 274
    total_image_tokens = batch_size * MM_TOKENS_PER_IMAGE

    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), dtype=torch.int64)

    img_start = 12
    img_end = img_start + MM_TOKENS_PER_IMAGE
    input_ids[:, img_start:img_end] = IMAGE_TOKEN_ID

    image_features = torch.randn(total_image_tokens, HIDDEN_SIZE, dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "image_features": image_features,
    }
