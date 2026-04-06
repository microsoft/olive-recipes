import os
import sys
import torch

from transformers import Qwen3_5Config

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from codes.modeling_qwen3_5 import Qwen3_5Model

model_name = "Qwen/Qwen3.5-4B"
config = Qwen3_5Config.from_pretrained(model_name)


def _load_base_model(model_path):
    """Load weights from safetensors into custom Qwen3_5Model."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import glob

    config_path = hf_hub_download(model_path, "config.json")
    model_dir = os.path.dirname(config_path)
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    state_dict = {}
    for sf in st_files:
        tensors = load_file(sf)
        for k, v in tensors.items():
            if k.startswith("model."):
                stripped = k[6:]
                state_dict[stripped] = v
                # Map language_model.embed_tokens -> embed_tokens for our flat model
                if stripped.startswith("language_model.embed_tokens."):
                    state_dict[stripped[len("language_model."):]] = v

    custom_model = Qwen3_5Model(config)
    result = custom_model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"Warning: {len(result.missing_keys)} missing keys")
    custom_model = custom_model.to(torch.bfloat16)
    custom_model.eval()

    del state_dict
    return custom_model


# ── Embedding ────────────────────────────────────────────────────────────

def get_embedding_model(model_path=None):
    model = _load_base_model(model_path)
    model = model.to(torch.float32)
    model.get_fused_input_embeddings, model.forward = (
        model.forward,
        model.get_fused_input_embeddings,
    )
    return model


def get_embedding_io_config(model_path=None):
    return {
        "input_names": ["input_ids", "image_features"],
        "output_names": ["inputs_embeds"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "image_features": {0: "num_logical_patches"},
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        },
    }


def get_embedding_dummy_inputs(model=None):
    # Qwen3.5-4B: out_hidden_size=2560, patch_size=16
    batch_size, sequence_length, patches_per_image, out_hidden_size = (
        2, 216, 187, 2560,
    )
    num_logical_patches = batch_size * patches_per_image

    vision_start_token_id = config.vision_start_token_id
    vision_end_token_id = config.vision_end_token_id
    image_token_id = config.image_token_id

    inputs = {
        "input_ids": torch.randint(0, image_token_id, (batch_size, sequence_length), dtype=torch.int64),
        "image_features": torch.randn(num_logical_patches, out_hidden_size, dtype=torch.float32),
    }

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image

    for b in range(batch_size):
        inputs["input_ids"][b][2] = vision_start_token_id
        inputs["input_ids"][b][img_start_index:img_end_index] = image_token_id
        inputs["input_ids"][b][img_end_index] = vision_end_token_id

    return inputs


# ── Vision ───────────────────────────────────────────────────────────────

def get_vision_model(model_path=None):
    model = _load_base_model(model_path)
    model = model.to(torch.float32)
    model.forward, model.get_image_features = model.get_image_features, model.forward
    return model


def get_vision_io_config(model_path=None):
    return {
        "input_names": ["pixel_values", "image_grid_thw"],
        "output_names": ["image_features"],
        "dynamic_shapes": {
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": None,
        },
    }


def get_vision_dummy_inputs(model=None):
    # patch_size=16, temporal_patch_size=2, in_channels=3
    # patch dim: 3 * 2 * 16 * 16 = 1536
    # For 544x352 image: grid=(1, 22, 34), 748 raw patches
    patches = 22 * 34
    pixel_values = torch.randn((patches, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 22, 34]], dtype=torch.int64)
    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
