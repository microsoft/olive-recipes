import os
import sys
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from codes.modeling_qwen3_5_moe import Qwen3_5MoeModel
from transformers import AutoConfig

model_name = "Qwen/Qwen3.5-35B-A3B"
config = AutoConfig.from_pretrained(model_name)


def _load_model(model_path):
    """Load weights into the ONNX-export-friendly Qwen3_5MoeModel."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import glob

    cfg_path = hf_hub_download(model_path, "config.json")
    model_dir = os.path.dirname(cfg_path)
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    state_dict = {}
    for sf in st_files:
        tensors = load_file(sf)
        for k, v in tensors.items():
            if k.startswith("model."):
                stripped = k[6:]
                state_dict[stripped] = v
                if stripped.startswith("language_model.embed_tokens."):
                    state_dict[stripped[len("language_model."):]] = v

    custom_model = Qwen3_5MoeModel(config)
    result = custom_model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"Info: {len(result.missing_keys)} missing keys (expected -- MoE text layers not in this model)")
    if result.unexpected_keys:
        print(f"Info: {len(result.unexpected_keys)} unexpected keys (expected -- MoE expert/attention weights)")
    custom_model = custom_model.to(torch.bfloat16).eval()
    del state_dict
    return custom_model


# ── Embedding ────────────────────────────────────────────────────────────

def get_embedding_model(model_path=None):
    """Load the custom MoE model and swap forward with get_fused_input_embeddings."""
    model = _load_model(model_path or model_name)
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
    out_hidden_size = config.vision_config.out_hidden_size
    batch_size, sequence_length, patches_per_image = 2, 216, 187
    num_logical_patches = batch_size * patches_per_image

    inputs = {
        "input_ids": torch.randint(0, config.image_token_id, (batch_size, sequence_length), dtype=torch.int64),
        "image_features": torch.randn(num_logical_patches, out_hidden_size, dtype=torch.float32),
    }

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image
    for b in range(batch_size):
        inputs["input_ids"][b][2] = config.vision_start_token_id
        inputs["input_ids"][b][img_start_index:img_end_index] = config.image_token_id
        inputs["input_ids"][b][img_end_index] = config.vision_end_token_id

    return inputs


# ── Vision ───────────────────────────────────────────────────────────────

def get_vision_model(model_path=None):
    model = _load_model(model_path or model_name)
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
    patches = 22 * 34
    pixel_values = torch.randn((patches, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 22, 34]], dtype=torch.int64)
    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
