import os
import sys
import torch

from transformers import Qwen3VLConfig

# Add this script's directory to sys.path to import codes module
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Import custom model from codes directory
from codes.modeling_qwen3_vl import Qwen3VLModel

model_name = "Qwen/Qwen3-VL-2B-Instruct"
config = Qwen3VLConfig.from_pretrained(model_name)


def _load_base_model(model_path):
    """Load weights directly from safetensors, stripping 'model.' prefix,
    into our custom Qwen3VLModel without loading the full HF model."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import glob

    # Find safetensors file(s) in cache
    config_path = hf_hub_download(model_path, 'config.json')
    model_dir = os.path.dirname(config_path)
    st_files = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))

    # Load and strip 'model.' prefix, keeping native bfloat16 precision
    state_dict = {}
    for sf in st_files:
        tensors = load_file(sf)
        for k, v in tensors.items():
            if k.startswith('model.'):
                state_dict[k[6:]] = v

    # Create custom model and load weights in bfloat16 (native dtype)
    custom_model = Qwen3VLModel(config)
    result = custom_model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"Warning: {len(result.missing_keys)} missing keys")
    custom_model = custom_model.to(torch.bfloat16)
    custom_model.eval()

    del state_dict
    return custom_model


### Embedding
# Dynamo export

def get_embedding_model(model_path=None):
    model = _load_base_model(model_path)
    # Export in fp32 for Olive fp16 pass compatibility (same approach as vision)
    model = model.to(torch.float32)

    model.get_fused_input_embeddings, model.forward = (
        model.forward,
        model.get_fused_input_embeddings,
    )
    return model

def get_embedding_io_config(model_path=None):
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_logical_patches"},
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
    }
    return {
        "input_names": ["input_ids", "image_features"],
        "output_names": ["inputs_embeds"],
        "dynamic_axes": dynamic_axes,
    }


def get_embedding_dummy_inputs(model=None):
    # Qwen3-VL-2B: out_hidden_size=2048, patch_size=16
    # assume 2 batches, each with 1 image input
    # With patch_size=16, spatial_merge_size=2:
    #   For a 540x360 image: grid = (1, 22, 34) -> merged = (1, 11, 17) -> 187 logical patches per image
    #   raw patches = 22*34 = 748 per frame
    batch_size, sequence_length, patches_per_image, out_hidden_size = (
        2,
        216,     # approximate sequence length with image tokens
        187,     # logical patches per image after merge (11*17)
        2048,    # Qwen3-VL-2B out_hidden_size
    )
    num_logical_patches = batch_size * patches_per_image

    # Qwen3-VL special token IDs (same as Qwen2.5-VL)
    vision_start_token_id = config.vision_start_token_id  # 151652
    vision_end_token_id = config.vision_end_token_id      # 151653
    image_token_id = config.image_token_id                # 151655

    inputs = {
        "input_ids": torch.randint(
            low=0,
            high=image_token_id,
            size=(batch_size, sequence_length),
            dtype=torch.int64,
        ),
        "image_features": torch.randn(
            num_logical_patches,
            out_hidden_size,
            dtype=torch.float32,  # fp32 to match embedding model export dtype
        ),
    }

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image

    # Fill in with image token index
    inputs["input_ids"][0][2] = vision_start_token_id  # <|vision_start|>
    inputs["input_ids"][0][
        img_start_index:img_end_index
    ] = image_token_id  # <|image_pad|>
    inputs["input_ids"][0][img_end_index] = vision_end_token_id  # <|vision_end|>

    inputs["input_ids"][1][2] = vision_start_token_id  # <|vision_start|>
    inputs["input_ids"][1][
        img_start_index:img_end_index
    ] = image_token_id  # <|image_pad|>
    inputs["input_ids"][1][img_end_index] = vision_end_token_id  # <|vision_end|>

    return {
        "input_ids": inputs["input_ids"],  # input_ids: torch.LongTensor
        "image_features": inputs["image_features"],  # image_features: Optional[torch.FloatTensor] = None,
    }


### Vision
def get_vision_model(model_path=None):
    # Export in fp32 for maximum compatibility; Olive fp16 pass converts weights.
    model = _load_base_model(model_path)
    model = model.to(torch.float32)
    model.forward, model.get_image_features = model.get_image_features, model.forward
    return model

def get_vision_io_config(model_path=None):
    # Use dynamic_shapes format (required by use_dynamo_exporter=true).
    # pixel_values dim-0 is dynamic (total patches across all images).
    # image_grid_thw is static: shape [num_images, 3] is fixed at trace time.
    return {
        "input_names": ["pixel_values", "image_grid_thw"],
        "output_names": ["image_features"],
        "dynamic_shapes": {
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": None,
        },
    }

def get_vision_dummy_inputs(model=None):
    # Qwen3-VL: patch_size=16, temporal_patch_size=2, in_channels=3
    # Each raw patch: 3 * 2 * 16 * 16 = 1536 values
    #
    # vision.onnx is exported for a SINGLE image per call.
    # Variable num_images is handled by VisionState::Run in onnxruntime-genai,
    # which calls the vision model once per image and concatenates outputs.
    #
    # For a 544×352 image: grid = (1, 22, 34) → 748 raw patches
    patches = 22 * 34   # 748

    pixel_values = torch.randn((patches, 1536), dtype=torch.float32)
    pixel_values = pixel_values * (0.95 - (-1)) + (-1)

    grid_thw = torch.tensor([[1, 22, 34]], dtype=torch.int64)

    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
