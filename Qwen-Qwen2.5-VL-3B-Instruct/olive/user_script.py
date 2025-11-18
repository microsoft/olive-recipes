import argparse
import os
import sys
import torch
import shutil

import onnx
from onnxscript.rewriter import ort_fusions
from transformers import Qwen2_5_VLConfig
from torch.onnx._internal.exporter import _testing

import onnxscript
import onnx_ir as ir
from typing import Sequence
import onnx_ir.passes.common as common_passes

# Add parent directory to sys.path to import codes module
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import custom model from codes directory
from codes.modeling_qwen2_5_vl import Qwen2_5_VLModel

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
config = Qwen2_5_VLConfig.from_pretrained(model_name)


### Embedding
# Dynamo export

def get_embedding_model(model_path=None):
    model = Qwen2_5_VLModel.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model.get_fused_input_embeddings, model.forward = (
        model.forward,
        model.get_fused_input_embeddings,
    )
    return model

def get_embedding_io_config(model_path=None):
    dynamic_shapes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_logical_patches"},
    }
    return {
        "input_names": ["input_ids", "image_features"],
        "output_names": ["inputs_embeds"],
        "dynamic_shapes": dynamic_shapes,
    }


def get_embedding_dummy_inputs(model=None):
    # assume 2 batches, each with 1 image input (3577 logical patches)
    batch_size, sequence_length, patches_per_image, out_hidden_size = (
        2,
        3606,
        3577,
        3584,
    )
    num_logical_patches = batch_size * patches_per_image

    inputs = {
        "input_ids": torch.randint(
            low=0,
            high=config.image_token_id,
            size=(batch_size, sequence_length),
            dtype=torch.int64,
        ),
        "image_features": torch.randn(
            num_logical_patches,
            out_hidden_size,
            dtype=torch.float32,
        ),
    }

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image  # 3 + 3577 = 3580

    # Fill in with image token index
    inputs["input_ids"][0][2] = config.bos_token_id  # <start_of_image>
    inputs["input_ids"][0][
        img_start_index:img_end_index
    ] = config.image_token_id  # <image>
    inputs["input_ids"][0][img_end_index] = config.eos_token_id  # <end_of_image>

    inputs["input_ids"][1][2] = config.bos_token_id  # <start_of_image>
    inputs["input_ids"][1][
        img_start_index:img_end_index
    ] = config.image_token_id  # <image>
    inputs["input_ids"][1][img_end_index] = config.eos_token_id  # <end_of_image>

    return {
        "input_ids": inputs["input_ids"],  # input_ids: torch.LongTensor
        "image_features": inputs["image_features"],  # image_features: Optional[torch.FloatTensor] = None,
    }


### Vision
def get_vision_model(model_path=None):
    model = Qwen2_5_VLModel.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.forward, model.get_image_features = model.get_image_features, model.forward
    return model

def get_vision_io_config(model_path=None):
    return {
        "input_names": ["pixel_values", "image_grid_thw"],
        "output_names": ["image_features"],
        "dynamic_shapes": {"pixel_values": {0: "num_patches"}, "image_grid_thw": None},
    }

def get_vision_dummy_inputs(model=None):
    pixel_values = torch.randn((14308, 1176), dtype=torch.float32)
    # Scale the values to the range [-1, 0.95] to fit actual values we observed in the example.
    pixel_values = pixel_values * (0.95 - (-1)) + (-1)
    pixel_values = pixel_values.to(torch.float32)

    grid_thw = torch.tensor([[1, 98, 146]], dtype=torch.int64)

    # Dynamo export
    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
