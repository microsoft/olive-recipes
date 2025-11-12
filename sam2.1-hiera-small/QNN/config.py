# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


class ModelConfig:
    model_name = "facebook/sam2.1-hiera-small"
    data_dir = "quantization_dataset_100"
    ve_input_name = "pixel_values"
    ve_channel_size = 3
    ve_sample_size = 1024
    md_input_names = ("image_embeddings", "high_res_features1", "high_res_features2", "coords.1", "labels")
    ms_input_shapes = ((1, 256, 64, 64), (1, 32, 256, 256), (1, 64, 128, 128), (1, 5, 2), (1, 5))
