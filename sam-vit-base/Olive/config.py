# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

class ModelConfig:
    model_name = "facebook/sam-vit-base"
    data_dir = "quantization_dataset"
    image_dataset = "nielsr/coco-panoptic-val2017"
    ve_input_name = "pixel_values"
    ve_sample_size = 1024
    ve_channel_size = 3
    mask_point_input_names = ("input_points", "image_embeddings")
    mask_point_input_shapes = ((1, 1, 2), (256, 64, 64))
    mask_box_input_names = ("input_boxes", "image_embeddings")
    mask_box_input_shapes = ((1, 4), (256, 64, 64))
    mask_input_names = ("input_points", "input_labels", "image_embeddings")
    mask_input_shapes = ((1, 2, 2), (1, 2), (256, 64, 64))
