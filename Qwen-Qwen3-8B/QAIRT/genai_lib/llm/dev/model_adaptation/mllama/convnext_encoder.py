#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch
import torch.nn as nn
import argparse
from transformers import CLIPImageProcessor
from transformers import ConvNextModel, ConvNextConfig
from transformers.models.convnext.modeling_convnext import ConvNextStage


class ConvNeXtCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower,
                        mm_vision_select_layer, mm_vision_resolution,
                        vision_add_five_stage, vision_five_stage_width,
                        delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = mm_vision_select_layer
        self.update_resolution = mm_vision_resolution
        self.vision_add_five_stage = vision_add_five_stage
        self.vision_five_stage_width = vision_five_stage_width

        if not delay_load:
            self.load_model()
        else:
            print(f"deloy_load vision tower is: {self.vision_tower_name}")
            self.cfg_only = ConvNextConfig.from_pretrained(
                self.vision_tower_name)

    def load_model(self):
        print(f"entering load model, load {self.vision_tower_name}")
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name)
        self.vision_tower = ConvNextModel.from_pretrained(
            self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

        if self.select_layer == -2:
            self.select_layer = -1
            self.vision_tower.encoder.stages[-1].layers.pop(-1)
            print(
                f'Last block removed, select layer changed to {self.select_layer}')

        if self.update_resolution > 256:
            self.set_crop_size(self.update_resolution)
            print(
                f'Crop size changed to {self.update_resolution}x{self.update_resolution}')

        if self.vision_add_five_stage != 0:
            self.add_stage(self.vision_add_five_stage, self.vision_five_stage_width)
            print(
                f'Added stage with width {self.vision_five_stage_width}')

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # Get the embeddings of the image
                embedding_output = self.vision_tower.embeddings(image.unsqueeze(0))

                # Get the image features
                image_feature = self.vision_tower.encoder(embedding_output,
                                                        output_hidden_states=True,
                                                        return_dict=True)
                image_features = image_features.hidden_states[-1].permute(0, 2, 3, 1)
                image_features = image_features.reshape(image_features.shape[0], -1, image_features.shape[3]).to(images.dtype)

                image_features.append(image_feature)
        else:
            embedding_output = self.vision_tower.embeddings(images)
            image_features = self.vision_tower.encoder(embedding_output,
                                                       output_hidden_states=True,
                                                       return_dict=True)
            image_features = image_features.hidden_states[-1].permute(0, 2, 3, 1)
            image_features = image_features.reshape(image_features.shape[0], -1, image_features.shape[3]).to(images.dtype)

        return image_features

    def forward_multiscales(self, images):
        if type(images) is list:
            raise NotImplementedError
        else:
            embedding_output = self.vision_tower.embeddings(images)
            image_features = self.vision_tower.encoder(embedding_output,
                                                       output_hidden_states=True,
                                                       return_dict=True)
            list_image_features = []
            image_features_m3 = image_features.hidden_states[-3].permute(0, 2, 3, 1)
            image_features_m3 = image_features_m3.reshape(image_features_m3.shape[0], -1, image_features_m3.shape[3]).to(images.dtype)
            image_features_m2 = image_features.hidden_states[-2].permute(0, 2, 3, 1)
            image_features_m2 = image_features_m2.reshape(image_features_m2.shape[0], -1, image_features_m2.shape[3]).to(images.dtype)
            image_features_m1 = image_features.hidden_states[-1].permute(0, 2, 3, 1)
            image_features_m1 = image_features_m1.reshape(image_features_m1.shape[0], -1, image_features_m1.shape[3]).to(images.dtype)
            list_image_features = [image_features_m3, image_features_m2, image_features_m1]

        return list_image_features

    def set_crop_size(self, new_size):
        size_dict = {'height': new_size, 'width': new_size}
        self.image_processor.crop_size = size_dict
        self.image_processor.size = {"shortest_edge": new_size}
        self.vision_tower.config.image_size = new_size

    def add_stage(self, depths=3, hidden_dims=3072):
        self.vision_tower.encoder.stages.append(ConvNextStage(self.config, self.hidden_size, hidden_dims, depth=depths))
        self.vision_tower.config.depths.append(depths)
        self.vision_tower.config.hidden_sizes.append(hidden_dims)
        self.vision_tower.config.stage_names.append('stage5')
        self.vision_tower.config.out_features = ['stage5']
        self.vision_tower.config.out_indices = [5]
        self.vision_tower.config.num_stages += 1
        self.vision_tower.config._name_or_path = ''

    def save_config(self, path):
        self.vision_tower.config.save_pretrained(path)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_sizes[-1]

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // 32) ** 2

    @property
    def crop_size(self):
        return self.image_processor.crop_size
