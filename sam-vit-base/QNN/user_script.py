# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys

sys.path.append(os.path.dirname(__file__))

import random
from pathlib import Path

import numpy as np
import torch
from config import ModelConfig
from datasets import load_dataset
from model_patches import (
    ModSamMaskBoxDecoder,
    ModSamMaskPointDecoder,
    ModSamVisionEncoder,
    ModSamMaskdecoder,
)
from olive.data.registry import Registry
from transformers import SamModel, SamProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseDataLoader:
    def __init__(self, total):
        self.data = []
        self.total = total

    def __getitem__(self, idx):
        if idx >= len(self.data) or idx >= self.total:
            raise StopIteration
        # print(f"Process data {idx}")
        return self.data[idx]

    def load(self, file):
        self.data.append({key: torch.from_numpy(value) for key, value in np.load(file).items()})

    def finish_load(self):
        if len(self.data) > self.total:
            self.data = random.sample(self.data, self.total)


class VeEncoderGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        ve_generate_quant_data(total)
        self.data_files = [
            os.path.join(ModelConfig.data_dir, f.name)
            for f in os.scandir(ModelConfig.data_dir)
            if "images.npz" in f.name
        ]
        self.data_files.sort()
        for f in self.data_files:
            self.load(f)
        self.finish_load()


class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype), label


def ve_inputs(batch_size, torch_dtype):
    return {
        ModelConfig.ve_input_name: torch.rand(
            (
                batch_size,
                ModelConfig.ve_channel_size,
                ModelConfig.ve_sample_size,
                ModelConfig.ve_sample_size,
            ),
            dtype=torch_dtype,
        )
    }


def random_mask_point_decoder_inputs(batch_size, torch_dtype):
    return {
        input_name: torch.rand((batch_size, *input_shape), dtype=torch_dtype)
        for input_name, input_shape in zip(ModelConfig.mask_point_input_names, ModelConfig.mask_point_input_shapes)
    }


def random_mask_box_decoder_inputs(batch_size, torch_dtype):
    return {
        input_name: torch.rand((batch_size, *input_shape), dtype=torch_dtype)
        for input_name, input_shape in zip(ModelConfig.mask_box_input_names, ModelConfig.mask_box_input_shapes)
    }


def random_mask_decoder_inputs(batch_size, torch_dtype):
    return {
        input_name: torch.rand((batch_size, *input_shape), dtype=torch_dtype)
        for input_name, input_shape in zip(ModelConfig.mask_input_names, ModelConfig.mask_input_shapes)
    }


def vision_encoder_inputs(model=None):
    return tuple(ve_inputs(1, torch.float32).values())


def mask_point_decoder_inputs(model=None):
    return tuple(random_mask_point_decoder_inputs(1, torch.float32).values())


def mask_box_decoder_inputs(model=None):
    return tuple(random_mask_box_decoder_inputs(1, torch.float32).values())


def mask_decoder_inputs(model=None):
    return tuple(random_mask_decoder_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def vision_encoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(ve_inputs, batch_size, torch.float32)


@Registry.register_dataloader()
def vision_encoder_quantize_data_loader(dataset, data_num, *args, **kwargs):
    return VeEncoderGeneratedDataLoader(data_num)


@Registry.register_dataloader()
def mask_point_decoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(mask_point_decoder_inputs, batch_size, torch.float32)


@Registry.register_dataloader()
def mask_box_decoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(mask_box_decoder_inputs, batch_size, torch.float32)


def sam_ve_load(model_name):
    model = SamModel.from_pretrained(ModelConfig.model_name)
    return ModSamVisionEncoder(model)


def sam_mask_point_decoder_load(model_name):
    model = SamModel.from_pretrained(ModelConfig.model_name)
    return ModSamMaskPointDecoder(model)


def sam_mask_box_decoder_load(model_name):
    model = SamModel.from_pretrained(ModelConfig.model_name)
    return ModSamMaskBoxDecoder(model)


def sam_mask_decoder_load(model_name):
    model = SamModel.from_pretrained(ModelConfig.model_name)
    return ModSamMaskdecoder(model)


def ve_generate_quant_data(num_samples):
    p = Path(ModelConfig.data_dir)
    if p.is_dir() and (len([f for f in p.glob("*images.npz")]) >= num_samples):
        return

    processor = SamProcessor.from_pretrained(ModelConfig.model_name)
    dataset = load_dataset(ModelConfig.image_dataset)
    dataset = dataset["train"]
    os.makedirs(ModelConfig.data_dir, exist_ok=True)
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        image = sample["image"]
        inputs = processor(image, return_tensors="np")
        pixel_values = inputs["pixel_values"]
        np.savez(f"{ModelConfig.data_dir}/input_{i}_images.npz", pixel_values=pixel_values)


def md_generate_quant_data(num_samples):
    p = Path(ModelConfig.data_dir)
    if p.is_dir() and (len([f for f in p.glob("*points.npz")]) >= num_samples):
        return

    model = SamModel.from_pretrained(ModelConfig.model_name).to(device)
    processor = SamProcessor.from_pretrained(ModelConfig.model_name)
    dataset = load_dataset(ModelConfig.image_dataset)
    dataset = dataset["train"]
    os.makedirs(ModelConfig.data_dir, exist_ok=True)
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        image = sample["image"]
        point = np.random.randint(image.size[0], (1, *(ModelConfig.mask_input_shapes[0])))
        labels = np.random.randint(-1, 3, (1, *(ModelConfig.mask_input_shapes[1])))
        inputs = processor(image, input_points=point.tolist(), input_labels=labels.tolist(), return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        input_points = inputs["input_points"].detach().cpu().numpy()
        input_labels = inputs["input_labels"].detach().cpu().numpy()
        image_embeddings = (
            model.vision_encoder(pixel_values=pixel_values.to(device)).last_hidden_state.detach().cpu().numpy()
        )
        np.savez(
            f"{ModelConfig.data_dir}/input_{i}_points.npz",
            input_points=input_points,
            input_labels=input_labels,
            image_embeddings=image_embeddings,
        )
