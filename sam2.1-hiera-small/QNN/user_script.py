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
from olive.data.registry import Registry
from transformers import Sam2Processor


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
            (batch_size, ModelConfig.ve_channel_size, ModelConfig.ve_sample_size, ModelConfig.ve_sample_size),
            dtype=torch_dtype,
        )
    }


def md_inputs(batch_size, torch_dtype):
    return {
        input_name: torch.rand((batch_size, *input_shape), dtype=torch_dtype)
        for input_name, input_shape in zip(ModelConfig.md_input_names, ModelConfig.ms_input_shapes)
    }


def ve_conversion_inputs(model=None):
    return tuple(ve_inputs(1, torch.float32).values())


def md_conversion_inputs(model=None):
    return tuple(md_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def ve_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(ve_inputs, batch_size, torch.float32)


@Registry.register_dataloader()
def ve_quantize_data_loader(dataset, data_num, *args, **kwargs):
    return VeEncoderGeneratedDataLoader(data_num)


@Registry.register_dataloader()
def md_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(md_inputs, batch_size, torch.float32)


def ve_generate_quant_data(num_samples):
    p = Path(ModelConfig.data_dir)
    if p.is_dir() and (len([f for f in p.glob("*images.npz")]) >= num_samples):
        return

    processor = Sam2Processor.from_pretrained(ModelConfig.model_name)
    dataset = load_dataset("nielsr/coco-panoptic-val2017")
    dataset = dataset["train"]
    os.makedirs(ModelConfig.data_dir, exist_ok=True)
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        image = sample["image"]
        inputs = processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].detach().cpu().numpy()
        np.savez(f"{ModelConfig.data_dir}/input_{i}_images.npz", pixel_values=pixel_values)
