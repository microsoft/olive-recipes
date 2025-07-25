# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
import transformers
from torch import from_numpy, permute
from torch.utils.data import Dataset

from olive.data.registry import Registry

logger = getLogger(__name__)

def get_imagenet_label_map():
    import json
    cache_file = Path(f"./cache/data/imagenet_class_index.json")
    if not cache_file.exists():
        import requests        
        imagenet_class_index_url = (
            "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/imagenet_class_index.json"
        )
        response = requests.get(imagenet_class_index_url)
        response.raise_for_status()  # Ensure the request was successful
        content = response.json()
        cache_file.parent.resolve().mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(content, f)
    else:
        with open(cache_file) as f:
            content = json.loads(f.read())

    return {v[0]: int(k) for k, v in content.items()}

def adapt_label_for_mini_imagenet(labels: list, label_names: list):
    label_map = get_imagenet_label_map()
    return [label_map[label_names[x]] for x in labels]

class ImagenetDataset(Dataset):
    def __init__(self, data):
        self.images = from_numpy(data["images"])
        self.labels = from_numpy(data["labels"])

    def __len__(self):
        return min(len(self.images), len(self.labels))

    def __getitem__(self, idx):
        return {"pixel_values": self.images[idx]}, self.labels[idx]


@Registry.register_post_process()
def dataset_post_process(output):
    return (
        output.logits.argmax(axis=1)
        if isinstance(output, transformers.modeling_outputs.ModelOutput)
        else output.argmax(axis=1)
    )


from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)

@Registry.register_pre_process()
def dataset_pre_process(output_data, **kwargs):
    shuffle = kwargs.get("shuffle", True)
    if shuffle:
        seed = kwargs.get("seed", 42)
        output_data = output_data.shuffle(seed=seed)
    cache_key = kwargs.get("cache_key")
    size = kwargs.get("size", 256)
    transpose = kwargs.get("transpose", False)
    cache_file = None
    if cache_key:
        suffix = "nhwc" if transpose else "nchw"
        cache_file = Path(f"./cache/data/{cache_key}_{output_data.info.dataset_name}_{size}_{suffix}.npz")
        if cache_file.exists():
            with np.load(Path(cache_file)) as data:
                return ImagenetDataset(data)

    labels = []
    images = []
    for i, sample in enumerate(output_data):
        if i >= size:
            break
        image = sample["image"]
        label = sample["label"]
        image = image.convert("RGB")
        image = processor(image)["pixel_values"][0]
        if transpose:
            image = permute(image, (1, 2, 0))
        images.append(image)
        labels.append(label)

    if(output_data.info.dataset_name == "mini-imagenet"):
        labels = adapt_label_for_mini_imagenet(labels, output_data.features["label"].names)
    result_data = ImagenetDataset({"images": np.array(images), "labels": np.array(labels)})

    if cache_file:
        cache_file.parent.resolve().mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, images=np.array(images), labels=np.array(labels))

    return result_data