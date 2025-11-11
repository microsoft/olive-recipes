from io import BytesIO
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from tqdm import tqdm
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
import tarfile
import zipfile
from olive.data.registry import Registry
from torch.utils.data import Dataset
import numpy as np

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# -------------------------------------------------------------------------
# Common Dataset
# -------------------------------------------------------------------------

seed = 0
# seed everything to 0 for reproducibility, https://pytorch.org/docs/stable/notes/randomness.html
# do not set random seed and np.random.seed for aml test, since it will cause aml job name conflict
torch.manual_seed(seed)
# the following are needed only for GPU
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

COCO_CN_CACHE_FOLDER = Path("cache/coco_cn")

# Download val and train from https://cocodataset.org/#download and set it to the zip file
# AIMClab-RUC/COCO-CN's train, test, val use a mixture of train and validation of ms-coco
COCO_VAL2014_PATH = None
COCO_TRAIN2014_PATH = None


class ZipManager:
    def __init__(self, zip_path=None):
        self.zip_path = zip_path
        self.zip_file = None
        self.is_open = False

    def open(self):
        """Manually open ZIP file"""
        if self.zip_path and Path(self.zip_path).exists() and not self.is_open:
            try:
                self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
                self.is_open = True
                print(f"Opened ZIP: {self.zip_path}")
                return True
            except Exception as e:
                print(f"Error opening ZIP: {e}")
                return False
        return False

    def close(self):
        """Manually close ZIP file"""
        if self.is_open and self.zip_file:
            try:
                self.zip_file.close()
                self.is_open = False
                print(f"Closed ZIP: {self.zip_path}")
            except Exception as e:
                print(f"Error closing ZIP: {e}")

    def read_file(self, filename):
        """Read file from ZIP"""
        if not self.is_open:
            if not self.open():
                return None
        if not self.zip_file:
            return None
        try:
            with self.zip_file.open(filename, 'r') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None

    def load_image(self, filename):
        """Load PIL Image from ZIP"""
        image_data = self.read_file(filename)
        if image_data:
            try:
                return Image.open(BytesIO(image_data)).convert('RGB')
            except Exception as e:
                print(f"Error creating image from {filename}: {e}")
        return None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def wrap_collate_fn(processor, max_length):
    def collate_fn(image, chinese_caption: str):
        """Preprocess an example by loading and transforming image and text data.
        """
        inputs = processor(text=chinese_caption, images=[image], return_tensors="pt", padding=True)
        if inputs["input_ids"].shape[1] > max_length:
            return None
        return inputs

    return collate_fn


def prepare_calibration_data(dataloader: list[list[str]], init_steps: int, collate_fn):
    """Prepare calibration data from a dataloader for a specified number of initialization steps.

    Iterate over the dataloader, fetching batches and storing the relevant data.
    """
    with ZipManager(COCO_VAL2014_PATH) as cocoVal, ZipManager(COCO_TRAIN2014_PATH) as cocoTrain:
        data = []
        with tqdm(total=init_steps) as pbar:
            for imageId, caption in dataloader:
                image = None
                if cocoVal.is_open and imageId.startswith("COCO_val2014_"):
                    image = cocoVal.load_image("val2014/" + imageId + ".jpg")
                elif cocoTrain.is_open and imageId.startswith("COCO_train2014_"):
                    image = cocoTrain.load_image("train2014/" + imageId + ".jpg")
                if image is None:
                    continue
                # print(f"Processing image {imageId} with caption: {caption}")
                batch = collate_fn(image, caption)
                if batch:
                    pbar.update(1)
                    with torch.no_grad():
                        data.append(
                            {
                                "input_ids": batch["input_ids"].to("cpu"),
                                "pixel_values": batch["pixel_values"].to("cpu"),
                                "attention_mask": batch["attention_mask"].to("cpu"),
                            }
                        )
                        if len(data) == init_steps:
                            break
        return data


# Return [id, caption] pairs
def get_coco_cn(target_folder, split="train") -> list[list[str]]:
    if not target_folder.exists():
        """Extract tar.gz files from a Hugging Face dataset"""

        # Download only the tar.gz file
        print("Downloading coco-cn-version1805v1.1.tar.gz...")
        tar_path = hf_hub_download(
            repo_id="AIMClab-RUC/COCO-CN",
            filename="coco-cn-version1805v1.1.tar.gz",
            repo_type="dataset"
        )

        # Extract the tar.gz file
        print(f"Extracting to {target_folder}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=target_folder)

        print("Extraction completed!")

    with open(target_folder / "coco-cn-version1805v1.1" / f"coco-cn_{split}.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(target_folder / "coco-cn-version1805v1.1" / "imageid.human-written-caption.txt", "r", encoding="utf-8") as f:
        images_lines = f.readlines()
    image_caption_dict = {}
    for line in images_lines:
        # TODO add #1 etc.
        subs = line.strip().split("#0\t")
        if len(subs) != 2:
            continue
        image_id, caption = subs
        image_caption_dict[image_id] = caption
    result = []
    for line in lines:
        line = line.strip()
        if line in image_caption_dict:
            result.append([line, image_caption_dict[line]])
    print(f"Loaded {len(result)} captions from COCO-CN {split} set.")
    return result


@Registry.register_dataset()
def conceptual_captions_dataset(data_name, opt_init_steps=200, **kwargs):
    """Prepare a vision-text dataset for quantization."""
    if data_name != "AIMClab-RUC/COCO-CN":
        raise ValueError(
            f"Unsupported value for 'data_name': {data_name}. Only 'AIMClab-RUC/COCO-CN' is supported for this dataset."
        )
    if COCO_VAL2014_PATH is None or COCO_TRAIN2014_PATH is None:
        raise ValueError(
            "Missing required MS-COCO dataset zip file paths: 'COCO_VAL2014_PATH' and/or 'COCO_TRAIN2014_PATH' are not set. "
            "Please set both variables to the correct zip file paths before loading the dataset."
        )

    coco_cn = get_coco_cn(COCO_CN_CACHE_FOLDER)

    model_path = kwargs.get("model_path")
    if not model_path:
       raise ValueError(
            "The 'model_path' parameter is required in data_configs.load_dataset_config but was not provided."
        )
    model = ChineseCLIPModel.from_pretrained(model_path)
    processor = ChineseCLIPProcessor.from_pretrained(model_path)
    max_length = model.config.text_config.max_position_embeddings
    collate_fn = wrap_collate_fn(processor, max_length)
    # TODO shuffle coco_cn
    return prepare_calibration_data(coco_cn, opt_init_steps, collate_fn)


def custom_transform_func(data_item):
    np_inputs = {}
    for inp in data_item:
        # Drop the first dimension using slicing
        np_inputs[inp] = data_item[inp].numpy()[0, ...]
    return np_inputs


# Evaluation

class CLIPDataset(Dataset):
    def __init__(
        self,
        model_name,
        dataset_name,
        start=0,
        length=500,
        image_size=(224, 224),
    ):
        if dataset_name != "AIMClab-RUC/COCO-CN":
            raise ValueError(f"Unsupported data_name: {dataset_name}")

        assert 0 <= start
        assert length > 0
        end = start + length

        self.start = start
        self.end = end
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_name)
        self.length = length
        self.image_size = image_size
        self.coco_cn = get_coco_cn(COCO_CN_CACHE_FOLDER)
        self.cocoVal = ZipManager(COCO_VAL2014_PATH)
        self.cocoVal.open()
        assert self.cocoVal.is_open
        self.cocoTrain = ZipManager(COCO_TRAIN2014_PATH)
        self.cocoTrain.open()
        assert self.cocoTrain.is_open

    def __del__(self):
        # Ensure ZIP files are closed when the object is deleted
        if hasattr(self, "cocoVal") and self.cocoVal is not None:
            self.cocoVal.close()
        if hasattr(self, "cocoTrain") and self.cocoTrain is not None:
            self.cocoTrain.close()
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        text = [item[1] for item in self.coco_cn[self.start + idx : self.start + idx + 10]]
        text_inputs = self.processor(
            text=text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            # the model has no predefined maximum length
            max_length=77,
        )

        image_id = self.coco_cn[self.start + idx][0]
        if image_id.startswith("COCO_val2014_"):
            image = self.cocoVal.load_image("val2014/" + image_id + ".jpg")
        elif image_id.startswith("COCO_train2014_"):
            image = self.cocoTrain.load_image("train2014/" + image_id + ".jpg")
        image_input = self.processor(images=image.resize(self.image_size), return_tensors="np")
        model_inputs = [
            {
                "input_ids": text_inputs["input_ids"].astype(np.int64),
                "pixel_values": image_input["pixel_values"],
                "attention_mask": text_inputs["attention_mask"].astype(np.int64),
            }
        ]

        target = torch.Tensor([0]).to(torch.int32)
        return model_inputs[0], target


@Registry.register_dataset()
def clip_dataset(**kwargs):
    return CLIPDataset(**kwargs)


@Registry.register_post_process()
def clip_post_process(output):
    return output["logits_per_image"].argmax(axis=-1)
