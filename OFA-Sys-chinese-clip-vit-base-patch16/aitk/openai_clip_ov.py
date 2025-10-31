from io import BytesIO
from pathlib import Path

import requests
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from tqdm import tqdm
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
import tarfile

from olive.data.registry import Registry

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


def wrap_collate_fn(processor, max_length, coco_ms_train, coco_ms_val):
    def collate_fn(image_name: str, chinese_caption: str):
        """Preprocess an example by loading and transforming image and text data.
        """
        coco_ms = coco_ms_train if "COCO_train" in image_name else coco_ms_val
        filtered = coco_ms.filter(lambda x: x['filename'] == image_name + '.jpg')
        print(f"Filtered length for {image_name}: {len(filtered)}")
        if len(filtered) == 0:
            return None
        image = filtered[0]['image']
        inputs = processor(text=chinese_caption, images=[image], return_tensors="pt", padding=True)
        if inputs["input_ids"].shape[1] > max_length:
            return None
        return inputs

    return collate_fn


def prepare_calibration_data(dataloader, init_steps, collate_fn):
    """Prepare calibration data from a dataloader for a specified number of initialization steps.

    Iterate over the dataloader, fetching batches and storing the relevant data.
    """
    data = []
    with tqdm(total=init_steps) as pbar:
        for data in dataloader:
            if len(data) == init_steps:
                break
            batch = collate_fn(data[0], data[1])
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
    return data


def get_coco_cn(target_folder, split="train"):
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
        if line.strip() in image_caption_dict:
            result.append([line, image_caption_dict[line.strip()]])
    print(f"Loaded {len(result)} captions from COCO-CN {split} set.")
    return result


@Registry.register_dataset()
def conceptual_captions_dataset(data_name, opt_init_steps=40, **kwargs):
    """Prepare a vision-text dataset for quantization."""
    if data_name != "AIMClab-RUC/COCO-CN":
        raise ValueError(f"Unsupported data_name: {data_name}. Only 'AIMClab-RUC/COCO-CN' is supported.")

    target_folder = Path("cache/coco_cn")
    coco_cn = get_coco_cn(target_folder)
    # most rows of ms-coco in hf
    # only train and val are used in coco-cn
    coco_ms_train = load_dataset("bitmind/MS-COCO", split="train")
    coco_ms_val = load_dataset("bitmind/MS-COCO", split="validation")

    model_path = kwargs.get("model_path")
    if not model_path:
        raise ValueError(
            "The 'model_path' parameter is required in data_configs.load_dataset_config but was not provided."
        )
    model = ChineseCLIPModel.from_pretrained(model_path)
    processor = ChineseCLIPProcessor.from_pretrained(model_path)
    max_length = model.config.text_config.max_position_embeddings
    collate_fn = wrap_collate_fn(processor, max_length, coco_ms_train, coco_ms_val)
    # TODO shuffle coco_cn
    return prepare_calibration_data(coco_cn, opt_init_steps, collate_fn)
