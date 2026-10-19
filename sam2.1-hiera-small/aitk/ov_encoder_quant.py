import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from olive.data.registry import Registry
from ov_inference_utils import resize_normalize_adjust_dim
from PIL import Image


class COCOLoader(torch.utils.data.Dataset):
    _INPUT_IMAGE_SIZE = 1024

    def __init__(self, images_path):
        self.images = sorted(Path(images_path).iterdir())

    def __getitem__(self, index):
        image_path = self.images[index]
        with Image.open(image_path) as image:
            pixels, _ = resize_normalize_adjust_dim(image, self._INPUT_IMAGE_SIZE)
            return pixels[0]

    def __len__(self):
        return len(self.images)


@Registry.register_dataset()
def coco128_encoder_dataset(images_path, **kwargs):
    """
    Prepare the COCO128 dataset for SAM2.1 Vision Encoder Quantization.

    Args:
        images_path (str): Path to the directory containing COCO128 images.
        kwargs: Additional keyword arguments.

    Returns:
        List of preprocessed images ready for quantization.
    """
    return COCOLoader(images_path)


def encoder_transform_fn(image_data):
    """
    Quantization transform function for SAM2.1 Vision Encoder.

    Extracts and converts input data from DataLoader for NNCF quantization.

    Args:
        image_data: Batched image tensor from DataLoader, shape [1, 3, 1024, 1024].

    Returns:
        Dict with "image" key mapped to numpy array for model calibration.
    """
    return {"image": image_data.numpy()}
