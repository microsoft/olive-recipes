from pathlib import Path

import numpy as np
import torch
from olive.data.registry import Registry


class COCODecoderDataset(torch.utils.data.Dataset):
    _INPUT_IMAGE_SIZE = 1024
    _N_POINTS = 5

    def __init__(self, decoder_inputs_dir):
        self.decoder_inputs = sorted(Path(decoder_inputs_dir).glob("decoder_input_*.npz"))

    def __getitem__(self, index):
        input_path = self.decoder_inputs[index]
        data = np.load(input_path)
        return {
            "image_embeddings": data["image_embeddings"][0],  # shape [256, 64, 64]
            "high_res_feats_256": data["high_res_feats_256"][0],  # shape [32, 256, 256]
            "high_res_feats_128": data["high_res_feats_128"][0],  # shape [64, 128, 128]
            "point_coords": (torch.rand(self._N_POINTS, 2) * self._INPUT_IMAGE_SIZE).numpy(),  # shape [N_POINTS, 2]
            "point_labels": torch.ones(self._N_POINTS).numpy(),  # shape [N_POINTS]
        }

    def __len__(self):
        return len(self.decoder_inputs)


@Registry.register_dataset()
def coco128_decoder_dataset(decoder_inputs_dir, **kwargs):
    """
    Prepare the COCO128 dataset for SAM2.1 Mask Decoder Quantization.

    Args:
        decoder_inputs_dir (str): Path to the directory containing COCO128 decoder inputs.
        kwargs: Additional keyword arguments.

    Returns:
        List of preprocessed decoder inputs ready for quantization.
    """
    return COCODecoderDataset(decoder_inputs_dir)


def decoder_transform_fn(data_item):
    """
    Quantization transform function for SAM2.1 Mask Decoder.

    Extracts and converts input data from DataLoader for NNCF quantization.

    Args:
        data_item: Batched dict from DataLoader with decoder input tensors.

    Returns:
        Dict with numpy arrays for model calibration.
    """
    return {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in data_item.items()}
