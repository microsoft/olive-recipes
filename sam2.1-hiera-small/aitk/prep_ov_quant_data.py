import os
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from ov_encoder_quant import COCOLoader
from ov_model_utils import load_sam21_image_encoder
from tqdm import tqdm

COCO_URL = "https://ultralytics.com/assets/coco128.zip"
COCO_ROOT = Path("quantization_dataset")
COCO_ZIP = COCO_ROOT / "coco128.zip"
COCO_DIR = COCO_ROOT / "coco128" / "images" / "train2017"
COCO_DECODER_INPUTS_DIR = Path(COCO_ROOT / "decoder_inputs")


def _safe_extract(zf, member, target_dir):
    """Extract a zip member after verifying it doesn't escape target_dir (Zip Slip guard)."""
    target = os.path.realpath(os.path.join(target_dir, member.filename))
    if not target.startswith(os.path.realpath(str(target_dir)) + os.sep) and target != os.path.realpath(
        str(target_dir)
    ):
        raise ValueError(f"Zip path traversal detected: {member.filename}")
    zf.extract(member, target_dir)


def download_and_extract_coco128():
    """Download and extract COCO128 into the cwd if not already present."""
    if COCO_DIR.exists():
        print(f"COCO128 already exists at {COCO_DIR}, skipping download and extraction.")
        return

    if not COCO_ZIP.exists():
        COCO_ROOT.mkdir(parents=True, exist_ok=True)
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {COCO_URL.rsplit('/', 1)[-1]}",
            colour="cyan",
        ) as bar:

            def _hook(block_num, block_size, total_size):
                if total_size > 0 and bar.total != total_size:
                    bar.total = total_size
                bar.update(block_num * block_size - bar.n)

            urllib.request.urlretrieve(COCO_URL, COCO_ZIP, reporthook=_hook)

    else:
        print(f"{COCO_ZIP} already exists, skipping download.")

    with ZipFile(COCO_ZIP, "r") as zf:
        members = zf.infolist()
        for m in tqdm(members, desc=f"Extracting {COCO_ZIP.name}", unit="file", colour="yellow"):
            _safe_extract(zf, m, COCO_ROOT)


def generate_decoder_inputs():
    """Generate and save inputs for SAM2.1 Mask Decoder quantization."""
    # if the decoder inputs all already exist, skip generation
    if COCO_DECODER_INPUTS_DIR.exists() and any(COCO_DECODER_INPUTS_DIR.iterdir()):
        print(f"Decoder inputs already exist in {COCO_DECODER_INPUTS_DIR}, skipping generation.")
        return

    if not COCO_DECODER_INPUTS_DIR.exists():
        COCO_DECODER_INPUTS_DIR.mkdir(parents=True)

    # load the PyTorch SAM2.1 Vision Encoder
    vision_encoder = load_sam21_image_encoder("facebook/sam2.1-hiera-small")

    # iterate through COCO128 images and generate decoder inputs
    # decoder inputs are image_embeddings, high_res_feats_256, high_res_feats_128
    # point_coords and point_labels aren't stored and are generated randomly during quantization
    coco_dset = COCOLoader(COCO_DIR)
    pbar = tqdm(range(len(coco_dset)), desc="Generating Decoder Inputs", unit="image", colour="green")
    for idx in pbar:
        pbar.set_description(f"Processing image {idx + 1}/{len(coco_dset)}")
        image = coco_dset[idx]

        # Forward pass through Vision Encoder to get image_embeddings, high_res_feats_256, high_res_feats_128
        image_embeddings, high_res_feats_256, high_res_feats_128 = vision_encoder(torch.from_numpy(image).unsqueeze(0))

        # Save to COCO_DECODER_INPUTS_DIR
        save_path = COCO_DECODER_INPUTS_DIR / f"decoder_input_{idx:04d}.npz"
        np.savez(
            save_path,
            image_embeddings=image_embeddings.numpy(),
            high_res_feats_256=high_res_feats_256.numpy(),
            high_res_feats_128=high_res_feats_128.numpy(),
        )


def main():
    download_and_extract_coco128()
    generate_decoder_inputs()


if __name__ == "__main__":
    main()
