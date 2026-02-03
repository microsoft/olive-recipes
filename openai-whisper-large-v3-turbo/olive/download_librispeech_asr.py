import argparse
import os

import numpy as np
from datasets import load_dataset


def download_librispeech_asr(save_dir):
    # Create save_dir if it doesn't exist
    save_dir = os.path.join(save_dir, "librispeech_asr_clean_test")
    os.makedirs(save_dir, exist_ok=True)

    # Load streaming dataset
    streamed_dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

    for batch in streamed_dataset:
        file_path = os.path.join(save_dir, f"{batch['id']}.npy")
        np.save(file_path, batch)

    print("Download complete!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save samples from librispeech_asr dataset.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the dataset samples.")
    args = parser.parse_args()

    download_librispeech_asr(args.save_dir)
