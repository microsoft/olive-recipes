# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Calibration data for the Gemma 4 decoder on QNN.

The decoder consumed by this recipe has already been through
``AttentionMaskToSequenceLengths``, so it takes ``past_seq_len`` and
``total_seq_len`` instead of ``attention_mask`` and ``position_ids``, and its
floating point inputs are fp16 after ``OnnxFloatToFloat16``.

Samples are produced by tokenizing WikiText 2 and running the exported
embedding component, so no calibration tensors need to be checked in.
"""

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from olive.data.registry import Registry
from tokenizers import Tokenizer
from torch.utils.data import Dataset

# Read the parquet directly rather than through ``datasets``, which pulls in
# pyarrow. pyarrow publishes no wheel for Windows on ARM64, so the ``datasets``
# route cannot be installed on the same machine that runs the QNN target.
DATASET_REPO = "Salesforce/wikitext"
DATASET_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
NUM_KV_LAYERS = 15
HEAD_DIM = 256
GLOBAL_LAYERS = frozenset({4, 9, 14})
GLOBAL_HEAD_DIM = 512


class DecoderCalibrationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


@Registry.register_dataset()
def wikitext_decoder_calibration_dataset(
    embedding_model_path,
    tokenizer_path=None,
    num_samples=32,
    seq_len=64,
    past_len=64,
    **kwargs,
):
    """Build decoder calibration samples from WikiText 2."""
    del kwargs
    if num_samples <= 0 or seq_len <= 0 or past_len <= 0:
        raise ValueError("num_samples, seq_len, and past_len must all be greater than zero.")

    model_dir = _parent(embedding_model_path)
    tokenizer = Tokenizer.from_file(tokenizer_path or f"{_parent(model_dir)}/tokenizer.json")

    parquet_path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    texts = pd.read_parquet(parquet_path, engine="fastparquet")["text"].tolist()

    ids = []
    needed = num_samples * seq_len
    for text in texts:
        text = text.strip()
        if len(text) < 200:
            continue
        ids.extend(tokenizer.encode(text, add_special_tokens=False).ids)
        if len(ids) >= needed * 4:
            break
    if len(ids) < needed:
        raise ValueError(f"Not enough calibration text: have {len(ids)} tokens, need {needed}.")

    stride = len(ids) // num_samples
    windows = [ids[index * stride : index * stride + seq_len] for index in range(num_samples)]

    session = ort.InferenceSession(embedding_model_path, providers=["CPUExecutionProvider"])
    empty_features = np.zeros((0, 1536), dtype=np.float32)
    empty_kv = {
        layer: torch.zeros(
            (1, 1, past_len, GLOBAL_HEAD_DIM if layer in GLOBAL_LAYERS else HEAD_DIM),
            dtype=torch.float16,
        )
        for layer in range(NUM_KV_LAYERS)
    }

    samples = []
    for window in windows:
        embeds, per_layer = session.run(
            ["inputs_embeds", "per_layer_inputs"],
            {
                "input_ids": np.asarray([window], dtype=np.int64),
                "image_features": empty_features,
                "audio_features": empty_features,
            },
        )
        sample = {
            "inputs_embeds": torch.from_numpy(embeds.astype(np.float16)),
            "per_layer_inputs": torch.from_numpy(per_layer.astype(np.float16)),
            # GroupQueryAttention reads this as seqlens_k, which is
            # total_sequence_length - 1, despite the graph input name.
            "past_seq_len": torch.full((1, 1), seq_len - 1, dtype=torch.int32),
            "total_seq_len": torch.tensor(seq_len, dtype=torch.int32),
        }
        for layer in range(NUM_KV_LAYERS):
            sample[f"past_key_values.{layer}.key"] = empty_kv[layer].clone()
            sample[f"past_key_values.{layer}.value"] = empty_kv[layer].clone()
        samples.append(sample)

    return DecoderCalibrationDataset(samples)


def _parent(path):
    return str(path).replace("\\", "/").rsplit("/", 1)[0]
