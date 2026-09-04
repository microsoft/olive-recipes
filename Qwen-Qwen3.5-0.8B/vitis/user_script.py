"""Adapt WikiText tokens to the Mobius Qwen3.5 decoder's multimodal input contract."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig

from olive.data.registry import Registry


class Qwen35DecoderDataLoader:
    def __init__(self, dataset, model_name, exported_model_path):
        self.dataset = dataset
        self.config = AutoConfig.from_pretrained(model_name).text_config
        package = Path(exported_model_path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.embedding = ort.InferenceSession(
            str(package / "embedding" / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(package / "decoder" / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        expected_inputs = {"inputs_embeds", "attention_mask", "position_ids", *self.empty_states()}
        actual_inputs = {item.name for item in self.decoder.get_inputs()}
        if actual_inputs != expected_inputs:
            raise ValueError(
                f"Unexpected Mobius decoder inputs. Missing: {expected_inputs - actual_inputs}; "
                f"extra: {actual_inputs - expected_inputs}. Export the full FP32 Qwen3.5 VLM."
            )
        self.state_outputs = [
            name.replace("past_key_values.", "present.", 1) for name in self.empty_states()
        ]

    def empty_states(self):
        config = self.config
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        states = {}
        for layer, layer_type in enumerate(config.layer_types):
            prefix = f"past_key_values.{layer}"
            if layer_type == "linear_attention":
                states[f"{prefix}.conv_state"] = np.zeros(
                    (1, 2 * key_dim + value_dim, config.linear_conv_kernel_dim - 1), dtype=np.float32
                )
                states[f"{prefix}.recurrent_state"] = np.zeros(
                    (1, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim),
                    dtype=np.float32,
                )
            elif layer_type == "full_attention":
                for name in ("key", "value"):
                    states[f"{prefix}.{name}"] = np.zeros(
                        (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
                    )
            else:
                raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type}")
        return states

    def __iter__(self):
        for batch, _ in torch.utils.data.DataLoader(self.dataset, batch_size=1):
            token_ids = batch["input_ids"][batch["attention_mask"].bool()].numpy().reshape(1, -1)
            states = self.empty_states()
            # Advance the original decoder so calibration sees real recurrent/KV states,
            # rather than treating every WikiText token as an independent empty-cache prompt.
            for position in range(token_ids.shape[1]):
                embeddings = self.embedding.run(
                    ["inputs_embeds"],
                    {
                        "input_ids": token_ids[:, position : position + 1],
                        "image_features": np.zeros((0, self.config.hidden_size), dtype=np.float32),
                    },
                )[0]
                inputs = {
                    "inputs_embeds": embeddings,
                    "attention_mask": np.ones((1, position + 1), dtype=np.int64),
                    "position_ids": np.full((3, 1, 1), position, dtype=np.int64),
                    **states,
                }
                yield {name: torch.from_numpy(value) for name, value in inputs.items()}
                outputs = self.decoder.run(self.state_outputs, inputs)
                states = dict(zip(states, outputs))


@Registry.register_dataloader()
def qwen35_decoder_dataloader(dataset, model_name, exported_model_path, **kwargs):
    return Qwen35DecoderDataLoader(dataset, model_name, exported_model_path)
