# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Offline Olive CLI regressions using a small, local checkpoint."""

import json
import os
import subprocess
import sysconfig
from pathlib import Path

import onnx
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from ling_model_builder import LingModelBuilder


@pytest.mark.parametrize("precision", ["bf16", "q4_int8kv", "q8_int8kv"])
def test_olive_cli_exports_native_precisions(tmp_path, precision):
    recipe_dir = Path(__file__).parent
    config = json.loads((recipe_dir / f"inclusionAI-Ling-3.0-tiny_cuda_{precision}.json").read_text())
    checkpoint = tmp_path / "checkpoint"
    torch.manual_seed(7)
    hf_config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    LlamaForCausalLM(hf_config).save_pretrained(checkpoint)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"token{i}": i for i in range(4, 32)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(checkpoint)
    config["input_model"]["model_path"] = str(checkpoint)
    config["input_model"]["load_kwargs"]["trust_remote_code"] = False
    output = tmp_path / "output"
    config["output_dir"] = str(output)
    config["cache_dir"] = str(tmp_path / "cache")
    calibration = tmp_path / "kv_scales.json"
    calibration.write_text(
        json.dumps({"layer_ids": [0], "scales": {"k_scales": [[0.02] * 128], "v_scales": [[0.02] * 128]}})
        + "\n"
    )
    pass_config = next(iter(config["passes"].values()))
    pass_config["extra_options"].update(
        kv_cache_scale_file=str(calibration), num_blocks=16, max_scheduled_tokens=16
    )
    pass_config["extra_options"].pop("gpu_utilization_factor")
    run_config = tmp_path / "recipe.json"
    run_config.write_text(json.dumps(config) + "\n")
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(recipe_dir), os.environ.get("PYTHONPATH")])),
    }

    result = subprocess.run(
        [
            str(Path(sysconfig.get_path("scripts")) / "olive"),
            "run",
            "--config",
            str(run_config),
            "--package-config",
            str(recipe_dir / "package_config.json"),
            "--disable_telemetry",
        ],
        cwd=recipe_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    models = list(output.rglob("*.onnx"))
    assert len(models) == 1
    graph = onnx.load(models[0], load_external_data=False).graph
    floating_types = {
        tensor.data_type
        for tensor in graph.initializer
        if tensor.data_type in {onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16}
    }
    assert floating_types == {onnx.TensorProto.BFLOAT16}
    if precision == "bf16":
        assert not any(node.op_type == "MatMulNBits" for node in graph.node)
        return
    gathers = [
        node for node in graph.node
        if node.name == "/model/embed_tokens/GatherBlockQuantized"
    ]
    assert len(gathers) == 1
    attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in gathers[0].attribute}
    assert attributes["bits"] == 8
    assert attributes["gather_axis"] == 0
    assert attributes["quantize_axis"] == 1
    assert not any(node.name == "/model/embed_tokens/Gather" for node in graph.node)
    matmul_bits = {
        onnx.helper.get_attribute_value(attr)
        for node in graph.node if node.op_type == "MatMulNBits"
        for attr in node.attribute if attr.name == "bits"
    }
    assert matmul_bits == ({4, 8} if precision == "q4_int8kv" else {8})


def test_native_pass_rejects_an_already_patched_embedding_exporter(monkeypatch):
    from onnxruntime_genai.models import builder
    from olive.passes.onnx.model_builder import patched_make_embedding

    monkeypatch.setattr(builder.Model, "make_embedding", patched_make_embedding)

    with pytest.raises(RuntimeError, match="already patched by Olive"):
        LingModelBuilder.__new__(LingModelBuilder).maybe_patch_quant()
