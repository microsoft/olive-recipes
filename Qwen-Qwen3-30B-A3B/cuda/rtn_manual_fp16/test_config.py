import json
from pathlib import Path

import onnx_ir as ir
from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.selection import iter_quant_targets
from olive.common.quant.tensor import QuantTensor
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.mobius_model_builder import MobiusBuilder
from olive.passes.pytorch.rtn import Rtn
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM


def _quantization_config():
    config = json.loads(Path(__file__).with_name("config.json").read_text())["passes"][
        "rtn"
    ]
    return OliveHfQuantizationConfig(
        bits=config["bits"],
        symmetric=config["sym"],
        group_size=config["group_size"],
        moe=config["moe"],
        embeds=config["embeds"],
        lm_head=config["lm_head"],
        overrides=config["overrides"],
    )


def test_manual_overrides_match_layer_schedule():
    config = _quantization_config()
    pass_config = json.loads(Path(__file__).with_name("config.json").read_text())[
        "passes"
    ]["rtn"]
    assert pass_config["independent_qkv"] is True
    assert config.embeds is True
    assert config.lm_head is True
    assert config.get_qlinear_init_args("model.embed_tokens")["bits"] == 4
    assert config.get_qlinear_init_args("lm_head")["bits"] == 8
    higher_precision = set()
    for layer in range(48):
        selected = layer < 6 or layer >= 42 or (layer - 6) % 3 == 2
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"model.layers.{layer}.self_attn.{projection}"
            assert config.get_qlinear_init_args(name)["bits"] == (
                8 if selected and projection == "v_proj" else 4
            )
            if selected and projection == "v_proj":
                higher_precision.add(name)

        name = f"model.layers.{layer}.mlp.experts.down_proj"
        assert config.get_qlinear_init_args(name)["bits"] == (8 if selected else 4)
        if selected:
            higher_precision.add(name)
        assert (
            config.get_qlinear_init_args(
                f"model.layers.{layer}.mlp.experts.gate_up_proj"
            )["bits"]
            == 4
        )

    assert len(higher_precision) == 24 * 2
    assert config.get_qlinear_init_args("model.layers.48.self_attn.q_proj")["bits"] == 4
    assert (
        config.get_qlinear_init_args("model.layers.48.mlp.experts.down_proj")["bits"]
        == 4
    )
    assert (
        OliveHfQuantizationConfig(**config.to_dict()).get_qlinear_init_args(
            "model.layers.41.mlp.experts.down_proj"
        )["bits"]
        == 8
    )


def test_manual_overrides_target_real_qwen3_moe_parameters():
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            hidden_size=64,
            intermediate_size=96,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=256,
            max_position_embeddings=128,
        )
    )
    names = {
        name
        for _, _, name in iter_quant_targets(
            model,
            quantize_lm_head=True,
            quantize_embeds=_quantization_config().embeds,
            quantize_moe=True,
        )
    }
    assert "model.embed_tokens" in names
    assert "lm_head" in names
    for layer in range(2):
        assert {
            f"model.layers.{layer}.self_attn.{projection}"
            for projection in ("q_proj", "k_proj", "v_proj")
        } <= names
        assert f"model.layers.{layer}.mlp.experts.down_proj" in names
        assert f"model.layers.{layer}.mlp.experts.gate_up_proj" in names
        assert f"model.layers.{layer}.mlp.gate" not in names


def test_manual_rtn_quantizes_selected_fused_experts_and_exports_head(tmp_path):
    model_dir = tmp_path / "input"
    model_dir.mkdir()
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            hidden_size=128,
            intermediate_size=128,
            moe_intermediate_size=128,
            num_hidden_layers=7,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=2,
            num_experts_per_tok=1,
            vocab_size=32,
            max_position_embeddings=128,
        )
    )
    model.save_pretrained(model_dir)
    tokenizer = Tokenizer(
        models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0", pad_token="t0"
    ).save_pretrained(model_dir)

    config = json.loads(Path(__file__).with_name("config.json").read_text())["passes"][
        "rtn"
    ]
    quantizer = create_pass_from_dict(
        Rtn, {k: v for k, v in config.items() if k != "type"}
    )
    output = quantizer.run(
        HfModelHandler(model_path=str(model_dir)), str(tmp_path / "rtn")
    )
    quantized_model = output.load_model()
    embedding = quantized_model.model.embed_tokens.weight.data
    assert isinstance(embedding, QuantTensor)
    assert embedding.bits == 4
    head = quantized_model.lm_head.weight.data
    assert isinstance(head, QuantTensor)
    assert head.bits == 8
    assert (
        quantized_model.config.quantization_config.get_qlinear_init_args("lm_head")[
            "bits"
        ]
        == 8
    )
    layers = quantized_model.model.layers

    for layer, bits in ((layers[0], 8), (layers[6], 4)):
        assert isinstance(layer.mlp.experts.gate_up_proj.data, QuantTensor)
        assert layer.mlp.experts.gate_up_proj.data.bits == 4
        assert isinstance(layer.mlp.experts.down_proj.data, QuantTensor)
        assert layer.mlp.experts.down_proj.data.bits == bits
        for projection in ("q_proj", "k_proj", "v_proj"):
            weight = getattr(layer.self_attn, projection).weight.data
            assert isinstance(weight, QuantTensor)
            assert weight.bits == (bits if projection == "v_proj" else 4)

    builder = create_pass_from_dict(
        MobiusBuilder,
        {"precision": "fp16"},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(
            accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"
        ),
    )
    builder.run(output, str(tmp_path / "mobius"))
    graph = ir.load(tmp_path / "mobius" / "model.onnx")
    quantized_linears = {
        node.inputs[1].name: node.attributes["bits"].as_int()
        for node in graph.graph
        if node.op_type == "MatMulNBits" and node.inputs[1] is not None
    }
    assert quantized_linears["lm_head.weight"] == 8
    for layer in (0, 6):
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"model.layers.{layer}.self_attn.{projection}.weight"
            assert quantized_linears[name] == (
                8 if layer == 0 and projection == "v_proj" else 4
            )
    assert sum(node.op_type == "QMoE" for node in graph.graph) == 7
    assert sum(node.op_type == "GatherBlockQuantized" for node in graph.graph) == 1
