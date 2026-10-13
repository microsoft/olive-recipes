import json
from pathlib import Path

from olive.common.quant.hf_utils import OliveHfQuantizationConfig


def test_mixed_int2_layer_schedule():
    recipe = json.loads(Path(__file__).with_name("config.json").read_text())
    rtn = recipe["passes"]["rtn"]
    config = OliveHfQuantizationConfig(
        bits=rtn["bits"],
        symmetric=rtn["sym"],
        group_size=rtn["group_size"],
        moe=rtn["moe"],
        embeds=rtn["embeds"],
        lm_head=rtn["lm_head"],
        overrides=rtn["overrides"],
    )
    selected = {
        6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22,
        24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40,
    }
    assert len(selected) == 24
    assert config.moe and config.embeds and config.lm_head
    for layer in range(48):
        for projection in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
            args = config.get_qlinear_init_args(
                f"model.layers.{layer}.mlp.experts.{projection}"
            )
            expected_bits = 2 if layer in selected and projection != "down_proj" else 4
            assert args["bits"] == expected_bits
            assert args["group_size"] == 64
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            args = config.get_qlinear_init_args(
                f"model.layers.{layer}.self_attn.{projection}"
            )
            assert args["bits"] == 4
            assert args["group_size"] == 64
    for name, bits in (("model.embed_tokens", 4), ("lm_head", 8)):
        args = config.get_qlinear_init_args(name)
        assert args["bits"] == bits
        assert args["group_size"] == 64
    assert not Path(recipe["output_dir"]).is_absolute()