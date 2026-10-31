"""Export the speech graphs of LFM2.5-Audio as FP32 ONNX, the input of the `_audio_*.json` recipes.

The decoder recipes build the LFM2 backbone with the ONNX Runtime GenAI model builder. The other
graphs of the pipeline are built from the checkpoint by LiquidAI's onnx-export (`liquidonnx`), one
directory each:

    audio_encoder/      FastConformer encoder + adapter: log-mel frames -> decoder features
    embedding/          the decoder's token table; scatters the encoder features over the
                        audio placeholders and returns `inputs_embeds`
    depthformer/        decoder hidden state -> one frame of audio codes
    audio_embedding/    audio codes -> the decoder's next input
    audio_detokenizer/  audio codes -> STFT features (turned into a waveform by inference.py)

Usage:
    python export_audio.py [output_dir]
"""

import argparse
import json
import tempfile
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download
from liquidonnx.embeddings import build_embeddings
from liquidonnx.lfm2_audio.builder.config import ConformerConfig
from liquidonnx.lfm2_audio.builder.conformer_builder import ConformerEncoderBuilder
from liquidonnx.lfm2_audio.builder.depthformer_builder import export_vocoder_depthformer
from liquidonnx.lfm2_audio.builder.detokenizer_builder import export_audio_detokenizer_builder
from liquidonnx.lfm2_audio.export import AUDIO_TOKEN_ID, export_audio_embedding
from onnx import helper, numpy_helper
from safetensors import safe_open

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"

# The audio recipes pick out the text embedding table by this node name.
EMBED_TOKENS_NODE = "/embed_tokens/Gather"


def save(model: onnx.ModelProto, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=True, location=f"{path.name}.data")
    print(f"Wrote {path}")


def pointwise_convs_as_matmul(model: onnx.ModelProto) -> int:
    """Rewrite the encoder's 1x1 convolutions as MatMul, so Olive quantizes them as Q4_0/Q8_0 do.

    Olive's block quantizer only rewrites MatMul and Gather. Each convolution sits next to a
    [B, T, C] <-> [B, C, T] Transpose, which the MatMul absorbs.
    """
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def is_transpose_021(node):
        return node is not None and node.op_type == "Transpose" and list(helper.get_attribute_value(node.attribute[0])) == [0, 2, 1]

    replace, drop = {}, set()
    for n in graph.node:
        weight = inits.get(n.input[1]) if n.op_type == "Conv" else None
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if weight is None or list(weight.dims[2:]) != [1] or attrs.get("group", 1) != 1:
            continue
        w = numpy_helper.from_array(numpy_helper.to_array(weight)[:, :, 0].T.copy(), f"{n.input[1]}.matmul")
        graph.initializer.append(w)
        before = producer.get(n.input[0])
        after = consumers.get(n.output[0], [])
        stem = n.name.removesuffix("/Conv")
        nodes = []
        if is_transpose_021(before) and len(consumers[before.output[0]]) == 1:
            drop.add(before.name)
            x, out = before.input[0], f"{stem}/btc"
        else:
            nodes.append(helper.make_node("Transpose", [n.input[0]], [f"{stem}/in_btc"], name=f"{stem}/in_transpose", perm=[0, 2, 1]))
            x, out = f"{stem}/in_btc", f"{stem}/btc"
        if len(after) == 1 and is_transpose_021(after[0]):
            drop.add(after[0].name)
            out = after[0].output[0]
        nodes.append(helper.make_node("MatMul", [x, w.name], [f"{stem}/mm"], name=f"{stem}/MatMul"))
        nodes.append(helper.make_node("Add", [f"{stem}/mm", n.input[2]], [out], name=f"{stem}/Add"))
        if out == f"{stem}/btc":
            nodes.append(helper.make_node("Transpose", [out], [n.output[0]], name=f"{stem}/out_transpose", perm=[0, 2, 1]))
        replace[n.name] = nodes
    new_nodes = []
    for n in graph.node:
        if n.name in drop:
            continue
        new_nodes.extend(replace.get(n.name, [n]))
    del graph.node[:]
    graph.node.extend(new_nodes)
    used = {i for n in graph.node for i in n.input}
    keep = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    produced = {o for n in graph.node for o in n.output}
    keep_vi = [v for v in graph.value_info if v.name in produced]
    del graph.value_info[:]
    graph.value_info.extend(keep_vi)
    return len(replace)


def load_tensors(names: list[str]) -> dict:
    with safe_open(hf_hub_download(MODEL_ID, "model.safetensors"), framework="pt") as f:
        return {name: f.get_tensor(name).float().numpy() for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", nargs="?", default="audio_fp32", help="default: audio_fp32")
    output_dir = Path(parser.parse_args().output_dir)

    config = json.loads(Path(hf_hub_download(MODEL_ID, "config.json")).read_text())
    weights = load_tensors(["lfm.embed_tokens.weight", "audio_embedding.embedding.weight"])

    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)

        build_embeddings(weights["lfm.embed_tokens.weight"], AUDIO_TOKEN_ID, "audio_features", scratch / "e.onnx")
        embedding = onnx.load(scratch / "e.onnx")
        gather = next(node for node in embedding.graph.node if node.op_type == "Gather")
        gather.name = EMBED_TOKENS_NODE
        save(embedding, output_dir / "embedding" / "model.onnx")

        encoder = ConformerEncoderBuilder(ConformerConfig.from_hf_config(config["encoder"]), config["lfm"]["hidden_size"])
        encoder = encoder.build(MODEL_ID)
        pointwise_convs_as_matmul(encoder)
        save(encoder, output_dir / "audio_encoder" / "model.onnx")

        for name, path in (
            ("audio_embedding", export_audio_embedding(weights, config, scratch)),
            ("depthformer", export_vocoder_depthformer(MODEL_ID, scratch)),
            ("audio_detokenizer", export_audio_detokenizer_builder(MODEL_ID, scratch)),
        ):
            save(onnx.load(path), output_dir / name / "model.onnx")


if __name__ == "__main__":
    main()
