"""Wire the decoder and the speech graphs into one ONNX Runtime GenAI package.

The decoder recipe writes ``model.onnx``, ``genai_config.json`` and the tokenizer files to the
output directory; the audio recipe adds one ``<graph>.onnx`` per speech graph next to them. This
script adds the ``embedding``, ``speech`` and ``audio_output`` sections to ``genai_config.json``,
and gives the embedding model the output type of the decoder's ``inputs_embeds``.

Usage:
    python finalize.py [model_dir]
"""

import argparse
import json
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download
from onnx import helper

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"

# The placeholder the embedding model replaces with encoder features; export_audio.py builds the
# embedding model with the same id. The other two are where the model turns from text to speech.
AUDIO_TOKEN = ("<|reserved_123|>", 133)
AUDIO_START = ("<|audio_start|>", 128)
TEXT_END = ("<|text_end|>", 130)

GRAPHS = ("audio_encoder", "embedding", "depthformer", "audio_embedding", "audio_detokenizer")


def match_embeds_type(model_dir: Path) -> bool:
    """Make the embedding model return inputs_embeds in the type the decoder takes.

    The runtime lets the embedding model write straight into the decoder's input buffer. The
    export returns float32, which a CPU decoder takes; CUDA and WebGPU decoders take float16.
    """
    decoder = onnx.load(model_dir / "model.onnx", load_external_data=False)
    embeds_type = next(i.type.tensor_type.elem_type for i in decoder.graph.input if i.name == "inputs_embeds")
    path = model_dir / "embedding.onnx"
    embedding = onnx.load(path, load_external_data=False)
    output = next(o for o in embedding.graph.output if o.name == "inputs_embeds")
    if output.type.tensor_type.elem_type == embeds_type:
        return False
    for node in embedding.graph.node:
        node.output[:] = [f"{name}_fp32" if name == output.name else name for name in node.output]
    embedding.graph.node.append(helper.make_node("Cast", [f"{output.name}_fp32"], [output.name], to=embeds_type))
    output.type.tensor_type.elem_type = embeds_type
    onnx.save(embedding, path)
    return True


def check_token_ids(tokenizer_path: Path):
    ids = {token["content"]: token["id"] for token in json.loads(tokenizer_path.read_text())["added_tokens"]}
    for content, expected in (AUDIO_TOKEN, AUDIO_START, TEXT_END):
        if ids.get(content) != expected:
            raise SystemExit(f"{tokenizer_path}: expected {content} to be {expected}, got {ids.get(content)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", nargs="?", default="model", help="output_dir of the recipes (default: model)")
    model_dir = Path(parser.parse_args().model_dir)

    for filename in ("genai_config.json", "tokenizer.json", *(f"{graph}.onnx" for graph in GRAPHS)):
        if not (model_dir / filename).exists():
            raise SystemExit(f"{model_dir / filename} not found: run both the decoder and the audio recipe first")

    genai_config_path = model_dir / "genai_config.json"
    genai_config = json.loads(genai_config_path.read_text())
    model = genai_config["model"]
    if model["type"] != "lfm2_audio":
        raise SystemExit(f'expected model type "lfm2_audio" in {genai_config_path}, got "{model["type"]}"')
    check_token_ids(model_dir / "tokenizer.json")
    hf_config = json.loads(Path(hf_hub_download(MODEL_ID, "config.json")).read_text())

    # Without their own session_options the speech graphs run on the decoder's execution provider.
    # On WebGPU the encoder and the audio head stay on CPU: the WebGPU EP does not always grow the
    # depthformer's KV cache correctly (microsoft/onnxruntime#32716), and both run faster on CPU.
    provider_options = model["decoder"]["session_options"]["provider_options"]
    on_webgpu = any(key.lower() == "webgpu" for option in provider_options for key in option)
    cpu = {"session_options": {"log_id": "onnxruntime-genai", "provider_options": []}} if on_webgpu else {}

    model["audio_token_id"] = AUDIO_TOKEN[1]
    # The builder ends a text-only answer at the switch to speech; with audio_output the model
    # speaks there instead, and the runtime refuses a config that still stops on them.
    eos = model["eos_token_id"] if isinstance(model["eos_token_id"], list) else [model["eos_token_id"]]
    model["eos_token_id"] = [token for token in eos if token not in (AUDIO_START[1], TEXT_END[1])]
    model["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {"input_ids": "input_ids", "audio_features": "audio_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
    }
    model["speech"] = {
        "filename": "audio_encoder.onnx",
        "inputs": {"audio_embeds": "mel_spectrogram", "audio_lengths": "mel_lengths", "audio_sizes": "audio_sizes"},
        "outputs": {"audio_features": "audio_embeddings"},
        **cpu,
    }
    model["audio_output"] = {
        "depthformer": {"filename": "depthformer.onnx", **cpu},
        "embedding": {"filename": "audio_embedding.onnx", **cpu},
        "num_codebooks": hf_config["codebooks"],
        "audio_start_token_id": AUDIO_START[1],
        "text_end_token_id": TEXT_END[1],
        "interleaved_n_text": hf_config["interleaved_n_text"],
        "interleaved_n_audio": hf_config["interleaved_n_audio"],
    }
    genai_config_path.write_text(json.dumps(genai_config, indent=4) + "\n")
    print(f"Updated {genai_config_path}")

    if match_embeds_type(model_dir):
        print(f"Cast the output of {model_dir / 'embedding.onnx'} to the decoder's inputs_embeds type")


if __name__ == "__main__":
    main()
