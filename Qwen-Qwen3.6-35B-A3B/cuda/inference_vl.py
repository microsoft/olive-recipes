"""ONNX Runtime GenAI multimodal inference for Qwen3.6-35B-A3B on CUDA.

Usage:
    python cuda/inference_vl.py --image /path/to/image.png
    python cuda/inference_vl.py --image /path/to/image.png --prompt "Describe this image."
"""

import argparse
import json
import time
from pathlib import Path

import onnxruntime_genai as og

DEFAULT_MODEL_PATH = "cuda/vl_kquant_fp16/models"


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-35B-A3B multimodal ORT GenAI inference")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Model directory")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", default="Describe this image.", help="Text prompt")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum total sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum tokens to generate")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    image_path = Path(args.image)
    if not model_path.is_dir():
        parser.error(f"model directory not found: {model_path}")
    if not image_path.is_file():
        parser.error(f"image not found: {image_path}")

    start = time.time()
    model = og.Model(str(model_path))
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    processor = model.create_multimodal_processor()

    messages = json.dumps(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": args.prompt}],
            }
        ]
    )
    chat_template = (model_path / "chat_template.jinja").read_text(encoding="utf-8")
    prompt = tokenizer.apply_chat_template(
        messages=messages,
        tools="",
        add_generation_prompt=True,
        template_str=chat_template,
    )
    inputs = processor(prompt, images=og.Images.open(str(image_path)))

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=args.max_length, do_sample=False, top_k=1)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    print(f"Input tokens: {generator.token_count()}")
    print("Output: ", end="", flush=True)
    output_tokens = []
    while not generator.is_done() and len(output_tokens) < args.max_new_tokens:
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        output_tokens.append(token)
        print(tokenizer_stream.decode(token), end="", flush=True)

    print()
    elapsed = time.time() - start
    print(f"Output tokens: {len(output_tokens)}, elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
