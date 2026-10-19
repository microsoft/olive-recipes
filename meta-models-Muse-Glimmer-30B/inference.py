"""Text and image inference for the CUDA Muse Glimmer ORT GenAI package."""

import argparse
import json
import os
from pathlib import Path

# cuDNN SDPA currently rejects long BF16 GQA prefills on Hopper. The CUDA
# Flash/MEA kernels remain enabled and handle both text and image prompts.
os.environ.setdefault("ORT_ENABLE_CUDNN_FLASH_ATTENTION", "0")

import onnxruntime_genai as og


_RECIPE_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _RECIPE_DIR / "cuda" / "int4" / "models"


def format_prompt(tokenizer, prompt: str, *, include_image: bool) -> str:
    """Apply the checkpoint chat template to a user request."""
    if include_image:
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
    else:
        content = prompt
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)


def generate(
    model: og.Model,
    processor,
    tokenizer,
    prompt: str,
    *,
    image_path: str | None,
    max_length: int,
) -> None:
    """Generate and stream one Muse Glimmer response."""
    images = og.Images.open(image_path) if image_path else None
    formatted_prompt = format_prompt(tokenizer, prompt, include_image=images is not None)
    inputs = processor(formatted_prompt, images=images)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length, do_sample=False, top_k=1)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    stream = processor.create_stream()

    while not generator.is_done():
        generator.generate_next_token()
        print(stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Muse Glimmer with ONNX Runtime GenAI")
    parser.add_argument("--model-path", type=Path, default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--image", help="Optional path to an input image")
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    if not (args.model_path / "genai_config.json").is_file():
        parser.error(
            f"{args.model_path} is not an ORT GenAI package; run the Olive recipe first"
        )

    model = og.Model(str(args.model_path))
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    generate(
        model,
        processor,
        tokenizer,
        args.prompt,
        image_path=args.image,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
