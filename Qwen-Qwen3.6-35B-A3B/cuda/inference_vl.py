"""ONNX Runtime GenAI multimodal inference for Qwen3.6-35B-A3B on CUDA."""

import argparse
import json
import time
from pathlib import Path

import onnxruntime_genai as og

DEFAULT_MODEL_PATH = "cuda/vl_kquant_fp16/models"
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tiff", ".webp"}


def load_runtime(model_path: Path):
    """Load the model and its multimodal processing assets."""
    start = time.perf_counter()
    model = og.Model(str(model_path))
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    chat_template = (model_path / "chat_template.jinja").read_text(encoding="utf-8")
    print(f"Model loaded in {time.perf_counter() - start:.2f}s")
    return model, processor, tokenizer, chat_template


def generate_response(
    model,
    processor,
    tokenizer,
    chat_template: str,
    prompt: str,
    image_path: Path | None,
    max_length: int,
    max_new_tokens: int,
    stream: bool = True,
    system_prompt: str = "",
):
    """Generate one response and return its text and timing metrics."""
    content = prompt
    images = None
    if image_path is not None:
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        images = og.Images.open(str(image_path))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    full_prompt = tokenizer.apply_chat_template(
        messages=json.dumps(messages),
        tools="",
        add_generation_prompt=True,
        template_str=chat_template,
    )

    preprocess_start = time.perf_counter()
    inputs = processor(full_prompt, images=images)
    preprocess_time = time.perf_counter() - preprocess_start

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length, do_sample=False, top_k=1)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    input_tokens = generator.token_count()

    tokenizer_stream = tokenizer.create_stream()
    output_tokens = []
    generation_start = time.perf_counter()
    time_to_first_token = None

    if stream:
        print("Response: ", end="", flush=True)
    while not generator.is_done() and len(output_tokens) < max_new_tokens:
        generator.generate_next_token()
        if time_to_first_token is None:
            time_to_first_token = time.perf_counter() - generation_start
        token = generator.get_next_tokens()[0]
        output_tokens.append(token)
        if stream:
            print(tokenizer_stream.decode(token), end="", flush=True)

    generation_time = time.perf_counter() - generation_start
    if stream:
        print()
    del generator

    decode_time = generation_time - (time_to_first_token or 0)
    decode_tokens = max(len(output_tokens) - 1, 0)
    tokens_per_second = decode_tokens / decode_time if decode_time > 0 else 0
    text = tokenizer.decode(output_tokens)
    metrics = {
        "input_tokens": input_tokens,
        "output_tokens": len(output_tokens),
        "preprocess_s": preprocess_time,
        "ttft_s": time_to_first_token or 0,
        "decode_tokens_per_s": tokens_per_second,
        "generation_s": generation_time,
    }
    return text, metrics


def print_metrics(metrics: dict):
    print(f"Input tokens: {metrics['input_tokens']}")
    print(f"Output tokens: {metrics['output_tokens']}")
    print(f"Preprocess: {metrics['preprocess_s'] * 1000:.1f} ms")
    print(f"TTFT: {metrics['ttft_s'] * 1000:.1f} ms")
    print(f"Decode: {metrics['decode_tokens_per_s']:.2f} tokens/s")
    print(f"Generation: {metrics['generation_s']:.2f}s")


def benchmark_folder(runtime, folder: Path, prompt: str, max_length: int, max_new_tokens: int):
    image_paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"no supported images found in: {folder}")

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")
        text, metrics = generate_response(
            *runtime,
            prompt,
            image_path,
            max_length,
            max_new_tokens,
            stream=False,
        )
        results.append(metrics)
        print(
            f"  output_tokens={metrics['output_tokens']} "
            f"TTFT={metrics['ttft_s'] * 1000:.1f}ms "
            f"decode={metrics['decode_tokens_per_s']:.2f} tokens/s"
        )
        print(f"  output={text.strip()[:200]!r}")

    count = len(results)
    print(f"\nBenchmark summary ({count} images)")
    print(f"Average TTFT: {sum(item['ttft_s'] for item in results) / count * 1000:.1f} ms")
    print(
        "Average decode: "
        f"{sum(item['decode_tokens_per_s'] for item in results) / count:.2f} tokens/s"
    )


def interactive_mode(runtime, max_length: int, max_new_tokens: int):
    print("Enter 'image:/path/to/image.jpg prompt' for image + text, or 'quit' to exit.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user_input.lower() in {"exit", "quit"}:
            return
        if not user_input:
            continue

        image_path = None
        prompt = user_input
        if user_input.startswith("image:"):
            image_spec, separator, prompt = user_input.partition(" ")
            image_path = Path(image_spec.removeprefix("image:"))
            if not image_path.is_file():
                print(f"Image not found: {image_path}")
                continue
            if not separator:
                prompt = "Describe this image."

        text, metrics = generate_response(
            *runtime,
            prompt,
            image_path,
            max_length,
            max_new_tokens,
        )
        if not text:
            print("No output generated.")
        print_metrics(metrics)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-35B-A3B multimodal ORT GenAI inference")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, type=Path, help="Model directory")
    parser.add_argument("--image", type=Path, help="Input image path")
    parser.add_argument("--prompt", default="Describe this image.", help="Text prompt")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum total sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive prompt loop")
    parser.add_argument("--benchmark", type=Path, help="Run all supported images in a directory")
    args = parser.parse_args()

    if not args.model_path.is_dir():
        parser.error(f"model directory not found: {args.model_path}")
    if args.image is not None and not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    if args.benchmark is not None and not args.benchmark.is_dir():
        parser.error(f"benchmark directory not found: {args.benchmark}")

    runtime = load_runtime(args.model_path)
    if args.benchmark is not None:
        benchmark_folder(runtime, args.benchmark, args.prompt, args.max_length, args.max_new_tokens)
    elif args.interactive:
        interactive_mode(runtime, args.max_length, args.max_new_tokens)
    else:
        _, metrics = generate_response(
            *runtime,
            args.prompt,
            args.image,
            args.max_length,
            args.max_new_tokens,
        )
        print_metrics(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
