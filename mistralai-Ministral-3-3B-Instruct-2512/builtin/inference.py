"""ONNX Runtime GenAI inference for Ministral-3-3B vision-language model.

Usage:
    python inference.py --prompt "What is the capital of France?"
    python inference.py --image photo.jpg --prompt "Describe this image"
    python inference.py --interactive
    python inference.py --model_path cuda/models --prompt "Hello"
"""

import argparse
import json
import time

import onnxruntime_genai as og


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime GenAI inference for Ministral-3-3B"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="cpu_and_mobile/models",
        help="Path to model directory containing genai_config.json and ONNX models",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum total tokens"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = processor.create_stream()

    if args.interactive:
        interactive_mode(model, processor, tokenizer, tokenizer_stream, args)
    elif args.prompt:
        generate_response(
            model,
            processor,
            tokenizer,
            tokenizer_stream,
            args.prompt,
            args.image,
            args.max_length,
        )
    else:
        print("Please provide --prompt or --interactive")
        parser.print_help()


def generate_response(
    model, processor, tokenizer, tokenizer_stream, prompt, image_path, max_length=4096
):
    """Run a single generation."""
    images = None
    if image_path:
        print(f"Loading image: {image_path}")
        images = og.Images.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    full_prompt = tokenizer.apply_chat_template(
        json.dumps(messages), add_generation_prompt=True
    )
    print(f"\nPrompt: {prompt}")
    if image_path:
        print(f"Image: {image_path}")
    print("\nGenerating response...")

    inputs = processor(full_prompt, images=images)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    token_count = 0
    ttft = None
    t_start = time.perf_counter()

    print("\nResponse: ", end="", flush=True)
    while not generator.is_done():
        generator.generate_next_token()
        if ttft is None:
            ttft = time.perf_counter() - t_start
        token_count += 1
        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end="", flush=True)

    t_total = time.perf_counter() - t_start
    print()
    del generator

    decode_tokens = max(token_count - 1, 1)
    decode_time = t_total - (ttft or 0)
    tps = decode_tokens / decode_time if decode_time > 0 else 0

    print(f"\n  Tokens generated : {token_count}")
    print(f"  TTFT             : {(ttft or 0) * 1000:.1f} ms")
    print(f"  Decode TPS       : {tps:.1f} tokens/sec")
    print(f"  Total time       : {t_total:.2f} s")


def interactive_mode(model, processor, tokenizer, tokenizer_stream, args):
    """Run in interactive mode."""
    print("\n" + "=" * 50)
    print("Interactive Mode - Enter 'quit' to stop")
    print("To include an image: image:/path/to/image.jpg your prompt")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        image_path = None
        prompt = user_input
        if user_input.startswith("image:"):
            parts = user_input.split(" ", 1)
            image_path = parts[0][6:]
            prompt = parts[1] if len(parts) > 1 else "Describe this image"

        try:
            generate_response(
                model,
                processor,
                tokenizer,
                tokenizer_stream,
                prompt,
                image_path,
                args.max_length,
            )
        except Exception as e:
            print(f"Error: {e}")

        print("-" * 50 + "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
