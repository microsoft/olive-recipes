# -------------------------------------------------------------------------
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""OGA inference for VideoChat-Flash ONNX models.

Uses onnxruntime-genai to run exported models with automatic KV-cache
management, image preprocessing, and streaming token generation.

Usage:
    python inference.py --image photo.jpg --prompt "Describe this image"
    python inference.py --prompt "What is the capital of France?"
    python inference.py --interactive
    python inference.py --model-dir ./cpu_and_mobile/models --image cat.png --prompt "What animal is this?"
"""
import argparse
import json

import onnxruntime_genai as og

NUM_VISUAL_TOKENS = 64


def generate_response(model, processor, tokenizer_stream, prompt,
                      image_path=None, max_length=4096):
    """Run a single prompt through the model and stream the response."""
    images = None
    if image_path:
        print(f"Loading image: {image_path}")
        images = og.Images.open(image_path)
        image_pads = "<|image_pad|>" * NUM_VISUAL_TOKENS
        messages = [{
            "role": "user",
            "content": f"<|vision_start|>{image_pads}<|vision_end|>\n{prompt}",
        }]
    else:
        messages = [{"role": "user", "content": prompt}]

    tokenizer = og.Tokenizer(model)
    full_prompt = tokenizer.apply_chat_template(
        json.dumps(messages), add_generation_prompt=True
    )

    print(f"\nPrompt: {prompt}")
    print("Generating response...")

    inputs = processor(full_prompt, images=images)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    print("\nResponse: ", end="", flush=True)
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end="", flush=True)
    print()
    del generator


def interactive_mode(model, processor, tokenizer_stream, max_length=4096):
    """Multi-turn interactive session."""
    print("\n" + "=" * 50)
    print("Interactive Mode — type 'quit' or 'exit' to stop")
    print("To include an image:  image:/path/to/image.jpg your prompt")
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
            generate_response(model, processor, tokenizer_stream,
                              prompt, image_path, max_length)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 50 + "\n")

    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="OGA inference for VideoChat-Flash ONNX models"
    )
    parser.add_argument(
        "--model-dir", type=str, default="cpu_and_mobile/models",
        help="Path to the model directory containing genai_config.json",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_dir}")
    model = og.Model(args.model_dir)
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    if args.interactive:
        interactive_mode(model, processor, tokenizer_stream, args.max_length)
    elif args.prompt:
        generate_response(model, processor, tokenizer_stream,
                          args.prompt, args.image, args.max_length)
    else:
        print("Please provide --prompt or use --interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()
