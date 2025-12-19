"""
Vision-Language inference script for Qwen2.5-VL using ONNX Runtime GenAI.

Uses onnxruntime-genai for multimodal inference.
"""

import argparse
import onnxruntime_genai as og


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime GenAI inference for Qwen2.5-VL"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory containing genai_config.json and ONNX models"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming output"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer_stream = model.create_tokenizer_stream()

    if args.interactive:
        interactive_mode(model, processor, tokenizer_stream, args)
    elif args.prompt:
        generate_response(model, processor, tokenizer_stream, args.prompt, args.image, args.max_new_tokens, args.streaming)
    else:
        print("Please provide --prompt or use --interactive mode")
        parser.print_help()


def generate_response(model, processor, tokenizer_stream, prompt, image_path, max_new_tokens, streaming):
    """Generate a response for the given prompt and optional image."""

    # Load image if provided
    images = None
    if image_path:
        print(f"Loading image: {image_path}")
        images = og.Images.open(image_path)
        # Add vision tokens to prompt
        full_prompt = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
    else:
        full_prompt = prompt

    print(f"\nPrompt: {prompt}")
    if image_path:
        print(f"Image: {image_path}")
    print("\nGenerating response...")

    # Process inputs
    inputs = processor(full_prompt, images=images)

    # Set up generation parameters
    params = og.GeneratorParams(model)
    params.set_inputs(inputs)
    params.set_search_options(max_length=max_new_tokens)

    # Generate
    if streaming:
        generator = og.Generator(model, params)
        print("\nResponse: ", end="", flush=True)
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)
        print()
        del generator
    else:
        output_tokens = model.generate(params)
        response = processor.decode(output_tokens[0])
        print(f"\nResponse:\n{response}")


def interactive_mode(model, processor, tokenizer_stream, args):
    """Run in interactive mode."""
    print("\n" + "="*50)
    print("Interactive Mode - Enter 'quit' or 'exit' to stop")
    print("To include an image, type: image:/path/to/image.jpg")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input:
            print("Please enter a prompt.")
            continue

        # Check for image path
        image_path = None
        prompt = user_input
        if user_input.startswith("image:"):
            parts = user_input.split(" ", 1)
            image_path = parts[0][6:]  # Remove "image:" prefix
            prompt = parts[1] if len(parts) > 1 else "Describe this image"

        try:
            generate_response(
                model, processor, tokenizer_stream,
                prompt, image_path, args.max_new_tokens, args.streaming
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print("-"*50 + "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
