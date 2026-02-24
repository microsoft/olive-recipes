import argparse
import json

import onnxruntime_genai as og


def generate_response(model, processor, tokenizer_stream, prompt, image_path=None, max_length=4096):
    images = None
    if image_path:
        print(f"Loading image: {image_path}")
        images = og.Images.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    tokenizer = og.Tokenizer(model)
    full_prompt = tokenizer.apply_chat_template(
        json.dumps(messages), add_generation_prompt=True
    )

    inputs = processor(full_prompt, images=images)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    print("Response: ", end="", flush=True)
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end="", flush=True)
    print()
    del generator


def interactive_mode(model, processor, tokenizer_stream, max_length):
    print("\n" + "=" * 50)
    print("Interactive Mode — type 'quit' to exit")
    print("To include an image: image:/path/to/image.jpg your prompt here")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
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
            generate_response(model, processor, tokenizer_stream, prompt, image_path, max_length)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 50 + "\n")

    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL inference with onnxruntime-genai")
    parser.add_argument("--model_path", type=str, default="models", help="Model directory")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--max_length", type=int, default=4096, help="Max generation length")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    if args.interactive:
        interactive_mode(model, processor, tokenizer_stream, args.max_length)
    elif args.prompt:
        generate_response(model, processor, tokenizer_stream, args.prompt, args.image, args.max_length)
    else:
        print("Please provide --prompt or use --interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()
