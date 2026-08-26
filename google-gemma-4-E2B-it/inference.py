"""ONNX Runtime GenAI inference for Gemma 4 models.

Supports text and image inference with chat template formatting.

Usage:
    python inference.py --prompt "What is the capital of France?"
    python inference.py --image photo.jpg --prompt "Describe this image."
    python inference.py --device gpu --variant int4 --prompt "Explain quantum computing"
    python inference.py --interactive
    python inference.py --model-path /path/to/models --prompt "Hello"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import onnxruntime_genai as og


def resolve_model_path(device: str, variant: str | None) -> str:
    """Resolve the model directory from device + variant args."""
    if device == "cpu":
        variant = variant or "fp32"
        return f"cpu/{variant}/models"
    variant = variant or "int4"
    return f"cuda/{variant}/models"


def format_chat_prompt(
    tokenizer,
    prompt: str,
    system_prompt: str | None = None,
    has_image: bool = False,
) -> str:
    """Format a prompt using Gemma4's chat template.

    Gemma4 instruction-tuned models require chat formatting for best results.
    The tokenizer's apply_chat_template handles <bos>, turn markers, etc.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content = (
        [{"type": "image"}, {"type": "text", "text": prompt}] if has_image else prompt
    )
    messages.append({"role": "user", "content": content})

    # ORT GenAI tokenizer expects the messages as a JSON string. Older
    # releases cannot execute Gemma 4's newer Jinja template, so fall back to
    # the equivalent canonical turn markers for simple text generation.
    try:
        return tokenizer.apply_chat_template(json.dumps(messages))
    except RuntimeError as exc:
        if "Invalid or unsupported chat template" not in str(exc):
            raise

    turns = ["<bos>"]
    if system_prompt:
        turns.append(f"<|turn>system\n{system_prompt.strip()}<turn|>\n")
    image_token = "<|image|>" if has_image else ""
    turns.append(f"<|turn>user\n{image_token}{prompt.strip()}<turn|>\n<|turn>model\n")
    return "".join(turns)


def generate(
    model: og.Model,
    tokenizer: og.Tokenizer,
    prompt: str,
    max_length: int = 2048,
    system_prompt: str | None = None,
    image_path: str | None = None,
    verbose: bool = False,
) -> str:
    """Generate text from a prompt."""
    formatted = format_chat_prompt(
        tokenizer,
        prompt,
        system_prompt,
        has_image=image_path is not None,
    )
    input_ids = None
    multimodal_inputs = None
    if image_path is not None:
        processor = model.create_multimodal_processor()
        images = og.Images.open(image_path)
        multimodal_inputs = processor(formatted, images=images)
    else:
        input_ids = tokenizer.encode(formatted)

    if verbose and input_ids is not None:
        print(f"  Input tokens: {len(input_ids)}")

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=max_length,
        do_sample=False,
        top_k=1,
    )

    start = time.time()
    generator = og.Generator(model, params)
    if multimodal_inputs is not None:
        generator.set_inputs(multimodal_inputs)
    else:
        generator.append_tokens([input_ids])

    output_tokens = []
    tokenizer_stream = tokenizer.create_stream()

    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        output_tokens.append(token)

        if verbose:
            piece = tokenizer_stream.decode(token)
            print(piece, end="", flush=True)

    elapsed = time.time() - start
    output_text = tokenizer.decode(output_tokens)

    if verbose:
        print()
        tps = len(output_tokens) / elapsed if elapsed > 0 else 0
        print(
            f"  Output tokens: {len(output_tokens)}, Time: {elapsed:.2f}s, Speed: {tps:.1f} tok/s"
        )

    return output_text


def interactive_mode(model: og.Model, tokenizer: og.Tokenizer, max_length: int):
    """Run interactive chat loop."""
    print("Interactive mode. Type 'quit' to exit.")
    print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        print("Gemma: ", end="", flush=True)
        generate(model, tokenizer, prompt, max_length=max_length, verbose=True)
        print()


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 ORT GenAI Inference")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--variant", choices=["fp32", "fp16", "int4"], default=None)
    parser.add_argument("--model-path", default=None, help="Override model directory")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt")
    parser.add_argument(
        "--image", type=str, default=None, help="Path to an input image"
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Max generation length"
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--verbose", action="store_true", help="Show token-by-token output"
    )
    args = parser.parse_args()

    model_path = args.model_path or resolve_model_path(args.device, args.variant)

    if not Path(model_path).exists():
        print(f"ERROR: Model directory not found: {model_path}")
        print(
            "Run `olive run --config <cpu|cuda>/<variant>/config.json` first to generate the models."
        )
        sys.exit(1)
    if args.image and not Path(args.image).is_file():
        print(f"ERROR: Image file not found: {args.image}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    config = og.Config(model_path)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    print("Model loaded.")
    print()

    if args.interactive:
        interactive_mode(model, tokenizer, args.max_length)
    elif args.prompt:
        response = generate(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            system_prompt=args.system_prompt,
            image_path=args.image,
            verbose=args.verbose,
        )
        if not args.verbose:
            print(response)
    else:
        # Default demo
        demo_prompt = "What are the three laws of thermodynamics? Explain briefly."
        print(f"Demo prompt: {demo_prompt}")
        print()
        generate(
            model,
            tokenizer,
            demo_prompt,
            max_length=args.max_length,
            verbose=True,
        )


if __name__ == "__main__":
    main()
