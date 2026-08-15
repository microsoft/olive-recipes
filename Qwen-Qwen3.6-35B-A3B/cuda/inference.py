"""ONNX Runtime GenAI inference for Qwen3.6-35B-A3B (CUDA, RTN fp16 export).

Usage:
    python inference.py --prompt "What is the capital of France?"
    python inference.py --interactive
    python inference.py --model-path /path/to/models --prompt "Hello"
"""

import argparse
import time
from pathlib import Path

import onnxruntime_genai as og

DEFAULT_MODEL_PATH = "cuda/rtn_fp16/models"


def generate(
    model: og.Model,
    tokenizer: og.Tokenizer,
    prompt: str,
    max_length: int = 2048,
    verbose: bool = False,
) -> str:
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt)

    if verbose:
        print(f"  Input tokens: {len(input_ids)}")

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length, do_sample=False, top_k=1)

    start = time.time()
    generator = og.Generator(model, params)
    generator.append_tokens([input_ids])

    output_tokens = []
    tokenizer_stream = tokenizer.create_stream()

    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        output_tokens.append(token)

        if verbose:
            print(tokenizer_stream.decode(token), end="", flush=True)

    elapsed = time.time() - start
    output_text = tokenizer.decode(output_tokens)

    if verbose:
        print()
        tps = len(output_tokens) / elapsed if elapsed > 0 else 0
        print(f"  Output tokens: {len(output_tokens)}, Time: {elapsed:.2f}s, Speed: {tps:.1f} tok/s")

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

        print("Assistant: ", end="", flush=True)
        generate(model, tokenizer, prompt, max_length=max_length, verbose=True)
        print()


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-35B-A3B ORT GenAI Inference")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Model directory")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--max-length", type=int, default=2048, help="Max generation length")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--verbose", action="store_true", help="Show token-by-token output")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"ERROR: Model directory not found: {args.model_path}")
        return 1

    print(f"Loading model from {args.model_path} ...")
    model = og.Model(args.model_path)
    tokenizer = og.Tokenizer(model)

    if args.interactive:
        interactive_mode(model, tokenizer, args.max_length)
        return 0

    prompt = args.prompt or "The quick brown fox"
    print(f"Prompt: {prompt}")
    output = generate(model, tokenizer, prompt, max_length=args.max_length, verbose=True)
    if not args.verbose:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
