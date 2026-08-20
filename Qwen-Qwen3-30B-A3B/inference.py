"""ONNX Runtime GenAI inference for a Qwen3-30B-A3B Mobius CUDA package.

This is a decoder-only (text-in / text-out) package: no vision encoder, image
processor, or multimodal pipeline is involved.

Run from the `Qwen-Qwen3-30B-A3B/` recipe root:

    python inference.py --prompt "What is the capital of France?"
    python inference.py --prompt "What is 17 * 23?" --no-think
    python inference.py --interactive
    python inference.py --model-path cuda/kquant_fp16/models --prompt "Hello" --verbose
"""

import argparse
import json
import sys
import time
from pathlib import Path

import onnxruntime_genai as og

DEFAULT_MODEL_PATH = "cuda/kquant_fp16/models"
DEFAULT_MAX_NEW_TOKENS = 1024
DEMO_PROMPT = "In two sentences, explain what a mixture-of-experts language model is."

# Qwen3 soft switch: suppresses the <think> reasoning trace for the turn it appears in.
NO_THINK_TAG = "/no_think"


def validate_model_path(model_path: str) -> Path:
    """Return the ORT GenAI model directory, exiting with a clear message if it is unusable."""
    path = Path(model_path)
    if not path.is_dir():
        sys.exit(
            f"ERROR: model directory not found: {path}\n"
            f"Run `olive run --config {path.parent / 'config.json'}` from the recipe root first."
        )
    if not (path / "genai_config.json").is_file():
        sys.exit(f"ERROR: {path} is not an ORT GenAI package: genai_config.json is missing.")
    return path


def positive_int(value: str) -> int:
    """Parse a positive integer for generation limits."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def apply_no_think(prompt: str, system_prompt: str | None) -> str:
    """Append Qwen3's `/no_think` switch unless the prompt or system prompt already carries it."""
    if NO_THINK_TAG in prompt or (system_prompt and NO_THINK_TAG in system_prompt):
        return prompt
    return f"{prompt} {NO_THINK_TAG}"


def build_chat_prompt(
    tokenizer: og.Tokenizer,
    prompt: str,
    system_prompt: str | None = None,
    no_think: bool = False,
) -> str:
    """Render one user turn with the chat template packaged in the model directory."""
    if no_think:
        prompt = apply_no_think(prompt, system_prompt)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # ORT GenAI expects the message list as a JSON string.
    return tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)


def generate(
    model: og.Model,
    tokenizer: og.Tokenizer,
    prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    no_think: bool = False,
    verbose: bool = False,
) -> str:
    """Stream one greedy response and report TTFT, decode throughput, and total time."""
    formatted = build_chat_prompt(tokenizer, prompt, system_prompt, no_think)
    input_ids = tokenizer.encode(formatted)
    max_length = len(input_ids) + max_new_tokens

    if verbose:
        print(f"  Input tokens     : {len(input_ids)}")
        print(f"  Max length       : {max_length} (input + {max_new_tokens} new tokens)")

    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=max_length)

    generator = og.Generator(model, params)
    stream = tokenizer.create_stream()

    tokens = []
    ttft = None
    print("Response: ", end="", flush=True)
    start = time.perf_counter()
    generator.append_tokens(input_ids)

    while not generator.is_done():
        generator.generate_next_token()
        if ttft is None:
            ttft = time.perf_counter() - start
        token = generator.get_next_tokens()[0]
        tokens.append(token)
        print(stream.decode(token), end="", flush=True)

    total = time.perf_counter() - start
    print()
    del generator

    decode_time = total - (ttft or total)
    decode_tps = (len(tokens) - 1) / decode_time if len(tokens) > 1 and decode_time > 0 else 0.0

    print(f"\n  Tokens generated : {len(tokens)}")
    print(f"  TTFT             : {(ttft or 0.0) * 1000:.1f} ms")
    print(f"  Decode TPS       : {decode_tps:.1f} tokens/sec")
    print(f"  Total time       : {total:.2f} s")

    return tokenizer.decode(tokens)


def interactive_mode(model: og.Model, tokenizer: og.Tokenizer, args: argparse.Namespace) -> None:
    """Run a stateless prompt loop: every turn is an independent single-turn chat."""
    print("Interactive mode. Each turn is independent (no history).")
    print("Type 'quit', 'exit', or press Ctrl-D to leave.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            break

        try:
            generate(
                model,
                tokenizer,
                prompt,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                no_think=args.no_think,
                verbose=args.verbose,
            )
        except KeyboardInterrupt:
            print("\n[generation interrupted]")
        print()

    print("Goodbye!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-30B-A3B Mobius CUDA ORT GenAI inference")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"ORT GenAI model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--prompt", help="Prompt to run once; omit to run the demo prompt")
    parser.add_argument("--system-prompt", help="Optional system prompt")
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Tokens to generate on top of the prompt (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument("--interactive", action="store_true", help="Run a stateless chat loop")
    parser.add_argument("--verbose", action="store_true", help="Print token counts and search settings")
    parser.add_argument(
        "--no-think",
        action="store_true",
        help=f"Append Qwen3's {NO_THINK_TAG} switch to skip the reasoning trace",
    )
    args = parser.parse_args()

    model_path = validate_model_path(args.model_path)

    print(f"Loading model from {model_path} ...")
    start = time.perf_counter()
    model = og.Model(str(model_path))
    tokenizer = og.Tokenizer(model)
    print(f"Model loaded in {time.perf_counter() - start:.1f} s.\n")

    if args.interactive:
        interactive_mode(model, tokenizer, args)
        return

    prompt = args.prompt or DEMO_PROMPT
    if not args.prompt:
        print("No --prompt given, running the demo prompt.")
    print(f"Prompt: {prompt}\n")

    generate(
        model,
        tokenizer,
        prompt,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        no_think=args.no_think,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
