# ruff: noqa: RUF001, RUF002
"""ONNX Runtime GenAI translation inference for Tencent Hy-MT2-1.8B.

Hy-MT2 is a multilingual (33-language) translation model. This script loads
a mobius-exported ONNX package (fp32 / fp16 / bf16 / int4) and translates a
source text into a target language using greedy decoding.

Tokenizer note: the Hy-MT BPE vocab uses a custom regex pre-tokenizer that
ort-extensions does not currently round-trip, so by default we tokenize and
detokenize with the HuggingFace tokenizer and feed raw token IDs to
``og.Generator`` (this still exercises the full ORT GenAI inference path).
Pass ``--use-ort-tokenizer`` to force the ``og.Tokenizer`` path.

Usage:
    python inference.py --source "今天天气真好。" --target-lang English
    python inference.py --device gpu --variant int4 --source "Hello world" \
        --target-lang Chinese
    python inference.py --device gpu --variant bf16 --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime_genai as og

HF_MODEL_ID = "tencent/Hy-MT2-1.8B"
# <｜hy_place▁holder▁no▁2｜>, the turn-end / EOS from generation_config.
EOS_TOKEN_ID = 120020

_RECIPE_DIR = Path(__file__).resolve().parent


def resolve_model_path(device: str, variant: str | None) -> str:
    """Resolve the model directory from device + variant args."""
    if device == "cpu":
        variant = variant or "fp32"
        return str(_RECIPE_DIR / "cpu" / variant / "models")
    variant = variant or "int4"
    return str(_RECIPE_DIR / "cuda" / variant / "models")


def format_prompt(tokenizer, source: str, target_lang: str) -> str:
    """Apply the Hy-MT chat template to a translation instruction."""
    instruction = (
        f"Translate the following text into {target_lang}. Note that you "
        f"should only output the translated result without any additional "
        f"explanation:\n\n{source}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def translate(
    model: og.Model,
    hf_tokenizer,
    source: str,
    target_lang: str,
    max_new_tokens: int,
) -> str:
    """Greedy-decode a single translation via the HF tokenizer + ORT GenAI."""
    prompt = format_prompt(hf_tokenizer, source, target_lang)
    input_tokens = hf_tokenizer.encode(prompt, add_special_tokens=False)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_tokens) + max_new_tokens, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    generated: list[int] = []
    while not generator.is_done() and len(generated) < max_new_tokens:
        generator.generate_next_token()
        token = int(generator.get_next_tokens()[0])
        if token == EOS_TOKEN_ID:
            break
        generated.append(token)
    return hf_tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--variant",
        choices=("fp32", "fp16", "bf16", "int4"),
        default=None,
        help="Precision variant. Defaults to fp32 (cpu) / int4 (gpu).",
    )
    parser.add_argument("--model-path", default=None, help="Override the model directory.")
    parser.add_argument("--source", default="黄河之水天上来", help="Text to translate.")
    parser.add_argument("--target-lang", default="English", help="Target language name.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path or resolve_model_path(args.device, args.variant)
    if not (Path(model_path) / "genai_config.json").exists():
        raise SystemExit(
            f"No genai_config.json in {model_path}. Build the recipe first, e.g.\n"
            f"  olive run --config {args.device}/"
            f"{args.variant or ('fp32' if args.device == 'cpu' else 'int4')}/config.json"
        )

    print(f"Loading ORT GenAI model from {model_path}")
    model = og.Model(model_path)

    import transformers

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(HF_MODEL_ID)

    if args.interactive:
        print("Interactive translation. Ctrl-C to exit.")
        print(f"Target language: {args.target_lang}\n")
        try:
            while True:
                source = input("source> ").strip()
                if not source:
                    continue
                out = translate(
                    model, hf_tokenizer, source, args.target_lang, args.max_new_tokens
                )
                print(f"  {out}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
        return

    out = translate(model, hf_tokenizer, args.source, args.target_lang, args.max_new_tokens)
    print(f"\nsource ({args.target_lang}): {args.source}")
    print(f"translation: {out}")


if __name__ == "__main__":
    main()
