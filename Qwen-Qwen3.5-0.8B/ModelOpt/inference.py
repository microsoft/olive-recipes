# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Run text generation with a ModelOpt NVFP4 Hugging Face checkpoint using vLLM."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/nvfp4/model"))
    parser.add_argument("--prompt", default="Explain what NVFP4 quantization is in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_model_len <= 0:
        parser.error("--max-new-tokens and --max-model-len must be positive.")
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must be smaller than --max-model-len to leave room for the prompt.")
    if not (args.model_path / "config.json").is_file():
        parser.error(f"No Hugging Face config.json found in {args.model_path}. Run the Olive recipe first.")

    from vllm import LLM, SamplingParams

    llm = LLM(model=str(args.model_path.resolve()), max_model_len=args.max_model_len)
    responses = llm.chat(
        [{"role": "user", "content": args.prompt}],
        sampling_params=SamplingParams(temperature=0, max_tokens=args.max_new_tokens),
    )
    print(responses[0].outputs[0].text)


if __name__ == "__main__":
    main()
