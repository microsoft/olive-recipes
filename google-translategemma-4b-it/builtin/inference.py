# -------------------------------------------------------------------------
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import argparse
import sys
import io
from pathlib import Path

import onnxruntime_genai as og
from transformers import AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Run TranslateGemma-4B-IT inference (text or image translation)")
parser.add_argument("--model-dir", default=str(SCRIPT_DIR / "cpu_and_mobile" / "models"),
                    help="Path to the ONNX model directory containing genai_config.json")
parser.add_argument("--hf-model-dir", default="google/translategemma-4b-it",
                    help="HF model name or local path for tokenizer/chat template")
parser.add_argument("--source-lang", default="en", help="Source language code (default: en)")
parser.add_argument("--target-lang", default="fr", help="Target language code (default: fr)")
parser.add_argument("--text", default="The weather is beautiful today. Let's go for a walk in the park.",
                    help="Text to translate (for text mode)")
parser.add_argument("--image", default=None, help="Path to image file (enables image translation mode)")
parser.add_argument("--max-length", type=int, default=512, help="Maximum generation length")
args = parser.parse_args()

print("Loading model...")
model = og.Model(args.model_dir)
processor = model.create_multimodal_processor()
stream = processor.create_stream()

hf_tok = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)

if args.image:
    # Image translation mode
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_lang_code": args.source_lang,
                    "target_lang_code": args.target_lang,
                }
            ],
        }
    ]
    prompt = hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = og.Images.open(args.image)
    inputs = processor(prompt, images=images)
    print(f"Image: {args.image}")
    print(f"Translation ({args.source_lang} -> {args.target_lang}): ", end="", flush=True)
else:
    # Text translation mode
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": args.source_lang,
                    "target_lang_code": args.target_lang,
                    "text": args.text,
                }
            ],
        }
    ]
    prompt = hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt)
    print(f"Source ({args.source_lang}): {args.text}")
    print(f"Translation ({args.target_lang}): ", end="", flush=True)

params = og.GeneratorParams(model)
params.set_search_options(max_length=args.max_length, temperature=1.0, top_k=64, top_p=0.95)

generator = og.Generator(model, params)
generator.set_inputs(inputs)

while not generator.is_done():
    generator.generate_next_token()
    token = generator.get_next_tokens()[0]
    print(stream.decode(token), end="", flush=True)

print("\nDone.")
