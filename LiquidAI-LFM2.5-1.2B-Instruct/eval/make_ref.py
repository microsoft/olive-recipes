"""Write a llama.cpp KL-divergence base file from the Hugging Face model in FP32.

Takes the tokens from a base file llama-perplexity wrote (so both frameworks score the same
token stream), runs every chunk through the PyTorch model exactly as llama-perplexity would
(BOS in place of the first token, the second half of the window scored), and writes the
log-probabilities in llama.cpp's format. llama-perplexity --kl-divergence then scores GGUFs
against it, and eval_onnx.py scores ONNX packages against the same file.

Usage: python make_ref.py MODEL_ID LLAMA_BASE.kld OUT.kld [--vl] [--device cpu|cuda|mps]
"""

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

import kld_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("llama_base")
    parser.add_argument("out")
    parser.add_argument("--vl", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    n_ctx, n_vocab, n_chunk, tokens, _ = kld_base.read_header(args.llama_base)
    bos = AutoTokenizer.from_pretrained(args.model).bos_token_id
    add_bos = int(tokens[0, 0]) == bos
    print(f"n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk} bos={bos} add_bos={add_bos}", flush=True)

    if args.device == "cuda":  # keep the reference in true FP32: no TF32 in matmuls or the short convs
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cls = AutoModelForImageTextToText if args.vl else AutoModelForCausalLM
    model = cls.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    first = n_ctx // 2
    t0 = time.time()
    with open(args.out, "wb") as f, torch.inference_mode():
        kld_base.write_header(f, n_ctx, n_vocab, tokens)
        for start in range(0, n_chunk, args.batch):
            ids = torch.from_numpy(tokens[start : start + args.batch].astype(np.int64))
            if add_bos:
                ids[:, 0] = bos
            logits = model(input_ids=ids.to(args.device)).logits[:, first : n_ctx - 1, :n_vocab]
            logits = logits.float().cpu().numpy()
            if logits.shape[-1] != n_vocab:
                raise SystemExit(f"model vocab {logits.shape[-1]} != GGUF n_vocab {n_vocab}")
            for chunk in logits:
                f.write(kld_base.encode_rows(chunk, n_vocab).tobytes())
            print(f"chunks {start + len(ids)}/{n_chunk} ({time.time() - t0:.0f}s)", flush=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"model": args.model, "add_bos": add_bos, "bos": bos, "dtype": "float32", "device": args.device}, f)
    print("REF_DONE", flush=True)


if __name__ == "__main__":
    main()
