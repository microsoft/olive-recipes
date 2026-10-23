"""Per-token KL divergence of a GGUF from the reference, from two llama.cpp KL-divergence base files.

`llama-perplexity --kl-divergence` prints only means. To compare a GGUF with an ONNX package token by
token (compare.py), let llama-perplexity write the GGUF's own log-probabilities as a base file:

    llama-perplexity -m MODEL-Q4_K_M.gguf -f wiki.test.raw -c 512 --chunks 64 --kl-divergence-base q4_k_m.kld

then score that file against the reference here. The GGUF's log-probabilities are stored at 16 bits over
the top 16 nats, so its mean KLD lands within a fraction of a percent of what llama-perplexity prints.

Usage: python score_gguf.py REFERENCE.kld GGUF_LOGPROBS.kld --label NAME [--json OUT.jsonl] [--dump DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import kld_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("gguf")
    parser.add_argument("--label", required=True)
    parser.add_argument("--json")
    parser.add_argument("--dump")
    args = parser.parse_args()

    n_ctx, n_vocab, tokens, ref_rows = kld_base.open_rows(args.reference)
    n_ctx_q, n_vocab_q, tokens_q, q_rows = kld_base.open_rows(args.gguf)
    if (n_ctx, n_vocab) != (n_ctx_q, n_vocab_q) or not np.array_equal(tokens, tokens_q[: len(tokens)]):
        raise SystemExit("the two files do not hold the same token stream")
    first = n_ctx // 2
    stats = kld_base.KldStats()
    for c in range(len(tokens)):
        base = kld_base.decode_rows(np.asarray(ref_rows[c]), n_vocab)
        log_q = kld_base.decode_rows(np.asarray(q_rows[c]), n_vocab)  # already log-probabilities
        stats.add_log_probs(log_q, base, tokens[c, first + 1 : n_ctx])
    s = {"package": args.label, "ep": "gguf", **stats.summary()}
    print(json.dumps(s), flush=True)
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(s) + "\n")
    if args.dump:
        Path(args.dump).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(args.dump) / f"{args.label}.npz", **stats.arrays())


if __name__ == "__main__":
    main()
