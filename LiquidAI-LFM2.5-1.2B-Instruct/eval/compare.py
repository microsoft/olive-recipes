"""Paired comparison of two scored variants on the same tokens (per-token files from eval_onnx.py --dump
or score_gguf.py --dump).

Both variants see the same 64 chunks, so their difference is far less noisy than either mean. Tokens of
one chunk are correlated, so the error of the difference is taken over the 64 chunk means.

Usage: python compare.py A.npz B.npz [A2.npz B2.npz ...]
"""

import sys

import numpy as np


def load(path):
    d = np.load(path)
    return d["kld"], d["same_top"]


def main():
    paths = sys.argv[1:]
    if not paths or len(paths) % 2:
        raise SystemExit(__doc__)
    for a_path, b_path in zip(paths[::2], paths[1::2], strict=True):
        (ka, ta), (kb, tb) = load(a_path), load(b_path)
        if ka.shape != kb.shape:
            raise SystemExit(f"{a_path} and {b_path} cover different tokens")
        d = (ka - kb).mean(axis=1)  # per-chunk mean difference
        diff, err = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        top = 100 * (ta.mean() - tb.mean())
        top_err = 100 * (ta.astype(float) - tb).mean(axis=1).std(ddof=1) / np.sqrt(len(d))
        print(f"{a_path} vs {b_path}")
        print(
            f"  KLD {ka.mean():.4f} vs {kb.mean():.4f}: difference {diff:+.4f} +/- {err:.4f} "
            f"({100 * diff / kb.mean():+.1f}% +/- {100 * err / kb.mean():.1f}%)"
        )
        print(f"  same top {100 * ta.mean():.2f}% vs {100 * tb.mean():.2f}%: {top:+.2f} +/- {top_err:.2f} points")


if __name__ == "__main__":
    main()
