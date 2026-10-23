"""Paired comparison of two scored variants on the same tokens (per-token files from eval_onnx.py --dump
or score_gguf.py --dump).

Both variants see the same 64 chunks. Tokens of one chunk are correlated, so errors are taken over the 64 chunk
means: the error of the difference from the spread of the per-chunk differences, and the error of the relative
change from the delta method for the ratio of the two means, which accounts for the error of the denominator and
its covariance with the numerator.

Usage: python compare.py A.npz B.npz [A2.npz B2.npz ...]
"""

import sys

import numpy as np


def load(path):
    d = np.load(path)
    return d["kld"], d["same_top"]


def ratio_err(a, b):
    """Standard error of mean(a) / mean(b) over paired chunk means (delta method)."""
    r = a.mean() / b.mean()
    return (a - r * b).std(ddof=1) / np.sqrt(len(a)) / b.mean()


def main():
    paths = sys.argv[1:]
    if not paths or len(paths) % 2:
        raise SystemExit(__doc__)
    for a_path, b_path in zip(paths[::2], paths[1::2], strict=True):
        (ka, ta), (kb, tb) = load(a_path), load(b_path)
        if ka.shape != kb.shape:
            raise SystemExit(f"{a_path} and {b_path} cover different tokens")
        ca, cb = ka.mean(axis=1), kb.mean(axis=1)  # per-chunk means; every chunk scores the same number of tokens
        d = ca - cb
        diff, err = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        rel, rel_err = ca.mean() / cb.mean() - 1, ratio_err(ca, cb)
        top = 100 * (ta.mean() - tb.mean())
        top_err = 100 * (ta.astype(float) - tb).mean(axis=1).std(ddof=1) / np.sqrt(len(d))
        print(f"{a_path} vs {b_path}")
        print(
            f"  KLD {ca.mean():.4f} vs {cb.mean():.4f}: difference {diff:+.4f} +/- {err:.4f} "
            f"({100 * rel:+.1f}% +/- {100 * rel_err:.1f}%)"
        )
        print(f"  same top {100 * ta.mean():.2f}% vs {100 * tb.mean():.2f}%: {top:+.2f} +/- {top_err:.2f} points")


if __name__ == "__main__":
    main()
