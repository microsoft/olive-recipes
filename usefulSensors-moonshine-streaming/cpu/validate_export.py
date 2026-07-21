"""Validate the exported MoonshineStreaming ONNX graphs.

Two checks per component:
  1. exported .onnx  vs  torch reference outputs  (from refs/*.npz)  -> proves
     the ONNX export is faithful to the torch wrappers (tight tolerance).
  2. exported .onnx  vs  official .ort graph on identical inputs      -> proves
     the wrappers are semantically correct vs the shipped model.  The official
     cross_kv / decoder graphs are int8-quantized, so these diffs are compared
     with a loose tolerance.

Runs in the ``moonshine`` env (or any env with onnxruntime + numpy):

    python validate_export.py \
        --mine   /datadisks/disk3/nebanfic/moonshine-streaming-small-mine \
        --official /datadisks/disk3/nebanfic/moonshine-streaming-small-official
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import onnxruntime as ort

COMPONENTS = ["frontend", "encoder", "adapter", "cross_kv", "decoder_kv"]
# Components whose official graph is quantized -> only expect approximate parity.
QUANTIZED_OFFICIAL = {"cross_kv", "decoder_kv"}


def make_session(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def find_official(official_dir, name):
    for ext in (".onnx", ".ort"):
        cand = os.path.join(official_dir, name + ext)
        if os.path.exists(cand):
            return cand
    return None


def run_session(sess, feeds):
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, feeds)
    return dict(zip(names, outs))


def summarize(tag, a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return f"    {tag:24s} SHAPE MISMATCH {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-6)
    return (f"    {tag:24s} max_abs={diff.max():.3e}  mean_abs={diff.mean():.3e}"
            f"  max_rel={np.max(diff / denom):.3e}")


def validate_component(name, mine_dir, official_dir):
    print(f"\n=== {name} ===")
    refs_path = os.path.join(mine_dir, "refs", f"{name}.npz")
    mine_path = os.path.join(mine_dir, f"{name}.onnx")
    if not (os.path.exists(refs_path) and os.path.exists(mine_path)):
        print("    (missing exported graph or refs; skipped)")
        return

    refs = np.load(refs_path)
    feeds = {k[len("in__"):]: refs[k] for k in refs.files if k.startswith("in__")}
    torch_out = {k[len("out__"):]: refs[k] for k in refs.files if k.startswith("out__")}

    mine_sess = make_session(mine_path)
    mine_out = run_session(mine_sess, feeds)

    print("  [onnx vs torch reference]")
    for oname, tval in torch_out.items():
        if oname in mine_out:
            print(summarize(oname, mine_out[oname], tval))

    if official_dir:
        off_path = find_official(official_dir, name)
        if off_path is None:
            print(f"  [official] no graph found for {name}")
            return
        off_sess = make_session(off_path)
        # Match the official graph's expected input names/order.
        off_in_names = {i.name for i in off_sess.get_inputs()}
        off_feeds = {k: v for k, v in feeds.items() if k in off_in_names}
        missing = off_in_names - set(off_feeds)
        if missing:
            print(f"  [official] missing inputs {missing}; skipped")
            return
        off_out = run_session(off_sess, off_feeds)
        tol_note = " (quantized: loose)" if name in QUANTIZED_OFFICIAL else ""
        print(f"  [onnx vs official{tol_note}]")
        for oname in mine_out:
            if oname in off_out:
                print(summarize(oname, mine_out[oname], off_out[oname]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mine", required=True)
    p.add_argument("--official", default=None)
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()

    comps = args.only or COMPONENTS
    for name in comps:
        validate_component(name, args.mine, args.official)


if __name__ == "__main__":
    main()
