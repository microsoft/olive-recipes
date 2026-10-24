from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def load_logits(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        if not {"logits", "generated_tokens"}.issubset(archive.files):
            raise ValueError(
                f"{path}: expected benchmark NPZ keys logits and generated_tokens"
            )
        logits = archive["logits"]
        tokens = archive["generated_tokens"]
    if logits.ndim != 2 or not all(logits.shape) or logits.dtype.kind != "f":
        raise ValueError(
            f"{path}: logits must be nonempty floating [predicted_steps, vocabulary]"
        )
    if tokens.shape != (logits.shape[0],) or tokens.dtype.kind not in "iu":
        raise ValueError(
            f"{path}: generated_tokens must be an integer vector matching predicted steps"
        )
    if np.isfinite(logits).all() and not np.array_equal(
        tokens, np.argmax(logits, axis=-1)
    ):
        raise ValueError(f"{path}: generated_tokens do not match logits argmax")
    return logits, tokens


def compare(reference: Path, candidate: Path, atol=1e-2, rtol=1e-2):
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError("atol and rtol must be finite and nonnegative")
    reference_logits, reference_tokens = load_logits(reference)
    candidate_logits, candidate_tokens = load_logits(candidate)
    sides = {}
    for label, path, logits in (
        ("reference", reference, reference_logits),
        ("candidate", candidate, candidate_logits),
    ):
        finite = np.isfinite(logits)
        sides[label] = {
            "path": str(path.resolve()),
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
            "predicted_steps": logits.shape[0],
            "decode_steps": logits.shape[0] - 1,
            "finite": bool(finite.all()),
            "nonfinite_count": int(logits.size - finite.sum()),
        }
    shape_equal = reference_logits.shape == candidate_logits.shape
    finite = sides["reference"]["finite"] and sides["candidate"]["finite"]
    report = {
        **sides,
        "shape_equal": shape_equal,
        "finite": finite,
        "atol": atol,
        "rtol": rtol,
        "allclose": False,
        "max_abs": None,
        "mean_abs": None,
        "rmse": None,
        "top1_agreement": None,
        "top1_agreement_count": None,
        "top1_mismatch_count": None,
        "top1_mismatch_indices": None,
        "top1_reference": None,
        "top1_candidate": None,
        "scope": "End-to-end last-position logits, including the prefill prediction and subsequent decode "
        "predictions. Divergence does not establish isolated-layer correctness or incorrectness. "
        "These NPZ files do not establish identical prompts, forced decode inputs, or runtime settings; "
        "free-running decode may compare different token contexts after a prediction mismatch.",
    }
    if not shape_equal or not finite:
        report["reason"] = "shape mismatch" if not shape_equal else "non-finite logits"
        return report
    reference64 = reference_logits.astype(np.float64)
    candidate64 = candidate_logits.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        absolute = np.abs(candidate64 - reference64)
        maximum = float(absolute.max())
        if math.isfinite(maximum):
            mean = float((absolute / maximum).mean() * maximum) if maximum else 0.0
            rmse = (
                float(np.sqrt(np.mean((absolute / maximum) ** 2)) * maximum)
                if maximum
                else 0.0
            )
        else:
            mean = rmse = None
        close = bool(
            np.allclose(candidate64, reference64, atol=atol, rtol=rtol, equal_nan=False)
        )
    mismatch = np.flatnonzero(reference_tokens != candidate_tokens)
    steps = reference_logits.shape[0]
    report.update(
        {
            "allclose": close,
            "max_abs": maximum if math.isfinite(maximum) else None,
            "mean_abs": mean,
            "rmse": rmse,
            "top1_agreement": float((steps - mismatch.size) / steps),
            "top1_agreement_count": int(steps - mismatch.size),
            "top1_mismatch_count": int(mismatch.size),
            "top1_mismatch_indices": mismatch.tolist(),
            "top1_reference": reference_tokens.tolist(),
            "top1_candidate": candidate_tokens.tolist(),
        }
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare benchmark NPZ last-position logits (candidate against reference)"
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--output", "--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare(args.reference, args.candidate, args.atol, args.rtol)
    except (ValueError, OSError, BadZipFile, EOFError) as error:
        report = {"allclose": False, "error": str(error)}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if report["allclose"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
