"""Score ONNX Runtime GenAI packages against a llama.cpp KL-divergence base file.

Every chunk of the base file is prefilled through the package's decoder (for LFM2-VL, through its
embedding model first, text only) on the package's execution provider. The logits of the scored
half are compared with the base file's log-probabilities using llama-perplexity's statistics, so
the numbers line up with `llama-perplexity --kl-divergence` run against the same file.

Usage: python eval_onnx.py BASE.kld EP:PACKAGE_DIR [EP:PACKAGE_DIR ...] [--json OUT.jsonl] [--dump DIR]
                           [--chunks N] [--provider-option KEY=VALUE ...] [--tag TAG]
EP is cpu, cuda or webgpu. --dump writes the per-token values of every package for compare.py.
--provider-option goes to the CUDA or WebGPU execution provider (e.g. use_tf32=0).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import kld_base

_webgpu_device = None


def session(path: Path, ep: str, options: dict) -> ort.InferenceSession:
    global _webgpu_device
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if ep == "cpu":
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    elif ep == "cuda":
        s = ort.InferenceSession(str(path), so, providers=[("CUDAExecutionProvider", options), "CPUExecutionProvider"])
    elif ep == "webgpu":
        if _webgpu_device is None:
            import onnxruntime_ep_webgpu as w

            ort.register_execution_provider_library(w.get_ep_name(), w.get_library_path())
            _webgpu_device = [d for d in ort.get_ep_devices() if d.ep_name == w.get_ep_name()][0]
        so.add_provider_for_devices([_webgpu_device], options)
        s = ort.InferenceSession(str(path), so)
    else:
        raise SystemExit(f"unknown EP {ep}")
    expected = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "webgpu": "WebGpuExecutionProvider"}[ep]
    if s.get_providers()[0] != expected:
        raise SystemExit(f"{path}: wanted {expected}, session has {s.get_providers()}")
    return s


def np_type(s: ort.InferenceSession, name: str):
    t = next(i.type for i in s.get_inputs() if i.name == name)
    return np.float16 if "float16" in t else np.float32


def files_size(paths) -> int:
    total = 0
    for p in paths:
        for f in Path(p).parent.glob(Path(p).name + "*"):
            total += f.stat().st_size
    return total


class Package:
    def __init__(self, spec: str, options: dict):
        self.ep, path = spec.split(":", 1)
        self.dir = Path(path).resolve()
        self.name = f"{self.dir.parent.parent.name}/{self.dir.parent.name}/{self.dir.name}"
        cfg = json.loads((self.dir / "genai_config.json").read_text())["model"]
        self.dc = cfg["decoder"]
        self.decoder = session(self.dir / self.dc["filename"], self.ep, options)
        self.inputs = {i.name for i in self.decoder.get_inputs()}
        self.embedding = None
        files = [self.dir / self.dc["filename"]]
        if "inputs_embeds" in self.inputs:
            self.embedding = session(self.dir / cfg["embedding"]["filename"], self.ep, options)
            files.append(self.dir / cfg["embedding"]["filename"])
        self.size = files_size(files)
        self.stats = kld_base.KldStats()
        self.seconds = 0.0

    def logits(self, ids: np.ndarray) -> np.ndarray:
        dc, n = self.dc, ids.shape[1]
        feeds = {}
        if self.embedding is not None:
            hidden = dc["hidden_size"]
            feats = np.zeros((0, hidden), np_type(self.embedding, "image_features"))
            embeds = self.embedding.run(None, {"input_ids": ids, "image_features": feats})[0]
            feeds["inputs_embeds"] = embeds.astype(np_type(self.decoder, "inputs_embeds"))
        else:
            feeds["input_ids"] = ids
        if "attention_mask" in self.inputs:
            feeds["attention_mask"] = np.ones((1, n), np.int64)
        if "position_ids" in self.inputs:
            feeds["position_ids"] = np.arange(n, dtype=np.int64)[None]
        names = dc["inputs"]
        for i, layer in enumerate(dc["layer_types"]):
            if layer == "full_attention":
                for key in ("past_key_names", "past_value_names"):
                    name = names[key] % i
                    feeds[name] = np.zeros(
                        (1, dc["num_key_value_heads"], 0, dc["head_size"]), np_type(self.decoder, name)
                    )
            else:
                name = names["past_conv_names"] % i
                feeds[name] = np.zeros((1, dc["hidden_size"], dc["conv_cache_size"]), np_type(self.decoder, name))
        return self.decoder.run([dc["outputs"]["logits"]], feeds)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("packages", nargs="+")
    parser.add_argument("--json")
    parser.add_argument("--dump")
    parser.add_argument("--chunks", type=int, default=0)
    parser.add_argument("--provider-option", action="append", default=[])
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    options = dict(o.split("=", 1) for o in args.provider_option)

    n_ctx, n_vocab, tokens, rows = kld_base.open_rows(args.base)
    n_chunk = min(args.chunks or len(tokens), len(tokens))
    first = n_ctx // 2
    packages = [Package(spec, options) for spec in args.packages]
    meta = json.loads(Path(args.base + ".json").read_text())  # written by make_ref.py
    t0 = time.time()
    for c in range(n_chunk):
        ids = tokens[c].astype(np.int64)[None].copy()
        if meta["add_bos"]:
            ids[0, 0] = meta["bos"]  # llama-perplexity puts the BOS in front of every chunk
        base = kld_base.decode_rows(np.asarray(rows[c]), n_vocab)
        next_tokens = tokens[c, first + 1 : n_ctx]
        for p in packages:
            t = time.time()
            logits = p.logits(ids)[first : n_ctx - 1, :n_vocab].astype(np.float32)
            p.seconds += time.time() - t
            p.stats.add(logits, base, next_tokens)
        if (c + 1) % 8 == 0 or c + 1 == n_chunk:
            print(f"chunks {c + 1}/{n_chunk} ({time.time() - t0:.0f}s)", flush=True)
    out = open(args.json, "a") if args.json else None
    for p in packages:
        s = {
            "package": p.name,
            "ep": p.ep,
            "tag": args.tag,
            "options": options,
            "size_gib": p.size / 2**30,
            "seconds": p.seconds,
            **p.stats.summary(),
        }
        print(json.dumps(s), flush=True)
        if out:
            out.write(json.dumps(s) + "\n")
        if args.dump:
            Path(args.dump).mkdir(parents=True, exist_ok=True)
            name = p.name.replace("/", ".") + f".{p.ep}" + (f".{args.tag}" if args.tag else "")
            np.savez_compressed(Path(args.dump) / f"{name}.npz", **p.stats.arrays())
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
