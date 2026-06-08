# -------------------------------------------------------------------------
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------
# Export FLUX.2-klein-4B sub-models to ONNX via Olive.
#
# Usage:
#   python export_models.py [--models transformer vae_decoder text_encoder]
#                           [--model_id <hf_id_or_local_path>]
#                           [--resolutions 1024x1024]
#                           [--output_dir ./output_model]
#
# Output layout:
#   output_model/
#     transformer/dd/replaced.onnx   NPU (VitisAI)
#     vae_decoder/dd/replaced.onnx   NPU (VitisAI)
#     text_encoder/model.onnx        CPU ONNX
#     tokenizer/                     from pipeline
#     scheduler/                     from pipeline

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from olive.workflows import run as olive_run

SCRIPT_DIR = Path(__file__).parent.resolve()

DEFAULT_MODEL_ID   = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_RESOLUTIONS = ["1024x1024"]
ALL_MODELS = ["transformer", "vae_decoder", "text_encoder"]

NON_ONNX_COMPONENTS = ["tokenizer", "tokenizer_2", "scheduler", "feature_extractor"]


def set_dd_env() -> None:
    if os.environ.get("DD_PLUGINS_ROOT"):
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("ryzenai_dynamic_dispatch")
        if spec and spec.origin:
            dd_root = os.environ.get("DD_ROOT")
            if not dd_root or not os.path.exists(dd_root):
                os.environ["DD_ROOT"] = os.path.dirname(spec.origin).replace("\\", "/")
            bin_dir = os.path.join(os.path.dirname(spec.origin), "bin")
            if os.path.isdir(bin_dir):
                os.environ["DD_PLUGINS_ROOT"] = bin_dir
    except Exception:
        pass


def _fmt_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def update_config_files(model_id: str | None, resolutions: list[str] | None) -> None:
    for name in ALL_MODELS:
        config_path = SCRIPT_DIR / f"config_{name}.json"
        if not config_path.exists():
            continue
        with config_path.open() as f:
            cfg = json.load(f)

        changed = False
        if model_id is not None and cfg.get("input_model", {}).get("model_path") != model_id:
            cfg["input_model"]["model_path"] = model_id
            changed = True
        if resolutions is not None:
            for pass_cfg in cfg.get("passes", {}).values():
                if "resolutions" in pass_cfg and pass_cfg["resolutions"] != resolutions:
                    pass_cfg["resolutions"] = resolutions
                    changed = True

        if changed:
            with config_path.open("w") as f:
                json.dump(cfg, f, indent=4)
            print(f"  [CONFIG] Updated {config_path.name}")


def load_olive_config(submodel_name: str) -> dict:
    config_path = SCRIPT_DIR / f"config_{submodel_name}.json"
    with config_path.open() as f:
        return json.load(f)


def _find_npu_root(footprints_subdir: Path) -> Path | None:
    """Return the directory directly containing dd/replaced.onnx, or None."""
    for root in [footprints_subdir, footprints_subdir / "dynamic"]:
        if (root / "dd" / "replaced.onnx").exists():
            return root
    return None


def _find_plain_onnx(footprints_subdir: Path) -> Path | None:
    if not footprints_subdir.is_dir():
        return None
    candidates = sorted(footprints_subdir.rglob("model.onnx"), key=lambda p: len(p.parts))
    if candidates:
        return candidates[0]
    all_onnx = sorted(footprints_subdir.rglob("*.onnx"), key=lambda p: len(p.parts))
    return all_onnx[0] if all_onnx else None


def _copy_dir_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest_item = dst / item.name
        if item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)


def assemble_output_dir(
    pipeline,
    submodel_names: list[str],
    footprints_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in submodel_names:
        src_dir = footprints_dir / name
        dst_dir = output_dir / name

        npu_root = _find_npu_root(src_dir)
        if npu_root is not None:
            shutil.rmtree(dst_dir, ignore_errors=True)
            _copy_dir_contents(npu_root, dst_dir)
            print(f"  [COPY NPU]  {name} → {dst_dir}  (from {npu_root.relative_to(footprints_dir)})")
        else:
            onnx_src = _find_plain_onnx(src_dir)
            if onnx_src is None:
                print(f"  [WARN] No ONNX output found for {name} in {src_dir}; skipping.")
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(onnx_src, dst_dir / "model.onnx")
            for companion in onnx_src.parent.iterdir():
                if companion == onnx_src:
                    continue
                dest = dst_dir / companion.name
                if companion.is_dir():
                    shutil.copytree(companion, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(companion, dest)
            print(f"  [COPY CPU]  {name} → {dst_dir / 'model.onnx'}")

    for attr in NON_ONNX_COMPONENTS:
        component = getattr(pipeline, attr, None)
        if component is None:
            continue
        save_fn = getattr(component, "save_pretrained", None)
        if save_fn is None:
            continue
        dest = output_dir / attr
        dest.mkdir(parents=True, exist_ok=True)
        save_fn(str(dest))
        print(f"  [SAVE]  {attr} → {dest}")

    print(f"\n  Pipeline assembled at: {output_dir}")


def optimize(args) -> dict[str, bool]:
    model_id   = args.model_id
    output_dir = Path(args.output_dir).resolve()

    print(f"\n[PIPELINE] Loading Flux2KleinPipeline from '{model_id}' ...")
    from diffusers import Flux2KleinPipeline
    pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

    t_cfg   = pipeline.transformer.config
    vae_cfg = pipeline.vae.config
    print(f"  Transformer : in_channels={t_cfg.in_channels}, "
          f"joint_attention_dim={t_cfg.joint_attention_dim}, "
          f"num_layers={t_cfg.num_layers}")
    print(f"  VAE         : latent_channels={vae_cfg.latent_channels}, "
          f"scaling_factor={getattr(vae_cfg, 'scaling_factor', 'N/A')}")

    results: dict[str, bool] = {}
    total_t0 = time.monotonic()

    for submodel_name in args.models:
        print(f"\n{'=' * 60}\n  Exporting: {submodel_name}\n{'=' * 60}")
        olive_config = load_olive_config(submodel_name)
        t0 = time.monotonic()
        try:
            olive_run(olive_config)
            success = True
        except Exception as exc:
            print(f"\n[ERROR] {submodel_name} export failed: {exc}")
            success = False
        elapsed = time.monotonic() - t0
        results[submodel_name] = success
        print(f"\n  [{'OK' if success else 'FAILED'}]  {submodel_name}  ({_fmt_seconds(elapsed)})")

    total_elapsed = time.monotonic() - total_t0

    print(f"\n{'=' * 60}\n  Assembling output directory ...\n{'=' * 60}")
    assemble_output_dir(pipeline, args.models, SCRIPT_DIR / "footprints", output_dir)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\n  Export Summary\n{'=' * 60}")
    for name in args.models:
        print(f"  {'OK    ' if results.get(name) else 'FAILED'}  {name}")
    print(f"{'─' * 60}")
    print(f"  Total time : {_fmt_seconds(total_elapsed)}")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)

    return results


def parse_args(raw_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FLUX.2-klein-4B sub-models to ONNX via Olive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python export_models.py\n"
            "  python export_models.py --models transformer\n"
            "  python export_models.py --model_id /local/path/to/model\n"
            "  python export_models.py --output_dir /data/flux2_klein_onnx"
        ),
    )
    parser.add_argument(
        "--model_id", default=None, type=str,
        help=(
            "HuggingFace model ID or local path. "
            "When provided, writes back to all config_*.json. "
            f"Default: value in config_*.json (initially '{DEFAULT_MODEL_ID}')."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=None, metavar="MODEL",
        help=f"Sub-models to export (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--resolutions", nargs="+", default=None, metavar="WxH",
        help=(
            "Target resolutions for VitisGenerateModelSD. "
            "When provided, writes back to all config_*.json. "
            f"Default: value in config_*.json (initially '{' '.join(DEFAULT_RESOLUTIONS)}')."
        ),
    )
    parser.add_argument(
        "--output_dir", default=str(SCRIPT_DIR / "output_model"), type=str,
        help="Assembled pipeline output directory. Default: <script_dir>/output_model",
    )
    return parser.parse_args(raw_args)


def main(raw_args=None) -> None:
    set_dd_env()
    args = parse_args(raw_args)

    if args.models:
        args.models = [m for m in ALL_MODELS if m in args.models]
    else:
        args.models = list(ALL_MODELS)

    if args.model_id is not None or args.resolutions is not None:
        print(f"\n[CONFIG] Syncing config_*.json ...")
        update_config_files(args.model_id, args.resolutions)

    if args.model_id is None:
        first_cfg_path = SCRIPT_DIR / f"config_{args.models[0]}.json"
        with first_cfg_path.open() as f:
            args.model_id = json.load(f)["input_model"]["model_path"]

    if args.resolutions is None:
        args.resolutions = DEFAULT_RESOLUTIONS
        for name in args.models:
            with (SCRIPT_DIR / f"config_{name}.json").open() as f:
                cfg = json.load(f)
            for pass_cfg in cfg.get("passes", {}).values():
                if "resolutions" in pass_cfg:
                    args.resolutions = pass_cfg["resolutions"]
                    break
            else:
                continue
            break

    print("=" * 60)
    print("  FLUX.2-klein-4B  —  Olive ONNX Export")
    print("=" * 60)
    print(f"  model_id    : {args.model_id}")
    print(f"  sub-models  : {', '.join(args.models)}")
    print(f"  resolutions : {', '.join(args.resolutions)}")
    print(f"  output_dir  : {args.output_dir}")
    print("=" * 60)

    results = optimize(args)
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
