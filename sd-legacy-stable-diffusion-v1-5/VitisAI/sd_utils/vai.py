# -------------------------------------------------------------------------
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------
"""
NPU (VitisAI) pipeline save and load: save_npu_pipeline, get_vai_pipeline.
Unet/vae_decoder use DD session options (dd_cache, onnx_custom_ops_const_key); other submodels use OnnxRuntimeModel.from_pretrained.
"""

import importlib
import json
import shutil
from pathlib import Path
import sd_utils.config

import onnxruntime as ort
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline
from transformers import CLIPTokenizer

# ruff: noqa: TID252, T201
def update_vai_config(config: dict, provider: str, submodel_name: str):
    if provider != "vitisai":
        raise ValueError(f"Unsupported provider: {provider}. Only vitisai is supported.")

    used_passes = {}
    if sd_utils.config.only_conversion:
        used_passes = {"convert"}
        config["evaluator"] = None

    if submodel_name in ("text_encoder", "vae_encoder", "safety_checker"):
        used_passes = {"convert", "optimize", "dynamic_shape_to_fixed"}
    elif submodel_name in ("unet", "vae_decoder"):
        used_passes = {"convert", "model_generation"}

    for pass_name in set(config["passes"].keys()):
        if pass_name not in used_passes:
            config["passes"].pop(pass_name, None)

    config["systems"] = {
        "local_system": {
            "type": "LocalSystem",
            "config": {
                "accelerators": [
                    {"execution_providers": ["CPUExecutionProvider"], "device": "cpu"},
                ],
            },
        },
    }
    return config


def save_vai_pipeline(
    has_safety_checker, model_info, optimized_model_dir, unoptimized_model_dir, pipeline, submodel_names
):
    """Save VitisAI pipeline: unet/vae_decoder as full dir (dd/, cache/); others as model.onnx."""
    print("\nCreating ONNX pipeline (VitisAI)...")

    # diffusers >= 0.30 expects provider_options in kwargs (kwargs.pop without default).
    _cpu_onnx_kw = {"providers": ["CPUExecutionProvider"], "provider_options": [{}]}

    if has_safety_checker:
        safety_checker = OnnxRuntimeModel.from_pretrained(
            str(model_info["safety_checker"]["unoptimized"]["path"].parent), **_cpu_onnx_kw
        )
    else:
        safety_checker = None

    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(
            str(model_info["vae_encoder"]["unoptimized"]["path"].parent), **_cpu_onnx_kw
        ),
        vae_decoder=OnnxRuntimeModel.from_pretrained(
            str(model_info["vae_decoder"]["unoptimized"]["path"].parent), **_cpu_onnx_kw
        ),
        text_encoder=OnnxRuntimeModel.from_pretrained(
            str(model_info["text_encoder"]["unoptimized"]["path"].parent), **_cpu_onnx_kw
        ),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(str(model_info["unet"]["unoptimized"]["path"].parent), **_cpu_onnx_kw),
        scheduler=pipeline.scheduler,
        safety_checker=safety_checker,
        feature_extractor=pipeline.feature_extractor,
        requires_safety_checker=True,
    )

    print("Saving unoptimized models...")
    onnx_pipeline.save_pretrained(unoptimized_model_dir)

    print("Copying optimized models (VitisAI: full dir for unet/vae_decoder)...")
    shutil.copytree(unoptimized_model_dir, optimized_model_dir, ignore=shutil.ignore_patterns("weights.pb"))

    NPU_SUBMODELS = ("unet", "vae_decoder")
    for submodel_name in submodel_names:
        src = Path(model_info[submodel_name]["optimized"]["path"])
        dst_subdir = optimized_model_dir / submodel_name

        if submodel_name in NPU_SUBMODELS:
            src_dir = src.parent if src.suffix == ".onnx" else src
            if not src_dir.is_dir():
                src_dir = src.parent
            shutil.rmtree(dst_subdir, ignore_errors=True)
            dst_subdir.mkdir(parents=True, exist_ok=True)
            for item in src_dir.iterdir():
                dest_item = dst_subdir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest_item)
        else:
            if src.is_file() and src.suffix == ".onnx":
                src_path = src
            else:
                src_path = (src if src.is_dir() else src.parent) / "model.onnx"
            if not src_path.exists():
                raise FileNotFoundError(f"Optimized model not found: {src_path}")
            shutil.copyfile(src_path, dst_subdir / "model.onnx")

            src_ctx_path = Path(str(src_path).replace(".onnx", "_qnn.bin"))
            if src_ctx_path.exists():
                shutil.copyfile(src_ctx_path, dst_subdir / "model_ctx_qnn.bin")

    print(f"The optimized NPU pipeline is located here: {optimized_model_dir}")


def _load_npu_model(model_dir, submodel_name):
    """Load unet or vae_decoder from model_dir/<submodel>/dd/replaced.onnx using VitisAI EP."""
    model_dir = Path(model_dir)
    replaced_onnx_path = model_dir / submodel_name / "dd" / "replaced.onnx"
    if not replaced_onnx_path.exists():
        raise FileNotFoundError(f"NPU optimized model not found: {replaced_onnx_path}")
    sess_opts = ort.SessionOptions()
    cache_dir = (replaced_onnx_path.parent / "cache").as_posix()
    sess_opts.add_session_config_entry("dd_cache", cache_dir)
    sess_opts.add_provider("VitisAIExecutionProvider", {})
    session = ort.InferenceSession(
        str(replaced_onnx_path), sess_options=sess_opts, providers=["VitisAIExecutionProvider"]
    )
    model = OnnxRuntimeModel(session)
    config_path = model_dir / submodel_name / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            model.config = json.load(f)
    else:
        model.config = {}
    return model


def get_vai_pipeline(model_dir, common_args):
    """Build pipeline for VitisAI: unet/vae_decoder from dd/replaced.onnx; text_encoder/vae_encoder on CPU."""
    ort.set_default_logger_severity(3)
    model_dir = Path(model_dir)
    provider = common_args.provider
    if provider != "vitisai":
        raise ValueError(f"Unsupported provider: {provider}. Only vitisai is supported.")

    unet = _load_npu_model(model_dir, "unet")
    vae_decoder = _load_npu_model(model_dir, "vae_decoder")
    print("Loading NPU pipeline (unet/vae_decoder from dd/replaced.onnx with VitisAI EP, text_encoder/vae_encoder with CPU EP)...")

    text_encoder_dir = model_dir / "text_encoder"
    vae_encoder_dir = model_dir / "vae_encoder"
    if not text_encoder_dir.exists() or not vae_encoder_dir.exists():
        raise FileNotFoundError(f"Missing text_encoder or vae_encoder under {model_dir}")

    # Must pass provider_options so diffusers' from_pretrained (kwargs.pop) does not raise KeyError; values must be dicts.
    text_encoder = OnnxRuntimeModel.from_pretrained(
        str(text_encoder_dir), providers=["CPUExecutionProvider"], provider_options=[{}]
    )
    vae_encoder = OnnxRuntimeModel.from_pretrained(
        str(vae_encoder_dir), providers=["CPUExecutionProvider"], provider_options=[{}]
    )
    tokenizer = CLIPTokenizer.from_pretrained(str(model_dir / "tokenizer"))

    with (model_dir / "scheduler" / "scheduler_config.json").open() as f:
        scheduler_name = json.load(f).get("_class_name", "PNDMScheduler")
    scheduler_cls = getattr(importlib.import_module("diffusers.schedulers"), scheduler_name)
    scheduler = scheduler_cls.from_pretrained(str(model_dir / "scheduler"))

    feature_extractor = None
    if (model_dir / "feature_extractor").exists():
        from transformers import CLIPImageProcessor
        feature_extractor = CLIPImageProcessor.from_pretrained(str(model_dir / "feature_extractor"))

    safety_checker = None
    if (model_dir / "safety_checker").exists():
        try:
            safety_checker = OnnxRuntimeModel.from_pretrained(
                str(model_dir / "safety_checker"),
                providers=["CPUExecutionProvider"],
                provider_options=[{}]
            )
        except Exception:
            pass

    return OnnxStableDiffusionPipeline(
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        requires_safety_checker=False,
    )
