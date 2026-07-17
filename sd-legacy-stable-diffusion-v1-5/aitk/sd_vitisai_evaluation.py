# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from diffusers import OnnxRuntimeModel
from sd_utils.vai import get_vai_pipeline

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)

# ruff: noqa: TID252, T201


def parse_args(raw_args):
    import argparse

    parser = argparse.ArgumentParser("Common arguments")

    parser.add_argument("--script_dir", required=True, type=str)
    parser.add_argument("--model_dir", default="optimized", type=str, help="model_dir path")
    parser.add_argument("--model_id", default="stable-diffusion-v1-5/stable-diffusion-v1-5", type=str)
    parser.add_argument(
        "--guidance_scale",
        default=7.5,
        type=float,
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance",
    )
    parser.add_argument("--num_inference_steps", default=25, type=int, help="Number of steps in diffusion process")
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="The seed to give to the generator to generate deterministic results.",
    )
    parser.add_argument("--image_size", default=512, type=int, help="Width and height of the images to generate")
    parser.add_argument(
        "--execution_provider",
        type=str,
        default="VitisAIExecutionProvider",
        help="ORT Execution provider",
    )
    parser.add_argument(
        "--device_str",
        type=str,
        default="npu",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    return parser.parse_args(raw_args)


def _patch_latency_recording():
    """Record per-call latency on every OnnxRuntimeModel instance (vitisai pipeline components
    are plain OnnxRuntimeModel, unlike the ov/qnn PatchedOnnxRuntimeModel subclasses)."""
    original_call = OnnxRuntimeModel.__call__

    def timed_call(self, **kwargs):
        start = time.perf_counter()
        result = original_call(self, **kwargs)
        if not hasattr(self, "latencies"):
            self.latencies = []
        self.latencies.append(time.perf_counter() - start)
        return result

    OnnxRuntimeModel.__call__ = timed_call


def main(raw_args=None):
    args = parse_args(raw_args)

    prompts = ["A baby is laying down with a teddy bear"]
    model_dir = Path(args.script_dir) / "model" / args.model_dir / args.model_id

    from winml import register_execution_providers

    register_execution_providers()

    _patch_latency_recording()

    pipeline = get_vai_pipeline(model_dir, SimpleNamespace(provider="vitisai"))

    text_encoder_latencies = []
    unet_latencies = []
    vae_decoder_latencies = []

    generator = None if args.seed is None else np.random.RandomState(seed=args.seed)

    for _ in range(10):
        pipeline.text_encoder.latencies = []
        pipeline.unet.latencies = []
        pipeline.vae_decoder.latencies = []
        pipeline(
            prompts,
            num_inference_steps=args.num_inference_steps,
            height=args.image_size,
            width=args.image_size,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        text_encoder_latencies.extend(pipeline.text_encoder.latencies)
        unet_latencies.extend(pipeline.unet.latencies)
        vae_decoder_latencies.extend(pipeline.vae_decoder.latencies)

    text_encoder_latency_avg = round(sum(text_encoder_latencies) / len(text_encoder_latencies) * 1000, 5)
    unet_latency_avg = round(sum(unet_latencies) / len(unet_latencies) * 1000, 5)
    vae_decoder_latency_avg = round(sum(vae_decoder_latencies) / len(vae_decoder_latencies) * 1000, 5)

    metrics = {
        "text-encoder-latency-avg": text_encoder_latency_avg,
        "unet-latency-avg": unet_latency_avg,
        "vae-decoder-latency-avg": vae_decoder_latency_avg,
    }
    resultStr = json.dumps(metrics, indent=4)
    with open(args.output_file, "w") as file:
        file.write(resultStr)
    logger.info("Model lab succeeded for evaluation.\n%s", resultStr)


if __name__ == "__main__":
    main()
