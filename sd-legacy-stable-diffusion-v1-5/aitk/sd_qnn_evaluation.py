# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from sd_utils.qdq import OnnxStableDiffusionPipelineWithSave

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)

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
        default="CPUExecutionProvider",
        help="ORT Execution provider",
    )
    parser.add_argument(
        "--device_str",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    return parser.parse_args(raw_args)

def get_device_type(device_str):
    if device_str.lower() == "gpu":
        return ort.OrtHardwareDeviceType.GPU
    elif device_str.lower() == "npu":
        return ort.OrtHardwareDeviceType.NPU
    else:
        return ort.OrtHardwareDeviceType.CPU

def add_ep_for_device(session_options, ep_name, device_type, ep_options=None):
    ep_devices = ort.get_ep_devices()
    for ep_device in ep_devices:
        if ep_device.ep_name == ep_name and ep_device.device.type == device_type:
            print(f"Adding {ep_name} for {device_type}")
            session_options.add_provider_for_devices([ep_device], {} if ep_options is None else ep_options)
            break

def register_execution_providers():
    import subprocess
    import sys

    worker_script = os.path.abspath('winml.py')
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    for item in paths.items():
        try:
            ort.register_execution_provider_library(item[0], item[1])
        except Exception as e:
            print(f"Failed to register execution provider {item[0]}: {e}")


def main(raw_args=None):
    args = parse_args(raw_args)

    prompts = ["A baby is laying down with a teddy bear"]
    model_dir = Path(args.script_dir) / "models" / args.model_dir / args.model_id

    register_execution_providers()

    sess_options = ort.SessionOptions()
    provider_options = [{}]

    add_ep_for_device(sess_options, args.execution_provider, get_device_type(args.device_str))

    pipeline = OnnxStableDiffusionPipelineWithSave.from_pretrained(
        model_dir, provider=args.execution_provider, sess_options=sess_options, provider_options=provider_options
    )
    pipeline.save_data_dir = None

    text_encoder_latencies = []
    unet_latencies = []
    vae_decoder_latencies = []

    generator = None if args.seed is None else np.random.RandomState(seed=args.seed)

    # we know that PatchedOnnxRuntimeModel records latencies for each call
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
        "vae-decoder-latency-avg": vae_decoder_latency_avg
    }
    resultStr = json.dumps(metrics, indent=4)
    with open(args.output_file, 'w') as file:
        file.write(resultStr)
    logger.info("Model lab succeeded for evaluation.\n%s", resultStr)

if __name__ == "__main__":
    main()
