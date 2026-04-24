# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import argparse
import json
import os
import time
from urllib import request
from PIL import Image
import onnxruntime as ort
from transformers import SamProcessor

# Load processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

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

def test_mask_ort(sess_ve, sess_md, image, ve_dtype, md_dtype, sess_ve_inputs, sess_md_inputs):
    w, h = image.size
    inputs = processor(image, input_boxes=[[[0, 0], [w, h]]], return_tensors="np")
    ort_pixel_values = inputs["pixel_values"]
    input_boxes = inputs["input_boxes"]

    ort_input_points = input_boxes.reshape(1, 1, 2, 2)
    ort_input_labels = np.array([2, 3]).reshape(1, 1, 2)

    input_ve = {sess_ve_inputs[0].name: np.array(ort_pixel_values, dtype=ve_dtype)}

    encoder_start = time.perf_counter()
    result_ve = sess_ve.run(None, input_ve)
    encoder_latency = time.perf_counter() - encoder_start

    input_md = {
        sess_md_inputs[0].name: np.array(ort_input_points, dtype=md_dtype),
        sess_md_inputs[1].name: np.array(ort_input_labels, dtype=md_dtype),
        sess_md_inputs[2].name: np.array(result_ve[0], dtype=md_dtype)
    }

    decoder_start = time.perf_counter()
    sess_md.run(None, input_md)
    decoder_latency = time.perf_counter() - decoder_start

    return encoder_latency, decoder_latency


def main():
    parser = argparse.ArgumentParser(description="Test SAM ONNX models.")
    parser.add_argument("--execution_provider", type=str, default="CPUExecutionProvider", help="ORT Execution provider")
    parser.add_argument("--device_str", type=str, default="cpu")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_ve", required=True, help="Path to vision encoder ONNX model")
    parser.add_argument("--model_md", required=True, help="Path to mask decoder ONNX model")
    args = parser.parse_args()

    # Loading models into ORT session
    register_execution_providers()
    sess_options = ort.SessionOptions()

    device_map = {
        "cpu": ort.OrtHardwareDeviceType.CPU,
        "gpu": ort.OrtHardwareDeviceType.GPU,
        "npu": ort.OrtHardwareDeviceType.NPU,
    }

    add_ep_for_device(sess_options, args.execution_provider, device_map[args.device_str])

    # Load image
    test_image_url = "https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/dog.jpg?raw=true"
    test_image_name = "dog.jpg"
    request.urlretrieve(test_image_url, test_image_name)
    raw_image = Image.open(test_image_name).convert("RGB")

    # Load models
    sess_ve = ort.InferenceSession(args.model_ve, sess_options=sess_options)
    sess_md = ort.InferenceSession(args.model_md, sess_options=sess_options)

    sess_ve_inputs = sess_ve.get_inputs()
    sess_md_inputs = sess_md.get_inputs()

    ve_dtype = np.float32 if sess_ve_inputs[0].type == "tensor(float)" else np.float16
    md_dtype = np.float32 if sess_md_inputs[0].type == "tensor(float)" else np.float16

    encoder_latencies = []
    decoder_latencies = []

    for _ in range(10):
        # Test mask
        encoder_latency, decoder_latency = test_mask_ort(sess_ve, sess_md, raw_image, ve_dtype, md_dtype, sess_ve_inputs, sess_md_inputs)
        encoder_latencies.append(encoder_latency)
        decoder_latencies.append(decoder_latency)

    encoder_latency_avg = round(sum(encoder_latencies) / len(encoder_latencies) * 1000, 5)
    decoder_latency_avg = round(sum(decoder_latencies) / len(decoder_latencies) * 1000, 5)

    metrics = {
        "vision-encoder-latency-avg": encoder_latency_avg,
        "mask-decoder-latency-avg": decoder_latency_avg
    }
    resultStr = json.dumps(metrics, indent=4)
    with open(args.output_file, 'w') as file:
        file.write(resultStr)
    print("Model lab succeeded for evaluation.\n%s", resultStr)


if __name__ == "__main__":
    main()
