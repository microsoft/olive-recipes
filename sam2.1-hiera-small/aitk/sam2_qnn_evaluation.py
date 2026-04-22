# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import os
import time
from urllib import request

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Load processor
sam2_transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),  # Resize to 1024x1024
        transforms.ToTensor(),  # Convert to tensor [C,H,W] and scale to [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize
    ]
)

def add_ep_for_device(session_options, ep_name, device_type, ep_options=None):
    ep_devices = ort.get_ep_devices()
    for ep_device in ep_devices:
        if ep_device.ep_name == ep_name and ep_device.device.type == device_type:
            print(f"Adding {ep_name} for {device_type}")
            session_options.add_provider_for_devices([ep_device], {} if ep_options is None else ep_options)
            break

def register_execution_providers():
    import json
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

def test_mask_ort(
    sess_ve, sess_md, image, ve_dtype, md_dtype, sess_ve_inputs, sess_md_inputs
):
    w, h = image.size
    processed_image = sam2_transform(image)
    inputs = processed_image.float().numpy()[None, :]
    ort_pixel_values = inputs

    box = np.array([[0, 0], [w, h]])
    box_coords = box.reshape(2, 2)
    box_coords[:, 0] = box_coords[:, 0] * 1024 / w
    box_coords[:, 1] = box_coords[:, 1] * 1024 / h

    box_labels = np.array([2, 3])
    blank_points = np.zeros([3, 2])
    blank_labels = -np.ones(3)
    ort_input_points = np.concatenate([blank_points, box_coords], axis=0)[None, :]
    ort_input_labels = np.concatenate([blank_labels, box_labels], axis=0)[None, :]

    input_ve = {sess_ve_inputs[0].name: np.array(ort_pixel_values, dtype=ve_dtype)}

    encoder_start = time.perf_counter()
    image_embedding, high_res_features1, high_res_features2 = sess_ve.run(
        None, input_ve
    )
    encoder_latency = time.perf_counter() - encoder_start

    input_md = {
        sess_md_inputs[0].name: image_embedding.astype(md_dtype),
        sess_md_inputs[1].name: high_res_features1.astype(md_dtype),
        sess_md_inputs[2].name: high_res_features2.astype(md_dtype),
        sess_md_inputs[3].name: ort_input_points.astype(md_dtype),
        sess_md_inputs[4].name: ort_input_labels.astype(md_dtype),
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
