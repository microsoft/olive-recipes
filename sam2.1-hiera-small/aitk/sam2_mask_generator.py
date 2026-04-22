# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import os

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

def get_mask_ort(
    sess_ve, sess_md, image, box, ve_dtype, md_dtype, sess_ve_inputs, sess_md_inputs
):
    w, h = image.size
    processed_image = sam2_transform(image)
    inputs = processed_image.float().numpy()[None, :]
    ort_pixel_values = inputs

    box = np.array(box)
    box_coords = box.reshape(2, 2)
    box_coords[:, 0] = box_coords[:, 0] * 1024 / w
    box_coords[:, 1] = box_coords[:, 1] * 1024 / h

    box_labels = np.array([2, 3])
    blank_points = np.zeros([3, 2])
    blank_labels = -np.ones(3)
    ort_input_points = np.concatenate([blank_points, box_coords], axis=0)[None, :]
    ort_input_labels = np.concatenate([blank_labels, box_labels], axis=0)[None, :]

    input_ve = {sess_ve_inputs[0].name: np.array(ort_pixel_values, dtype=ve_dtype)}
    image_embedding, high_res_features1, high_res_features2 = sess_ve.run(
        None, input_ve
    )

    input_md = {
        sess_md_inputs[0].name: image_embedding.astype(md_dtype),
        sess_md_inputs[1].name: high_res_features1.astype(md_dtype),
        sess_md_inputs[2].name: high_res_features2.astype(md_dtype),
        sess_md_inputs[3].name: ort_input_points.astype(md_dtype),
        sess_md_inputs[4].name: ort_input_labels.astype(md_dtype),
    }

    result_md = sess_md.run(None, input_md)
    pred_masks = result_md[0]
    scores = result_md[1]

    masks = (
        F.interpolate(
            torch.Tensor(pred_masks), size=(h, w), mode="bilinear", align_corners=False
        )
        .detach()
        .numpy()
    )

    pred_max_ind = np.argmax(scores)
    mask = masks[0, pred_max_ind]
    return np.array(mask, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Run SAM ONNX models and save mask.")
    parser.add_argument("--execution_provider", type=str, default="CPUExecutionProvider", help="ORT Execution provider")
    parser.add_argument("--device_str", type=str, default="cpu")
    parser.add_argument(
        "--model_ve", required=True, help="Path to vision encoder ONNX model"
    )
    parser.add_argument(
        "--model_md", required=True, help="Path to mask decoder ONNX model"
    )
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument(
        "--output_path",
        default="mask_output.png",
        help="Path to save the output mask image",
    )
    parser.add_argument(
        "--box_x", type=int, default=40, help="Top-Left X coordinate of input box"
    )
    parser.add_argument(
        "--box_y", type=int, default=235, help="Top-Left Y coordinate of input box"
    )
    parser.add_argument("--box_w", type=int, default=940, help="Width of input box")
    parser.add_argument("--box_h", type=int, default=490, help="Height of input box")
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
    raw_image = Image.open(args.image_path).convert("RGB")
    input_box = [
        [args.box_x, args.box_y],
        [args.box_x + args.box_w, args.box_y + args.box_h],
    ]

    # Load models
    sess_ve = ort.InferenceSession(args.model_ve, sess_options=sess_options)
    sess_md = ort.InferenceSession(args.model_md, sess_options=sess_options)

    sess_ve_inputs = sess_ve.get_inputs()
    sess_md_inputs = sess_md.get_inputs()

    ve_dtype = np.float32 if sess_ve_inputs[0].type == "tensor(float)" else np.float16
    md_dtype = np.float32 if sess_md_inputs[0].type == "tensor(float)" else np.float16

    # Get mask
    mask = get_mask_ort(
        sess_ve,
        sess_md,
        raw_image,
        input_box,
        ve_dtype,
        md_dtype,
        sess_ve_inputs,
        sess_md_inputs,
    )

    # Save mask using PIL
    mask_img = Image.fromarray(mask * 255)  # Convert binary mask to 0-255
    mask_img.save(args.output_path)
    print(f"Mask saved to {args.output_path}")


if __name__ == "__main__":
    main()
