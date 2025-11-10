# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SamProcessor

# Load processor
processor = SamProcessor.from_pretrained("QNN/facebook/sam-vit-base")


def get_mask_ort(sess_ve, sess_md, image, box, ve_dtype, md_dtype, sess_ve_inputs, sess_md_inputs):
    w, h = image.size
    inputs = processor(image, return_tensors="np")
    ort_pixel_values = inputs["pixel_values"]

    box_coords = box.reshape(2, 2)
    box_labels = np.array([2, 3])
    blank_points = np.zeros([3, 2])
    blank_labels = -np.ones(3)
    ort_input_points = np.concatenate([blank_points, box_coords], axis=0)[None, :]
    ort_input_labels = np.concatenate([blank_labels, box_labels], axis=0)[None, :]

    input_ve = {sess_ve_inputs[0].name: np.array(ort_pixel_values, dtype=ve_dtype)}
    image_embedding, high_res_features1, high_res_features2 = sess_ve.run(None, input_ve)

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

    masks = F.interpolate(torch.Tensor(pred_masks), size=(h, w), mode="bilinear", align_corners=False).detach().numpy()

    pred_max_ind = np.argmax(scores)
    mask = masks[0, pred_max_ind]
    return np.array(mask, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Run SAM ONNX models and save mask.")
    parser.add_argument("--model_ve", required=True, help="Path to vision encoder ONNX model")
    parser.add_argument("--model_md", required=True, help="Path to mask decoder ONNX model")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--output_path", default="mask_output.png", help="Path to save the output mask image")
    parser.add_argument("--box_x", type=int, default=40, help="Top-Left X coordinate of input box")
    parser.add_argument("--box_y", type=int, default=235, help="To-Left Y coordinate of input box")
    parser.add_argument("--box_w", type=int, default=940, help="Width of input box")
    parser.add_argument("--box_h", type=int, default=490, help="Height of input box")
    args = parser.parse_args()

    # Load image
    raw_image = Image.open(args.image_path).convert("RGB")
    input_box = [[args.box_x, args.box_y], [args.box_x + args.box_w, args.box_y + args.box_h]]

    # Load models
    sess_ve = ort.InferenceSession(args.model_ve, providers=["QNNExecutionProvider"])
    sess_md = ort.InferenceSession(args.model_md, providers=["QNNExecutionProvider"])

    sess_ve_inputs = sess_ve.get_inputs()
    sess_md_inputs = sess_md.get_inputs()

    ve_dtype = np.float32 if sess_ve_inputs[0].type == "tensor(float)" else np.float16
    md_dtype = np.float32 if sess_md_inputs[0].type == "tensor(float)" else np.float16

    # Get mask
    mask = get_mask_ort(sess_ve, sess_md, raw_image, input_box, ve_dtype, md_dtype, sess_ve_inputs, sess_md_inputs)

    # Save mask using PIL
    mask_img = Image.fromarray(mask * 255)  # Convert binary mask to 0-255
    mask_img.save(args.output_path)
    print(f"Mask saved to {args.output_path}")


if __name__ == "__main__":
    main()
