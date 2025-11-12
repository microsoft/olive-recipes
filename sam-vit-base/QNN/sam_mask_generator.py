# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import argparse
from PIL import Image
import onnxruntime as ort
from transformers import SamProcessor

# Load processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def get_mask_ort(sess_ve, sess_md, image, box, ve_dtype, md_dtype, sess_ve_inputs, sess_md_inputs):
    inputs = processor(image, input_boxes = box, return_tensors="np")
    ort_pixel_values = inputs["pixel_values"]
    input_boxes = inputs["input_boxes"]

    ort_input_points = input_boxes.reshape(1, 1, 2, 2)
    ort_input_labels = np.array([2, 3]).reshape(1, 1, 2)

    input_ve = {sess_ve_inputs[0].name: np.array(ort_pixel_values, dtype=ve_dtype)}
    result_ve = sess_ve.run(None, input_ve)

    input_md = {
        sess_md_inputs[0].name: np.array(ort_input_points, dtype=md_dtype),
        sess_md_inputs[1].name: np.array(ort_input_labels, dtype=md_dtype),
        sess_md_inputs[2].name: np.array(result_ve[0], dtype=md_dtype)
    }

    result_md = sess_md.run(None, input_md)
    scores = result_md[0]
    pred_masks = result_md[1]

    masks = processor.image_processor.post_process_masks(
        [pred_masks[0]],
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )[0].numpy()

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
    input_box = [[[args.box_x, args.box_y], [args.box_x + args.box_w, args.box_y + args.box_h]]]

    # Load models
    sess_ve = ort.InferenceSession(args.model_ve, providers=['QNNExecutionProvider'])
    sess_md = ort.InferenceSession(args.model_md, providers=['QNNExecutionProvider'])

    sess_ve_inputs = sess_ve.get_inputs()
    sess_md_inputs = sess_md.get_inputs()

    ve_dtype = np.float32 if sess_ve_inputs[0].type == 'tensor(float)' else np.float16
    md_dtype = np.float32 if sess_md_inputs[0].type == 'tensor(float)' else np.float16

    # Get mask
    mask = get_mask_ort(sess_ve, sess_md, raw_image, input_box, ve_dtype, md_dtype, sess_ve_inputs, sess_md_inputs)

    # Save mask using PIL
    mask_img = Image.fromarray(mask * 255)  # Convert binary mask to 0-255
    mask_img.save(args.output_path)
    print(f"Mask saved to {args.output_path}")

if __name__ == "__main__":
    main()
