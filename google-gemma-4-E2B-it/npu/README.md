# Gemma 4 vision A16W8 calibration recipe

This recipe consumes the vision encoder from the ONNX export of
`multi_comp/gemma4_quantize.json`. It modifies only the vision component;
`assemble_model.py` copies the other ONNX components unchanged.

Use one environment with local Olive, mobius, and onnxruntime-genai. Olive
must have `SplitVisionPooler` and the graph surgeries used below; mobius
must be at `223b438e8f6637c40ac883607f321a5ff4fbe863` with upstream
commit `6bd398de480c80042d57f37b40181b2a71ade5ad` cherry-picked
(the base commit alone cannot export the tied quantized `lm_head`).
Image inference requires the local ORT GenAI build
with two-stage Gemma 4 vision support.

The vision workflow consumes
`../multi_comp/gemma4_onnx/vision_encoder/model.onnx` and runs:

1. `MatMulNBitsToQDQ`
2. `GraphSurgeries` to simplify normalization, remove unused outputs, and replace
   large attention-mask values
3. `OnnxStaticQuantization` with UINT16 activations, INT8 weights, QDQ format,
   and MinMax calibration
4. `DynamicToFixedShape` with `num_patches = 2520`
5. `SplitVisionPooler` to split the fixed-shape encoder from the spatial
   pooler/projector, which produces a variable number of image features

The calibration set contains 128 images sampled evenly and deterministically
from eight `HuggingFaceM4/the_cauldron` subsets:

- Natural photographs: `vqav2`, `visual7w`
- Diagrams and charts: `ai2d`, `chartqa`
- Documents and scene text: `docvqa`, `textvqa`
- Screenshots and science imagery: `screen2words`, `scienceqa`

Images are streamed from Hugging Face instead of downloading the complete
Cauldron dataset. Each image is resized with its aspect ratio preserved,
and converted to Gemma 4's 16x16 patches. Calibration samples retain their
natural, variable patch lengths up to the configured 2,520-patch budget
(`max_soft_tokens = 280`, with nine patches per soft token), so activation
quantization preserves the model's dynamic input. The final pass independently
fixes that input dimension to 2,520. The split runs afterward: it finds the
unique tensor entering the named pooler that depends on `pixel_values`, names
that tensor `vision_features`, and keeps any positional-input shape logic
needed by both components. The result is a composite model with `encoder.onnx`
(inputs `pixel_values`, `pixel_position_ids`; output `vision_features`) and
`pooler_projector.onnx` (inputs `pixel_position_ids`, `vision_features`;
output `image_features`) under `models/vision_qdq`. Each component gets its own
external weight data. Removing the fixed-shape pass leaves the encoder dynamic.

By default, the loader takes the first 16 usable images from each streamed
subset (`shuffle_buffer_size = 0`) to avoid filling a large remote shuffle
buffer before calibration. Set `shuffle_buffer_size` to a positive value such
as 64 to enable approximate randomized sampling.

## Run

With that environment active, start in this `npu` directory and run the
workflows from their respective directories so relative paths and the
calibration script resolve:

```bash
cd ../multi_comp
olive run --config gemma4_quantize.json
olive capture-onnx-graph --model_name_or_path gemma4_quantized_hf --use_mobius_builder --precision fp32 --output_path gemma4_onnx
cd ../npu
olive run --config vision_config.json
python assemble_model.py
```

The model must be accessible through Hugging Face for the first workflow, and
vision calibration streams 128 images from `HuggingFaceM4/the_cauldron`.
The assembly script refuses to overwrite an existing `models/final` directory.

The split quantized vision models are written under `models/vision_qdq`.
Olive caches downloaded data and intermediate pass output under `cache`.

`models/final` copies the decoder, embedding, audio encoder, and package
sidecars unchanged from `multi_comp/gemma4_onnx`, adding only
`vision_encoder/model_encoder.onnx` and
`vision_encoder/model_pooler_projector.onnx` with their external weights. Its
`genai_config.json` routes vision through the two stages in order, passing
`vision_features` and `pixel_position_ids` to the projector. Absolute artifact
paths in `model_config.json` are rebased to the final directory.
The unsplit `vision_encoder/model.onnx` remains for compatibility with tooling
that reads the package metadata; released ORT GenAI wheels may parse the
pipeline without running its projector stage. The fixed-shape
encoder takes 2,520 patches; the local Gemma 4 image processor pads shorter
images to that size. This recipe creates QDQ graphs intended for QNN; confirm
provider compatibility and performance on a device with `QNNExecutionProvider`.

The upstream Mobius export's `audio_feature_extraction.json` is copied
unchanged. Mobius at the pinned commit emits `Gemma4LogMel`, which the local
ORT Extensions build supports.

To expand calibration after the 128-sample quality check, change
`samples_per_subset` in `vision_config.json` from 16 to 32 or 64 for a total of 256
or 512 images.

The input model already contains 114 INT4 `MatMulNBits` nodes. Converting those
nodes to QDQ preserves their INT4 weights; the static pass applies W8
quantization to eligible floating-point weights and A16 quantization to
eligible activations. The resulting graph is therefore mixed W4/W8 rather
than uniformly W8.
