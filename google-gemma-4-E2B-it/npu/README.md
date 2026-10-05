# Gemma 4 multimodal QNN build

Run `olive run --config npu/config_qnn.json` from the
`google-gemma-4-E2B-it` directory after exporting
`multi_comp/gemma4_onnx`. Use the local Olive checkout with multi-build
CompositeModel assembly support, and a Windows ARM64
environment with QNN for context-binary generation.

The two component builds run serially into `npu/output/.builds`. The
decoder build uses the complete original text recipe: graph surgery,
quantization, layer splitting, fixed KV shapes, `StaticLLM`, weight-sharing
QNN context binaries, and `ComposeOnnxModels`. Olive's decoder-only build
inherits the source `genai_config.json` as an additional file, so `StaticLLM`
and `ComposeOnnxModels` can update the decoder pipeline without replacing the
Gemma 4 multimodal model type. Only the decoder component carries this file;
the vision build remains independent and can reuse its cache. The package
assembler merges the generated decoder settings into the source package
config. The vision build converts existing INT4 weights to QDQ, simplifies the
graph, calibrates A16/W8 on **CPU**, then fixes the vision input to 2,520
patches. Decoder activation calibration also uses `CPUExecutionProvider`;
only decoder context-binary compilation requires QNN.

Olive assembles the output package with the untouched audio and embedding
components, composed decoder stage graphs under `decoder/`, and optimized
`vision_encoder/model.onnx`. `genai_config.json` selects QNN for the
context/iterator stages and the single vision model; embedding and LM head
stages remain on CPU. The complete package is in `npu/output`. The
assembler also supports unsplit decoder outputs and separate vision graphs.

For a CPU-only wiring check, make a temporary copy of this config outside the
package output, remove `cb` from the decoder build pipeline, select
`cpu_system` as the target, and choose a different `output_dir` and
`cache_dir`. Do not change the committed QNN recipe for that check. This
does not validate QNN compilation or on-device inference.
