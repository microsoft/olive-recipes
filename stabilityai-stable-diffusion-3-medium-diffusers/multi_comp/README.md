# Stable Diffusion 3 Medium — Multi-Component Optimization

This recipe demonstrates a **multi-component flow** for
[Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers):
export the pipeline to ONNX once with the Mobius builder, then run a single Olive config whose
`builds` apply a **different pipeline to each component**.

The flow is two explicit steps:

1. **Export** the model to a directory of per-component ONNX subfolders using the Olive CLI with the
   Mobius builder.
2. **Optimize** by pointing an Olive config at that directory; each component subfolder becomes a
   selectable component that a `build` can target.

There is no need to memorize component names: each exported component lives in its own folder, and
Olive loads the export directory as a `CompositeModel` whose **component names are the subfolder
names**.

## Prerequisites

```
pip install olive-ai
pip install mobius-ai
```

Exporting a diffusion pipeline also needs `diffusers`/`transformers` and access to the model on
Hugging Face (Stable Diffusion 3 is a gated model — accept its license and `huggingface-cli login`
first).

## Step 1 — Export with the CLI

```
olive capture-onnx-graph --model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers --use_mobius_builder --output_path exported_pkg
```

Mobius exports each neural-network component to its own subfolder:

```
exported_pkg/
  text_encoder/model.onnx    # CLIP-L text encoder
  text_encoder_2/model.onnx  # CLIP-G text encoder
  text_encoder_3/model.onnx  # T5-XXL text encoder
  transformer/model.onnx     # MMDiT denoising backbone
  vae_encoder/model.onnx
  vae_decoder/model.onnx
```

> **Note.** The exact subfolders depend on the pipeline; the optimize config below only
> needs `builds` for the components you actually want to optimize.

## Step 2 — Optimize each component

Run from the directory that contains `exported_pkg/`:

```
olive run --config sd3_optimize_components.json
```

This applies a different pipeline per component:

| component        | pipeline                     | intent                                     |
|------------------|------------------------------|--------------------------------------------|
| `transformer`    | `OrtTransformersOptimization`| FP16-optimize the heavy denoising backbone |
| `vae_encoder`    | `OrtTransformersOptimization`| FP16-optimize the VAE encoder              |
| `vae_decoder`    | `OrtTransformersOptimization`| FP16-optimize the VAE decoder              |

Output:

```
out/transformer/    # optimized transformer
out/vae_encoder/    # optimized VAE encoder
out/vae_decoder/    # optimized VAE decoder
```

Each build writes one optimized component; components without a build stay as exported.

## Step 3 — Inference

Run end-to-end image generation with the exported ONNX models:

```
python sd3_inference.py --prompt "A photo of a cat sitting on a windowsill" --steps 28 --output result.png
```

The inference script (`sd3_inference.py`) uses:
- **Text encoding**: ONNX Runtime with exported CLIP-L, CLIP-G, and T5-XXL encoders (run once)
- **Denoising**: ONNX Runtime with the exported SD3 transformer (28 steps)
- **VAE decoding**: ONNX Runtime with the exported VAE decoder

Options:
```
--prompt TEXT       Text prompt for image generation
--steps N           Number of denoising steps (default: 28)
--seed N            Random seed (default: 42)
--output PATH       Output image path (default: sd3_output.png)
--onnx_dir DIR      Path to exported model directory (default: exported_sd3_full2)
```

> **Note.** SD3 is a gated model — you need `huggingface-cli login` or set `HF_TOKEN` to export.
> The tokenizers (CLIP and T5) still run via the `transformers` library.

## Notes

- The passes here are **illustrative**. Swap in `OnnxStaticQuantization` (with a `data_config`),
  `OnnxDynamicQuantization`, or other ONNX passes for production-quality optimization.
- `builds.components` selects which exported components to optimize. Only the components with a build
  are touched; the rest remain as exported.
