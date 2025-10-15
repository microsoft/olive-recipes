# Olive Recipes CI

Simple CI pipeline for testing Olive recipes.

## What It Does

- Runs every Friday at 7 AM PT
- Tests 12 recipes across Linux CPU, Linux GPU, and Windows CPU
- Verifies output models are generated
- Uploads artifacts for 7 days

## Recipes Tested

**Linux CPU (4)**:
- PTQ
- INC Smooth Quant
- DeBERTa
 TensorRT

**Linux GPU (9)**:
-  Dora, HQQ, Lmeval, Lmeval-ONNX, Loha, Lokr, QLoRA, RTN

**Windows CPU (3)**:
- Same as Linux CPU

## Manual Testing

```bash
# Test locally
cd <recipe_path>
python -m olive run --config <config>.json

# Check output
python -c "from olive.common import WorkflowOutput; print(WorkflowOutput.from_json('olive-output/output.json').has_output_model())"
```

## Manual Trigger

```bash
gh workflow run ci.yml
gh workflow run ci.yml -f test_scope=linux-cpu-only
```

## Requirements

- Set secret: `HUGGING_FACE_HUB_TOKEN` for gated models
- Self-hosted GPU runner with label `linux, A100` for GPU tests

## Adding Recipes

Edit `ci-simple.yml` and add to the appropriate matrix:

```yaml
- name: "My Recipe"
  path: "model/folder"
  config: "config.json"
```
