# Llama 3.1-8B-Instruct Pipeline Migration — Hand-off

## What this is

A migration of the `meta-llama-Llama-3.1-8B-Instruct` QAIRT recipe in
`olive-recipes` from the legacy 3-pass workflow (`QairtPreparation →
QairtGenAIBuilder → QairtEncapsulation`) to the 2-pass `QairtPipelinePass`
workflow that matches the verified Phi-4-reasoning recipe.

The recipe files are pushed and ready to run. **The recipe has not yet been
validated on device** — that is the primary remaining task.

---

## Repository

- **Repo:** `CodeLinaro/olive-recipes` (pushes → `microsoft/olive-recipes` PR)
- **Branch:** `dev/qti-kromero/llama3.1-8b-instruct-pipeline`
- **Path:** `meta-llama-Llama-3.1-8B-Instruct/QAIRT/`

```
x_elite_config.json          # Olive config — SC8380XP (Snapdragon X Elite)
x_elite_config.yaml          # QairtPipeline YAML recipe — SC8380XP
x2_elite_config.json         # Olive config — SC8480XP (Snapdragon X2 Elite)
x2_elite_config.yaml         # QairtPipeline YAML recipe — SC8480XP
mixed_precision_exceptions.json  # Mixed precision overrides for pipeline flow
info.yml
README.md
```

Reference: Phi-4-reasoning recipe on `dev/qti-kromero/phi4-reasoning` is
the verified PoC — use it as the template if debugging recipe structure issues.

---

## Environment setup

```bash
git clone git@github.com:CodeLinaro/olive-recipes.git
cd olive-recipes
git checkout dev/qti-kromero/llama3.1-8b-instruct-pipeline

pip install olive-ai[qairt]
pip install qairt-dev[pipeline,onnx]==0.11.0
# optionally pin QAIRT SDK:
export QAIRT_SDK_ROOT=/path/to/qairt/2.45.40

huggingface-cli login   # Llama is gated; HF_TOKEN env var also works
```

---

## Running the recipe

```bash
cd meta-llama-Llama-3.1-8B-Instruct/QAIRT

# Snapdragon X2 Elite (SC8480XP):
olive run --config x2_elite_config.json

# Snapdragon X Elite (SC8380XP):
olive run --config x_elite_config.json
```

---

## What the recipe does

Two-stage quantization translating the legacy `llama3_1_instruct_script.py`:

1. **`quantization_opt` (AdaScale):** `ada_scale_hf` recipe, context_length=2048,
   C4 dataset (500 samples), 5000 iterations. Returns a clean HF model with
   AdaScale-optimized weights.

2. **`quantization` (LPBQ, lm_head only):** `lpbq_seqmse` recipe, SeqMSE disabled,
   LPBQ restricted to lm_head via `module_precisions.lm_head.apply: true`, mixed
   precision from `mixed_precision_exceptions.json` (embedding int8, lm_head int4,
   RMSNorm int16).

3. **`genai_builder`:** HTP compilation with `native_kv`, `weight_sharing`,
   `split.split_embedding/lm_head`. Extended UDMA enabled on SC8480XP only.

---

## Open validation items — check these on device

These are the unknowns that cannot be confirmed without a run. Surface any
failures back to the branch.

1. **LPBQ scope** — `module_precisions.lm_head.apply: true` is intended to
   restrict LPBQ topology to lm_head only (matching `apply_lm_head_lpbq=True,
   apply_decoder_lpbq=False` in the legacy script). Confirm `LPBQQuantizer
   .setup_blockwise` honors this; if LPBQ lands on decoder layers too, quality
   will differ from the legacy baseline.

2. **AdaScale context_length** — The `quantization_opt` stage has
   `context_length: 2048`. Confirm `QuantizationOptStage._resolve_lengths` uses
   this stage-level value rather than falling back to the top-level
   `generator_config.context_length: 4096`. If it falls back, AdaScale runs with
   a 2× longer context than the legacy script intended.

3. **AdaScale num_batches** — Currently `num_batches: 20`. The legacy script uses
   500 C4 samples at batch_size=2 (250 batches × 20 cycles = 5000 steps). The
   pipeline fixes batch_size=1, so 500 samples = 500 batches × 10 cycles. If
   quality is noticeably below the legacy baseline, try `num_samples: 250` to
   restore the 20-cycle cadence.

4. **exceptions_json path** — `./mixed_precision_exceptions.json` is resolved
   relative to the YAML recipe file. Confirm this path is found when
   `LLMPipeline` runs; if not, pass an absolute path or update the resolution
   logic.

5. **SC8380XP encapsulation overrides** — `x_elite_config.json` sets
   `n_threads: 3`, `weight_sharing_enabled: true`, and
   `memory.mem_type: shared_buffer` matching the legacy `htp_sc8380xp.json`.
   Confirm these are still appropriate for qairt-dev 0.11.0; the prior pipeline
   branch had `n_threads: 0` which may have been intentional.

---

## Differences from legacy script worth noting

| Aspect | Legacy script | New recipe |
|---|---|---|
| AdaScale dataset | C4 explicit download | `dataset: {name: c4}` in pipeline |
| Quantization algorithm | LPBQ (lm_head only) | `lpbq_seqmse` with `apply_seqmse: false` + `lm_head.apply: true` |
| Embedding bitwidth | 8-bit (runtime override) | 8-bit in `mixed_precision_exceptions.json` |
| Model preparation | Explicit `prepare_model()` call | Handled internally by `LPBQQuantizer` |
| SC8380XP PLATFORM_GEN | `PLATFORM_GEN=2` via script_config | Implicit via `soc_details: chipset:SC8380XP` |

---

## If the recipe fails to start

- Confirm `qairt-dev[pipeline,onnx]==0.11.0` is installed — the `[pipeline]`
  extra is required for `QuantizationOptStage` support.
- If `QuantizationOptStage` is unrecognized, the installed qairt-dev may be older
  than 0.11.0; check `pip show qairt-dev`.
- If AdaScale fails on context shapes, try explicitly setting
  `sequence_length: 2048` in the `quantization_opt` stage alongside
  `context_length: 2048`.
