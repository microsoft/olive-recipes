# LFM2.5 text recipes: quality measurements

The "Measured quality" sections of the LFM2.5 text recipe READMEs
([230M](../../LiquidAI-LFM2.5-230M), [350M](../../LiquidAI-LFM2.5-350M), [1.2B-Instruct](..),
[2.6B](../../LiquidAI-LFM2.5-2.6B)) come from the scripts in this folder. This page describes the method, the
measurements behind the notes in those sections, and how to repeat them.

## Method

- **Text**: the wikitext-2 test split, cut into 64 chunks of 512 tokens by llama.cpp's tokenizer. The second half
  of each chunk is scored: 16,320 tokens per model.
- **Reference**: the Hugging Face model in FP32 (PyTorch) on the same tokens, written in llama.cpp's
  `--kl-divergence-base` format by `make_ref.py`.
- **GGUFs**: LiquidAI's official Q4_K_M and Q8_0 files, scored with `llama-perplexity --kl-divergence`
  (llama.cpp 4e7481175) on its Metal backend and on its CPU backend (`-ngl 0`). The CPU backend quantizes
  activations to 8 bits for its quantized dot products, as ONNX Runtime's CPU EP does for these recipes
  (MatMulNBits `accuracy_level` 4); Metal does not.
- **ONNX packages**: each recipe built verbatim (Olive 43eb0775, onnxruntime-genai 0.17.0's model builder) and
  scored by `eval_onnx.py` with llama-perplexity's statistics: CPU and WebGPU EPs on an Apple M3 Ultra (onnxruntime
  1.30.0, onnxruntime-ep-webgpu 0.4.0 on Metal), CUDA EP on an NVIDIA A10 (onnxruntime-gpu 1.30.0.dev20260812001,
  a CUDA 12 build).
- **Metrics**: `KLD` is the mean KL divergence of the next-token distribution from FP32 (lower is better), and
  `same top` is how often the most likely next token matches FP32's.

## Errors

The `±` in the tables is the standard error of the mean over the 64 chunks. llama-perplexity's own `±` treats the
16,320 tokens as independent, but tokens of one chunk are correlated, so it is 1.3-6.8x smaller (CUDA INT4 on
1.2B-Instruct: ±0.0020 against ±0.013 over chunks).

Differences between two variants are measured on the same tokens: `compare.py` takes the error of the difference
over the 64 chunk means. For example, CPU INT4 on 350M scores 0.492 ± 0.012 and Q4_K_M on llama.cpp's CPU backend
0.481 ± 0.010; on the same tokens, the difference is +0.010 ± 0.007 (2% ± 1%).

## Size

The INT4 recipes are 1.4-3.3x the size of Q4_K_M, for two reasons:

- The embedding table is stored unquantized (FP32 on CPU, FP16 on CUDA and WebGPU) next to the INT8 LM head,
  where Q4_K_M keeps one 6-bit table for both. Olive's ModelBuilder pass writes a dense Gather instead of using
  onnxruntime-genai's shared embedding; [microsoft/Olive#2692](https://github.com/microsoft/Olive/pull/2692) (open)
  shares it.
- MatMulNBits stores a scale (FP32 on CPU, FP16 on CUDA and WebGPU) and a 4-bit zero point for every block of 32
  weights, where Q4_K packs 6-bit scales and minimums into 256-weight super-blocks.

## INT4 against Q4_K_M: k_quant's zero point

ONNX Runtime's `k_quant` (onnxruntime-genai's `algo_config` `k_quant`) searches each block's scale and minimum the
way llama.cpp's Q4_K does. It then rounds the minimum to the integer zero point that MatMulNBits stores and
requantizes with the scale it found for the unrounded minimum, so the stored grid is not the one it fitted.
[microsoft/onnxruntime#32814](https://github.com/microsoft/onnxruntime/pull/32814) (open) fits the scale to the
stored zero point. The CPU INT4 recipes built with its `k_quant` (head 6c7ea3aa, swapped into onnxruntime 1.30.0;
same package format and size), on the CPU EP; percentages are from the values shown, their ± from the paired
comparison on the same tokens:

| model | INT4 as shipped | INT4 with #32814 | change | Q4_K_M, CPU | Q4_K_M, Metal |
| --- | --- | --- | --- | --- | --- |
| 230M | 0.114 | 0.0950 | -17% ± 1% | 0.0988 (-4% ± 2%) | 0.0922 (+3% ± 1%) |
| 350M | 0.492 | 0.468 | -5% ± 1% | 0.481 (-3% ± 2%) | 0.465 (+1% ± 2%) |
| 1.2B-Instruct | 0.155 | 0.103 | -34% ± 7% | 0.109 (-6% ± 4%) | 0.104 (-1% ± 4%) |
| 2.6B | 0.165 | 0.146 | -12% ± 2% | 0.155 (-6% ± 2%) | 0.155 (-6% ± 2%) |

With it, INT4 scores 3-6% below Q4_K_M on llama.cpp's CPU backend on every model, and between 6% below and 3%
above its Metal backend. The recipes' INT8 LM head and INT8 sensitive layers (the layers Q4_K_M promotes to 6
bits) are unchanged; the same quantizer builds the CUDA and WebGPU INT4 recipes.

## fp16_int4 against int4 on WebGPU

`_webgpu_fp16_int4.json` quantizes every MatMul except the LM head with symmetric RTN: a scale of absmax/7.5 per
block of 32 and no zero point. `_webgpu_int4.json` uses `k_quant` with the sensitive layers and the LM head at INT8.
Each variant below changes one ModelBuilder option of a recipe and is built and scored on WebGPU like the recipes
(in brackets: the difference from the column to its left, or for the last column from `int4`, with the paired
error):

| model | `fp16_int4`: symmetric RTN, FP16 head | asymmetric RTN | + INT8 sensitive layers | `int4`: k_quant, INT8 layers and head | `int4` with an FP16 head |
| --- | --- | --- | --- | --- | --- |
| 230M | 0.179 | 0.162 (-9% ± 1%) | 0.122 (-25% ± 1%) | 0.111 | 0.111 (+0.0000) |
| 350M | 0.979 | 0.664 (-32% ± 1%) | 0.511 (-23% ± 1%) | 0.491 | 0.491 (+0.0001) |
| 1.2B-Instruct | 0.210 | 0.132 (-37% ± 5%) | 0.119 (-10% ± 3%) | 0.158 | 0.158 (-0.0000) |
| 2.6B | 0.283 | 0.216 (-24% ± 2%) | 0.166 (-23% ± 1%) | 0.170 | 0.170 (+0.0000) |

- The LM head: leaving it in FP16 (`nodes_to_exclude: /lm_head/MatMul` instead of `last_matmul:int8`) changes KLD
  by at most 0.0001 on every model and adds 0.05-0.22 GiB.
- The rounding: `is_symmetric: false` (a zero point per block) cuts fp16_int4's KLD by 9-37%.
- The 4-bit sensitive layers: `matmul_mixed_precision: mixed_layers:int8` on top cuts it by another 10-25%.

On 1.2B-Instruct, asymmetric RTN alone beats the int4 recipe, whose `k_quant` loses the most on that model
(above).

## INT8 on CUDA: FP16 execution

The CPU and CUDA INT8 recipes quantize the weights the same way (Olive's RTN, symmetric 8-bit, groups of 32), but
the CUDA package runs the whole graph in FP16, KV cache and logits included, where the CPU package runs in FP32. An
unquantized FP16 build of each model (onnxruntime-genai's builder, `-p fp16 -e cuda`) measures what FP16 execution
costs on its own:

| model | INT8, CPU EP | INT8, CUDA EP | unquantized FP16, CUDA EP | Q8_0, CPU | Q8_0, Metal |
| --- | --- | --- | --- | --- | --- |
| 230M | 0.00173 | 0.00170 | 0.000764 | 0.00167 | 0.000744 |
| 350M | 0.00896 | 0.0171 | 0.0123 | 0.00843 | 0.00370 |
| 1.2B-Instruct | 0.00206 | 0.00280 | 0.00198 | 0.00196 | 0.000750 |
| 2.6B | 0.00328 | 0.00507 | 0.00369 | 0.00289 | 0.00123 |

On 350M, 1.2B-Instruct and 2.6B, FP16 execution alone costs as much KLD as the INT8 weights do on the CPU EP, or
more; on 230M it costs little, and INT8 on CUDA matches the CPU EP.

## Reproduce

For one model (1.2B-Instruct here), from this folder:

```
pip install numpy torch "transformers>=5.0.0" onnxruntime   # onnxruntime-gpu or onnxruntime-ep-webgpu for those EPs
git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout 4e7481175
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build llama.cpp/build -j --target llama-perplexity
curl -LO https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && unzip wikitext-2-raw-v1.zip
hf download LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q4_K_M.gguf --local-dir .
LP=llama.cpp/build/bin/llama-perplexity

# The GGUF's log-probabilities on llama.cpp's tokens, then the FP32 reference on the same tokens
$LP -m LFM2.5-1.2B-Instruct-Q4_K_M.gguf -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 64 --kl-divergence-base q4_k_m.kld
python make_ref.py LiquidAI/LFM2.5-1.2B-Instruct q4_k_m.kld ref.kld

# llama.cpp's own numbers (-ngl 0 for its CPU backend), and the GGUF's per-token values
$LP -m LFM2.5-1.2B-Instruct-Q4_K_M.gguf --kl-divergence-base ref.kld --kl-divergence -c 512
python score_gguf.py ref.kld q4_k_m.kld --label q4_k_m --dump scores

# A recipe: build it, score it on its EP (cpu, cuda or webgpu), compare it with the GGUF on the same tokens
(cd ../cpu && olive run --config LiquidAI-LFM2.5-1.2B-Instruct_cpu_int4.json)
python eval_onnx.py ref.kld cpu:../cpu/model --dump scores
python compare.py scores/LiquidAI-LFM2.5-1.2B-Instruct.cpu.model.cpu.npz scores/q4_k_m.npz
```

`make_ref.py` reads only the token stream from `q4_k_m.kld`; any GGUF of the model gives the same tokens.
`eval_onnx.py --provider-option` passes options to the CUDA or WebGPU EP.
