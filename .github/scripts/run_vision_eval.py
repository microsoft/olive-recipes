"""Run vision evaluation on a pre-built VLM ONNX model, with optional PyTorch comparison.

This script is designed to be called from olive_ci.json commands.
It builds the model via olive run, then evaluates it on vision benchmarks.
When --pytorch-model is provided, it also runs the same benchmarks on the
original PyTorch/HuggingFace model and reports the accuracy delta.

Usage:
    # Build + eval (standard CI flow)
    python run_vision_eval.py --config cpu/fp32/config.json --benchmarks textvqa --limit 50

    # Eval only (pre-built model)
    python run_vision_eval.py --model-path /path/to/model --benchmarks textvqa,chartqa --limit 100

    # With PyTorch comparison
    python run_vision_eval.py --model-path /path/to/model --pytorch-model google/gemma-4-E2B-it --benchmarks textvqa --limit 50

    # Full suite with comparison
    python run_vision_eval.py --config cpu/fp32/config.json --pytorch-model Qwen/Qwen3-VL-2B-Instruct --benchmarks textvqa,chartqa,docvqa --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Benchmark definitions — public HuggingFace datasets
BENCHMARKS = {
    "textvqa": {
        "display_name": "TextVQA (exact_match)",
        "dataset_name": "facebook/textvqa",
        "split": "validation",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answers",
        "sub_type": "exact_match",
    },
    "chartqa": {
        "display_name": "ChartQA (relaxed_accuracy)",
        "dataset_name": "HuggingFaceM4/ChartQA",
        "split": "test",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answer",
        "sub_type": "relaxed_accuracy",
    },
    "docvqa": {
        "display_name": "DocumentVQA (word_sort_ratio)",
        "dataset_name": "HuggingFaceM4/DocumentVQA",
        "split": "validation",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answers",
        "sub_type": "word_sort_ratio",
    },
}


def build_model(config_path: str) -> str:
    """Build ONNX model via olive run in a subprocess and return model directory path.

    Runs olive in a separate process to release memory after build completes,
    avoiding OOM kills on CI runners.
    """
    import subprocess

    print(f"Building model from {config_path}...")
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "olive", "run", "--config", config_path],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: Model build failed (exit code {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Find the output model directory by looking for genai_config.json
    # Parse the output_dir from the config
    config_data = json.loads(Path(config_path).read_text())
    output_dir = Path(config_data.get("output_dir", "models/output"))

    # Search for genai_config.json in the output
    for p in output_dir.rglob("genai_config.json"):
        model_dir = str(p.parent)
        print(f"Model built in {elapsed:.1f}s: {model_dir}")
        return model_dir

    print(f"ERROR: No genai_config.json found in {output_dir}", file=sys.stderr)
    print(f"Olive stdout: {result.stdout}", file=sys.stderr)
    sys.exit(1)


def run_vision_eval(model_path: str, benchmark: dict, device: str, limit: int | None) -> float:
    """Run a single vision benchmark and return accuracy score."""
    from olive.data.container.huggingface_container import HuggingfaceContainer
    from olive.evaluator.metric import Metric, MetricType, SubMetric
    from olive.evaluator.olive_evaluator import OnnxEvaluator
    from olive.hardware import Device
    from olive.model import ONNXModelHandler

    model_dir = Path(model_path)
    text_onnx = model_dir / "text.onnx"
    if not text_onnx.exists():
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx files in {model_dir}")
        text_onnx = onnx_files[0]

    model = ONNXModelHandler(model_path=str(text_onnx))

    pre_process_params = {
        "type": "vision_vqa_pre_process",
        "image_col": benchmark["image_col"],
        "question_col": benchmark["question_col"],
        "answer_col": benchmark["answer_col"],
    }
    if limit:
        pre_process_params["limit"] = limit

    container = HuggingfaceContainer(
        load_dataset_config={"data_name": benchmark["dataset_name"], "split": benchmark["split"]},
        pre_process_data_config=pre_process_params,
        dataloader_config={"batch_size": 1},
    )
    dataset = container.create_dataset()
    dataloader = container.create_dataloader(dataset)

    metric = Metric(
        name="vision_accuracy",
        type=MetricType.ACCURACY,
        sub_types=[SubMetric(name=benchmark["sub_type"], priority=1)],
        data_config=container,
    )

    eval_device = Device.GPU if device == "gpu" else Device.CPU
    evaluator = OnnxEvaluator()
    result = evaluator._evaluate_onnx_accuracy(model, metric, dataloader, device=eval_device)

    for sub_result in result.value:
        if sub_result.name == benchmark["sub_type"]:
            return sub_result.value

    raise ValueError(f"No result for {benchmark['sub_type']}")


def run_perf_benchmark(model_path: str, device: str, num_samples: int = 10) -> dict:
    """Run performance benchmarks on the ONNX model. Returns latency and throughput metrics."""
    import tempfile

    import onnxruntime_genai as og
    from PIL import Image

    # Build og.Model
    config = og.Config(model_path)
    config.clear_providers()
    if device == "gpu":
        config.append_provider("cuda")
    model = og.Model(config)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)

    # Create a simple test image (solid color) for consistent perf measurement
    test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    prompt_text = "Describe this image briefly."

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    messages_json = json.dumps(messages)

    latencies = []
    token_counts = []
    peak_memory_mb = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_img = Path(tmp_dir) / "test.png"
        test_image.save(str(tmp_img), format="PNG")

        # Warmup run
        images = og.Images.open(str(tmp_img))
        prompt = tokenizer.apply_chat_template(messages_json, add_generation_prompt=True)
        inputs = processor(prompt, images=images)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=64, do_sample=False)
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()
        del generator

        # Timed runs
        for _ in range(num_samples):
            images = og.Images.open(str(tmp_img))
            prompt = tokenizer.apply_chat_template(messages_json, add_generation_prompt=True)
            inputs = processor(prompt, images=images)

            params = og.GeneratorParams(model)
            params.set_search_options(max_length=64, do_sample=False)

            t0 = time.perf_counter()
            generator = og.Generator(model, params)
            generator.set_inputs(inputs)

            num_tokens = 0
            while not generator.is_done():
                generator.generate_next_token()
                num_tokens += 1
            del generator
            elapsed = time.perf_counter() - t0

            latencies.append(elapsed)
            token_counts.append(num_tokens)

    # GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        # torch is optional; keep default when unavailable.
        peak_memory_mb = 0

    # Process memory (RSS) as fallback
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB
    except ImportError:
        rss_mb = 0

    del model

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0
    p50 = sorted(latencies)[len(latencies) // 2]
    p90 = sorted(latencies)[int(len(latencies) * 0.9)]

    return {
        "avg_latency_s": avg_latency,
        "p50_latency_s": p50,
        "p90_latency_s": p90,
        "avg_tokens": avg_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_gpu_memory_mb": peak_memory_mb,
        "peak_rss_mb": rss_mb,
        "num_runs": num_samples,
    }


def run_pytorch_vision_eval(model_id: str, benchmark: dict, device: str, limit: int | None) -> float:
    """Run a vision benchmark on a PyTorch/HuggingFace model. Returns accuracy score."""
    import tempfile

    import torch
    from datasets import load_dataset
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"  Loading PyTorch model: {model_id}")
    torch_device = "cuda" if device == "gpu" and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch_device == "cuda" else torch.float32

    pt_model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(torch_device).eval()
    pt_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Load dataset
    ds = load_dataset(benchmark["dataset_name"], split=benchmark["split"])
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    image_col = benchmark["image_col"]
    question_col = benchmark["question_col"]
    answer_col = benchmark["answer_col"]

    # Import metric computation
    from olive.evaluator.accuracy import AccuracyBase

    correct = 0
    total = 0

    for sample in ds:
        pil_image = sample.get(image_col)
        question = sample.get(question_col, "")
        answer = sample.get(answer_col, "")

        if pil_image is None:
            continue

        if isinstance(pil_image, dict) and "bytes" in pil_image:
            import io
            pil_image = Image.open(io.BytesIO(pil_image["bytes"])).convert("RGB")
        elif not isinstance(pil_image, Image.Image):
            pil_image = Image.open(pil_image).convert("RGB")

        # Build chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = pt_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process inputs — handle models that need process_vision_info (Qwen-VL family)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = pt_processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(torch_device)
        except ImportError:
            inputs = pt_processor(
                text=[text], images=[pil_image], return_tensors="pt"
            ).to(torch_device)

        with torch.no_grad():
            out = pt_model.generate(**inputs, max_new_tokens=128, do_sample=False)

        out_ids = out[0][inputs["input_ids"].shape[-1]:]
        pred = pt_processor.decode(out_ids, skip_special_tokens=True).strip()

        # Normalize answer for comparison
        if isinstance(answer, list):
            # Multi-answer: check if prediction matches any
            answer_strs = [str(a).strip().lower() for a in answer]
            hit = pred.strip().lower() in answer_strs
        else:
            hit = pred.strip().lower() == str(answer).strip().lower()

        if hit:
            correct += 1
        total += 1

    del pt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Vision evaluation for VLM ONNX models")
    parser.add_argument("--config", default=None, help="Olive config to build model (skipped if --model-path set)")
    parser.add_argument("--model-path", default=None, help="Pre-built model directory")
    parser.add_argument("--pytorch-model", default=None, help="HuggingFace model ID for PyTorch comparison")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--benchmarks", default="textvqa", help="Comma-separated: textvqa,chartqa,docvqa")
    parser.add_argument("--limit", type=int, default=50, help="Samples per benchmark (0=full)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min accuracy (fails if below)")
    parser.add_argument("--max-delta", type=float, default=0.0,
                        help="Max allowed accuracy drop from PyTorch to ONNX (e.g. 0.02 = 2pp). 0 = no check.")
    parser.add_argument("--perf", action="store_true",
                        help="Run performance benchmarks (latency, throughput, memory)")
    parser.add_argument("--perf-samples", type=int, default=10,
                        help="Number of inference runs for perf measurement (default: 10)")
    args = parser.parse_args()

    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    elif args.config:
        model_path = build_model(args.config)
    else:
        print("ERROR: Provide --config or --model-path", file=sys.stderr)
        sys.exit(1)

    # Verify it's a vision model
    genai_config = Path(model_path) / "genai_config.json"
    if not genai_config.exists():
        print(f"ERROR: genai_config.json not found in {model_path}", file=sys.stderr)
        sys.exit(1)

    cfg = json.loads(genai_config.read_text())
    if "vision" not in cfg.get("model", {}):
        print(f"WARNING: Model may not be a VLM (no 'vision' field in genai_config.json)")

    # Run benchmarks
    benchmark_names = [b.strip() for b in args.benchmarks.split(",")]
    limit = args.limit if args.limit > 0 else None
    onnx_results = {}
    pytorch_results = {}
    failed = False

    for name in benchmark_names:
        if name not in BENCHMARKS:
            print(f"ERROR: Unknown benchmark '{name}'. Available: {', '.join(BENCHMARKS)}", file=sys.stderr)
            sys.exit(1)

        benchmark = BENCHMARKS[name]
        print(f"\n{'='*60}")
        print(f"Running {benchmark['display_name']}")
        print(f"  Dataset: {benchmark['dataset_name']} (split={benchmark['split']})")
        print(f"  Limit: {limit or 'full'}")
        print(f"{'='*60}")

        # ONNX evaluation
        print(f"\n  [ONNX] Evaluating...")
        start = time.time()
        onnx_acc = run_vision_eval(model_path, benchmark, args.device, limit)
        onnx_elapsed = time.time() - start
        onnx_results[name] = {"acc": onnx_acc, "time": onnx_elapsed}

        status = "PASS" if onnx_acc >= args.threshold else "FAIL"
        if onnx_acc < args.threshold:
            failed = True
        print(f"  [ONNX] {status}: {onnx_acc:.4f} ({onnx_elapsed:.1f}s)")

        # PyTorch evaluation (optional)
        if args.pytorch_model:
            print(f"\n  [PyTorch] Evaluating {args.pytorch_model}...")
            start = time.time()
            pt_acc = run_pytorch_vision_eval(args.pytorch_model, benchmark, args.device, limit)
            pt_elapsed = time.time() - start
            pytorch_results[name] = {"acc": pt_acc, "time": pt_elapsed}
            print(f"  [PyTorch] {pt_acc:.4f} ({pt_elapsed:.1f}s)")

            # Check delta
            delta = pt_acc - onnx_acc
            print(f"  [Delta] PyTorch - ONNX = {delta:+.4f} ({delta*100:+.2f}pp)")
            if args.max_delta > 0 and delta > args.max_delta:
                print(f"  FAIL: ONNX accuracy dropped {delta:.4f} from PyTorch, exceeds max-delta {args.max_delta}")
                failed = True

    # Performance benchmarks (optional)
    perf_results = None
    if args.perf:
        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARKS")
        print(f"{'='*60}")
        print(f"  Running {args.perf_samples} inference iterations...")
        perf_results = run_perf_benchmark(model_path, args.device, args.perf_samples)
        print(f"  Avg latency:      {perf_results['avg_latency_s']:.3f}s")
        print(f"  P50 latency:      {perf_results['p50_latency_s']:.3f}s")
        print(f"  P90 latency:      {perf_results['p90_latency_s']:.3f}s")
        print(f"  Avg tokens/run:   {perf_results['avg_tokens']:.1f}")
        print(f"  Tokens/sec:       {perf_results['tokens_per_sec']:.1f}")
        if perf_results['peak_gpu_memory_mb'] > 0:
            print(f"  Peak GPU memory:  {perf_results['peak_gpu_memory_mb']:.0f} MB")
        if perf_results['peak_rss_mb'] > 0:
            print(f"  Peak RSS memory:  {perf_results['peak_rss_mb']:.0f} MB")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Benchmark':<35} {'ONNX':>8} {'PyTorch':>8} {'Delta':>8} {'ONNX Time':>10} {'PT Time':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for name in benchmark_names:
        onnx_acc = onnx_results[name]["acc"]
        onnx_time = onnx_results[name]["time"]
        if name in pytorch_results:
            pt_acc = pytorch_results[name]["acc"]
            pt_time = pytorch_results[name]["time"]
            delta = pt_acc - onnx_acc
            print(f"  {BENCHMARKS[name]['display_name']:<35} {onnx_acc:>8.4f} {pt_acc:>8.4f} {delta:>+8.4f} {onnx_time:>9.1f}s {pt_time:>9.1f}s")
        else:
            print(f"  {BENCHMARKS[name]['display_name']:<35} {onnx_acc:>8.4f} {'N/A':>8} {'N/A':>8} {onnx_time:>9.1f}s {'N/A':>10}")

    if pytorch_results:
        print()
        speedups = []
        for name in benchmark_names:
            if name in pytorch_results and pytorch_results[name]["time"] > 0:
                speedup = pytorch_results[name]["time"] / max(onnx_results[name]["time"], 1e-9)
                speedups.append(speedup)
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"  Avg ONNX speedup vs PyTorch: {avg_speedup:.2f}x")

    if failed:
        print(f"\nFAILED: One or more checks failed.")
        sys.exit(1)
    else:
        print("\nAll benchmarks passed.")


if __name__ == "__main__":
    main()
