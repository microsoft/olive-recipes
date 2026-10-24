"""Standalone direct-ORT, batch-one Qwen3 benchmark (no CUDA graphs)."""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time


PREFIX = "<|im_start|>user\n"
SUFFIX = (
    "Summarize the context in detail. /no_think<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
CONTEXT = (
    "Paris is the capital of France. Its museums and public parks attract visitors. "
)


def build_prompt(tokenizer, input_length):
    prefix, suffix, context = (
        tokenizer.encode(text, add_special_tokens=False).ids
        for text in (PREFIX, SUFFIX, CONTEXT)
    )
    remaining = input_length - len(prefix) - len(suffix)
    if remaining < 0 or not context:
        raise ValueError(
            "Input length cannot hold the prompt template, or context is empty"
        )
    return prefix + (context * (remaining // len(context) + 1))[:remaining] + suffix


def validate_tokens(tokens, output_length, vocab_size=None):
    if not isinstance(tokens, list) or len(tokens) != output_length:
        raise ValueError(
            "Token stream must be a JSON list with exactly output_length tokens"
        )
    if any(
        type(token) is not int
        or token < 0
        or (vocab_size is not None and token >= vocab_size)
        for token in tokens
    ):
        raise ValueError("Token IDs must be nonnegative integers within the vocabulary")
    return tokens


def decode_input(step, prediction, token_stream):
    if step < 1:
        raise ValueError("Decode steps start at one")
    return prediction if token_stream is None else token_stream[step - 1]


def token_counts(input_length, output_length):
    if input_length < 1 or output_length < 2:
        raise ValueError("Require input_length >= 1 and output_length >= 2")
    return {
        "input_tokens": input_length,
        "output_tokens": output_length,
        "decode_steps": output_length - 1,
        "final_cache_tokens": input_length + output_length - 1,
    }


def token_hash(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def percentile(values, percent):
    ordered = sorted(values)
    if (
        not ordered
        or not 0 <= percent <= 100
        or not all(math.isfinite(value) for value in ordered)
    ):
        raise ValueError(
            "Percentiles require finite, nonempty data and percent in [0, 100]"
        )
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows, input_length, output_length):
    counts = token_counts(input_length, output_length)
    if not rows or any(
        row[metric] <= 0 or not math.isfinite(row[metric])
        for row in rows
        for metric in ("ttft_ms", "prefill_seconds", "decode_seconds")
    ):
        raise ValueError("Trials must have finite positive durations")
    result = dict(counts)
    for metric in ("ttft_ms", "prefill_seconds", "decode_seconds"):
        for percent in (50, 95):
            result[f"{metric}_p{percent}"] = percentile(
                [row[metric] for row in rows], percent
            )
    result["prefill_tps"] = (
        input_length * len(rows) / sum(row["prefill_seconds"] for row in rows)
    )
    result["decode_tps"] = (
        counts["decode_steps"] * len(rows) / sum(row["decode_seconds"] for row in rows)
    )
    runs = [seconds * 1000 for row in rows for seconds in row["decode_run_seconds"]]
    for percent in (50, 95):
        result[f"decode_ort_run_p{percent}_ms"] = percentile(runs, percent)
    return result


def clean_environment(physical_gpu):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError(
            "Set CUDA_VISIBLE_DEVICES to exactly --physical-gpu before launching"
        )
    removed = {}
    for name in list(os.environ):
        if "QMOE" in name.upper() or name == "ORT_DISABLE_MOE_GEMV":
            removed[name] = os.environ.pop(name)
    return removed


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


class MemorySampler:
    """NVML residency sampled every requested 5ms, not an allocator peak."""

    def __init__(self, nvml, index):
        self.nvml = nvml
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(index)
        if nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
            raise RuntimeError("Benchmark GPU must initially have no compute processes")
        self.process_id = None
        self.samples = []
        self.error = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.sample()

    def sample(self):
        with self.lock:
            processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            if len(processes) > 1:
                raise RuntimeError(
                    f"Other compute GPU PIDs detected: {[process.pid for process in processes]}"
                )
            if processes:
                process_id = processes[0].pid
                if self.process_id is None:
                    self.process_id = process_id
                elif process_id != self.process_id:
                    raise RuntimeError(
                        f"GPU compute PID changed from {self.process_id} to {process_id}"
                    )
            process_bytes = sum(process.usedGpuMemory for process in processes)
            if not isinstance(process_bytes, int) or not 0 <= process_bytes < 2**60:
                raise RuntimeError("NVML process residency unavailable")
            device_bytes = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
            self.samples.append((time.perf_counter(), process_bytes, device_bytes))

    def collect(self):
        try:
            while not self.stop.wait(0.005):
                self.sample()
        except Exception as error:
            self.error = error

    def check(self):
        if self.error is not None:
            raise self.error

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.check()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-commit", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument(
        "--token-stream", type=Path, help="JSON token list of length output-length"
    )
    parser.add_argument(
        "--write-token-stream",
        type=Path,
        help="Write first measured greedy predictions",
    )
    parser.add_argument(
        "--logits-output",
        type=Path,
        help="First measured trial last-position logits, NPZ",
    )
    parser.add_argument("--int-dequant-max-scratch-bytes", type=int, default=67108864)
    parser.add_argument(
        "--cudart", type=Path, default=Path("/usr/local/cuda/lib64/libcudart.so")
    )
    args = parser.parse_args(argv)
    try:
        token_counts(args.input_length, args.output_length)
        if args.warmups < 0 or args.repeats < 1 or args.physical_gpu < 0:
            raise ValueError("Require warmups >= 0, repeats >= 1 and physical-gpu >= 0")
        if args.int_dequant_max_scratch_bytes < 1:
            raise ValueError("Scratch limit must be positive")
        if not args.artifact_commit.strip() or not args.runtime_commit.strip():
            raise ValueError("Commit labels must not be empty")
        if args.token_stream and args.write_token_stream:
            raise ValueError("--write-token-stream requires natural greedy mode")
        destinations = [
            path.resolve()
            for path in (args.output, args.write_token_stream, args.logits_output)
            if path is not None
        ]
        if len(set(destinations)) != len(destinations) or any(
            path.exists() for path in destinations
        ):
            raise ValueError("Output paths must be distinct and must not already exist")
    except ValueError as error:
        parser.error(str(error))
    return args


def run(args):
    removed_env = clean_environment(args.physical_gpu)
    import numpy as np
    import onnxruntime as ort
    import pynvml
    from tokenizers import Tokenizer

    cudart = ctypes.CDLL(str(args.cudart))
    cudart.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaRuntimeGetVersion.restype = ctypes.c_int
    version = ctypes.c_int()
    if (
        cudart.cudaRuntimeGetVersion(ctypes.byref(version))
        or version.value // 1000 != 13
    ):
        raise RuntimeError("--cudart must load a working CUDA 13 runtime")
    cudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    cudart.cudaMemcpy.restype = ctypes.c_int
    cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("This ORT build has no CUDAExecutionProvider")
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    prompt_ids = build_prompt(tokenizer, args.input_length)
    prompt = np.asarray([prompt_ids], dtype=np.int64)
    token_stream = None
    if args.token_stream:
        token_stream = validate_tokens(
            json.loads(args.token_stream.read_text()), args.output_length
        )
    artifact_files = [
        file_hash(path) for path in sorted(args.model_dir.iterdir()) if path.is_file()
    ]
    sampler = None
    try:
        sampler = MemorySampler(pynvml, args.physical_gpu)
        sampler.thread.start()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        options.enable_profiling = False
        options.add_session_config_entry(
            "ep.cuda.qmoe_int_dequant_max_scratch_bytes",
            str(args.int_dequant_max_scratch_bytes),
        )
        started = time.perf_counter()
        session = ort.InferenceSession(
            str(args.model_dir / "model.onnx"),
            options,
            providers=[
                ("CUDAExecutionProvider", {"enable_cuda_graph": "0"}),
                "CPUExecutionProvider",
            ],
        )
        session.disable_fallback()
        init_seconds = time.perf_counter() - started
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("CUDA provider failed to load; refusing CPU fallback")
        loaded_paths = {
            Path(line.split()[-1])
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "onnxruntime" in line
            and ".so" in line
            and line.split()[-1].startswith("/")
        }
        if not any(
            "libonnxruntime_providers_cuda.so" in path.name for path in loaded_paths
        ):
            raise RuntimeError("CUDA provider binary is not mapped in this process")
        binaries = [file_hash(path) for path in sorted(loaded_paths)]
        sampler.sample()
        loaded_bytes = sampler.samples[-1][1]
        output_info = session.get_outputs()
        output_names = [value.name for value in output_info]
        logits_info = next(value for value in output_info if value.name == "logits")
        cache_inputs = [
            value
            for value in session.get_inputs()
            if value.name.startswith("past_key_values.")
        ]
        if logits_info.type != "tensor(float16)" or not cache_inputs:
            raise ValueError("Expected FP16 logits and past_key_values inputs")
        if any(value.type != "tensor(float16)" for value in cache_inputs):
            raise ValueError("Expected FP16 KV cache inputs")
        if set(value.name for value in session.get_inputs()) != {
            "input_ids",
            "attention_mask",
            *(value.name for value in cache_inputs),
        }:
            raise ValueError("Unsupported model input contract")
        host_logits = np.empty(logits_info.shape[-1], dtype=np.float16)
        if token_stream is not None:
            validate_tokens(token_stream, args.output_length, host_logits.size)

        def run_step(input_ids, mask, caches):
            binding = session.io_binding()
            binding.bind_cpu_input("input_ids", input_ids)
            binding.bind_cpu_input("attention_mask", mask)
            for name, value in caches.items():
                binding.bind_ortvalue_input(name, value)
            for name in output_names:
                binding.bind_output(name, "cuda", 0)
            started = time.perf_counter()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            run_seconds = time.perf_counter() - started
            results = dict(zip(output_names, binding.get_outputs(), strict=True))
            logits = results.pop("logits")
            if logits.shape() != [1, input_ids.shape[1], host_logits.size]:
                raise ValueError(f"Unexpected logits shape: {logits.shape()}")
            address = logits.data_ptr() + (input_ids.shape[1] - 1) * host_logits.nbytes
            status = cudart.cudaMemcpy(
                host_logits.ctypes.data, address, host_logits.nbytes, 2
            )
            if status:
                raise RuntimeError(cudart.cudaGetErrorString(status).decode())
            if not np.isfinite(host_logits).all():
                raise RuntimeError("Non-finite last-position logits")
            prediction = int(np.argmax(host_logits))
            next_caches = {
                name.replace("present.", "past_key_values.", 1): value
                for name, value in results.items()
            }
            if next_caches.keys() != caches.keys():
                raise ValueError("Present/past KV cache names do not match")
            return prediction, next_caches, run_seconds

        def trial(capture=False):
            caches = {
                value.name: ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((1, value.shape[1], 0, value.shape[3]), dtype=np.float16),
                    "cuda",
                    0,
                )
                for value in cache_inputs
            }
            mask = np.ones((1, args.input_length), dtype=np.int64)
            vectors, hashes = [], []
            started = time.perf_counter()
            prediction, caches, prefill_seconds = run_step(prompt, mask, caches)
            first_token = time.perf_counter()
            predictions = [prediction]
            hashes.append(hashlib.sha256(host_logits.tobytes()).hexdigest())
            if capture:
                vectors.append(host_logits.copy())
            decode_runs = []
            decode_started = time.perf_counter()
            for step in range(1, args.output_length):
                ids = np.asarray(
                    [[decode_input(step, prediction, token_stream)]], dtype=np.int64
                )
                mask = np.ones((1, args.input_length + step), dtype=np.int64)
                prediction, caches, run_seconds = run_step(ids, mask, caches)
                predictions.append(prediction)
                decode_runs.append(run_seconds)
                hashes.append(hashlib.sha256(host_logits.tobytes()).hexdigest())
                if capture:
                    vectors.append(host_logits.copy())
            decode_seconds = time.perf_counter() - decode_started
            sampler.check()
            return {
                "ttft_ms": (first_token - started) * 1000,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "decode_run_seconds": decode_runs,
                "generated_tokens": predictions,
                "logits_sha256": hashes,
            }, vectors

        for index in range(args.warmups):
            trial()
            print(f"WARMUP {index + 1}", flush=True)
        sampler.sample()
        measurement_start = time.perf_counter()
        rows, first_logits = [], []
        for index in range(args.repeats):
            row, vectors = trial(capture=index == 0 and args.logits_output is not None)
            rows.append(row)
            if index == 0:
                first_logits = vectors
            print(f"TRIAL {index + 1} TTFT {row['ttft_ms']:.3f} ms", flush=True)
        sampler.sample()
        sampler.close()
        measured = [
            sample for sample in sampler.samples if sample[0] >= measurement_start
        ]
        summary = summarize(rows, args.input_length, args.output_length)
        summary.update(
            {
                "session_init_seconds": init_seconds,
                "session_loaded_process_gib": loaded_bytes / 2**30,
                "measurement_peak_process_gib": max(sample[1] for sample in measured)
                / 2**30,
                "measurement_peak_device_gib": max(sample[2] for sample in measured)
                / 2**30,
                "lifetime_peak_process_gib": max(
                    sample[1] for sample in sampler.samples
                )
                / 2**30,
                "max_measurement_sample_gap_ms": max(
                    (
                        (right[0] - left[0]) * 1000
                        for left, right in zip(measured, measured[1:])
                    ),
                    default=0,
                ),
                "generated_tokens_identical": all(
                    row["generated_tokens"] == rows[0]["generated_tokens"]
                    for row in rows
                ),
            }
        )
        report = {
            "summary": summary,
            "trials": rows,
            "metadata": {
                "artifact_commit": args.artifact_commit,
                "runtime_commit": args.runtime_commit,
                "artifact_files": artifact_files,
                "runtime_binaries": binaries,
                "ort_version": ort.__version__,
                "ort_build": ort.get_build_info(),
                "providers": session.get_providers(),
                "provider_options": session.get_provider_options(),
                "cuda_runtime_version": version.value,
                "cudart": str(args.cudart.resolve()),
                "gpu_name": pynvml.nvmlDeviceGetName(sampler.handle),
                "gpu_uuid": pynvml.nvmlDeviceGetUUID(sampler.handle),
                "nvml_process_pid": sampler.process_id,
                "arguments": {
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(args).items()
                },
                "removed_environment": removed_env,
                "prompt_ids": prompt_ids,
                "prompt_sha256": token_hash(prompt_ids),
                "forced_tokens": token_stream,
                "forced_token_sha256": token_hash(token_stream)
                if token_stream is not None
                else None,
                "generated_text": tokenizer.decode(rows[0]["generated_tokens"]),
            },
            "method": {
                "ttft": "Pretokenized prompt to CPU argmax; excludes session init and KV reset",
                "prefill": "Synchronized ORT run only, full-sequence logits on GPU",
                "decode": "output_length-1 steps, Python/binding/mask/GPU/D2H/finite/argmax/hash; "
                "first measured trial also copies logits if requested",
                "memory": "NVML process/device residency, requested 5ms sampling, not exact peak",
                "process_attribution": "Initially idle GPU; latch first sole NVML compute PID, "
                "reject additional/changed PIDs; unsampled races are not detectable",
                "logits": "SHA256 of each last-position FP16 vector bytes; NPZ first measured trial only",
                "eos": "Ignored; fixed output length",
                "batch_size": 1,
                "cuda_graph": False,
                "profiling": False,
                "debug": False,
                "cpu_ep": "Allowed for shape operations; CUDA provider must be loaded",
            },
            "memory_samples": sampler.samples,
        }
        if args.logits_output:
            args.logits_output.parent.mkdir(parents=True, exist_ok=True)
            with args.logits_output.open("xb") as stream:
                np.savez(
                    stream,
                    logits=np.stack(first_logits),
                    generated_tokens=np.asarray(
                        rows[0]["generated_tokens"], dtype=np.int64
                    ),
                )
        if args.write_token_stream:
            args.write_token_stream.parent.mkdir(parents=True, exist_ok=True)
            args.write_token_stream.write_text(
                json.dumps(rows[0]["generated_tokens"]) + "\n"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("SUMMARY", json.dumps(summary, allow_nan=False), flush=True)
    finally:
        try:
            if sampler is not None:
                sampler.close()
        finally:
            pynvml.nvmlShutdown()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
