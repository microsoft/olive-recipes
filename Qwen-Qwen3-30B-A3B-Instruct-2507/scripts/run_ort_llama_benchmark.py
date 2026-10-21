#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Measure isolated ORT GenAI and llama.cpp trials on physical GPU 0."""
from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import json
import math
import os
import select
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID


ALLOCATOR_KEY = "session.use_device_allocator_for_initializers"
SCHEMA_VERSION = 3
WORKER_TIMEOUT_SECONDS = 1800
PROCESS_STOP_TIMEOUT_SECONDS = 30
SCRIPT = Path(__file__).resolve()
GPU_UUID_ENV = "QWEN_BENCHMARK_GPU_UUID"


@dataclass(frozen=True)
class Trial:
    runtime: str
    ort_allocator_mode: str
    context_label: int
    repetition: int
    output_tokens: int
    sequence_capacity: int
    prompt: str
    ort_model: str
    gguf_model: str
    directory: str
    original_config_sha256: str
    sampling_interval_ms: float
    profile_dir: str | None = None
    chunk_size: int | None = None
    cold_warm: bool = False


def request_labels(trial: Trial) -> tuple[str, ...]:
    return ("cold", "warm_1", "warm_2") if trial.cold_warm else ("cold",)


def require_linux_gpu_execution() -> None:
    if sys.platform != "linux":
        raise RuntimeError("GPU benchmark execution requires Linux; CPU helpers and --help remain portable")


def normalize_gpu_uuid(value: str | bytes) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if not value.startswith("GPU-"):
        raise RuntimeError(f"Expected a full physical GPU UUID, got {value!r}")
    try:
        return "GPU-" + str(UUID(value[4:]))
    except ValueError as error:
        raise RuntimeError(f"Invalid physical GPU UUID: {value!r}") from error


def gpu_zero_environment() -> None:
    require_linux_gpu_execution()
    if os.environ.get("BENCHMARK_GPU_INDEX", "0") != "0":
        raise RuntimeError("BENCHMARK_GPU_INDEX must be 0; refusing another physical NVML GPU")
    with nvml_device() as (nvml, handle):
        target = normalize_gpu_uuid(nvml.nvmlDeviceGetUUID(handle))
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visibility not in (None, "0"):
        if normalize_gpu_uuid(visibility) != target:
            raise RuntimeError("CUDA_VISIBLE_DEVICES conflicts with the intended physical NVML GPU 0")
    os.environ["BENCHMARK_GPU_INDEX"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = target
    os.environ[GPU_UUID_ENV] = target


class CudaDeviceUUID(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 16)]


def cuda_driver_gpu_uuid() -> str:
    """Query logical device identity using the driver API, without creating a context."""
    require_linux_gpu_execution()
    library = ctypes.util.find_library("cuda")
    if library is None:
        raise RuntimeError("CUDA driver library not found; execution/monitoring identity cannot be verified")
    try:
        driver = ctypes.CDLL(library)
    except OSError as error:
        raise RuntimeError(f"Cannot load CUDA driver identity API from {library}") from error
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None)
    if get_uuid is None:
        get_uuid = getattr(driver, "cuDeviceGetUuid", None)
    if get_uuid is None:
        raise RuntimeError("CUDA driver does not expose a device UUID query")
    get_uuid.argtypes = [ctypes.POINTER(CudaDeviceUUID), ctypes.c_int]
    get_uuid.restype = ctypes.c_int

    def check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"CUDA driver identity preflight {operation} failed with status {status}")

    check(driver.cuInit(0), "cuInit")
    count = ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    if count.value != 1:
        raise RuntimeError(f"Expected one UUID-selected CUDA device, got {count.value}")
    device = ctypes.c_int()
    check(driver.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    identifier = CudaDeviceUUID()
    check(get_uuid(ctypes.byref(identifier), device.value), "cuDeviceGetUuid")
    return "GPU-" + str(UUID(bytes=bytes(identifier.data)))


def validate_execution_gpu_identity() -> str:
    expected = os.environ.get(GPU_UUID_ENV)
    if expected is None:
        raise RuntimeError("Resolve the intended NVML GPU before CUDA identity validation")
    expected = normalize_gpu_uuid(expected)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError("CUDA visibility must be bound to the resolved physical GPU UUID")
    actual = cuda_driver_gpu_uuid()
    if actual != expected:
        raise RuntimeError(f"CUDA execution GPU {actual} differs from NVML monitoring GPU {expected}")
    return actual


@contextmanager
def gpu_run_lock() -> Generator[None, None, None]:
    require_linux_gpu_execution()
    import fcntl

    path = Path(tempfile.gettempdir()) / f"qwen-benchmark-{os.getuid()}-gpu0.lock"
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another benchmark invocation holds the GPU 0 run lock") from error
        yield


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def has_allocator_override(value: Any) -> bool:
    if isinstance(value, dict):
        return ALLOCATOR_KEY in value or any(has_allocator_override(child) for child in value.values())
    if isinstance(value, list):
        return any(has_allocator_override(child) for child in value)
    return False


def config_digest(model_path: Path, expected: str | None = None) -> str:
    raw = (model_path / "genai_config.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    config = json.loads(raw)
    if has_allocator_override(config["model"]["decoder"]["session_options"]):
        raise RuntimeError("ORIGINAL CONFIG MUTATED: allocator override is present")
    if expected is not None and digest != expected:
        raise RuntimeError(f"ORIGINAL CONFIG MUTATED: expected {expected}, got {digest}")
    return digest


@contextmanager
def ort_model_variant(
    model_path: Path, use_device_allocator: bool, sequence_capacity: int
) -> Generator[Path, None, None]:
    original_hash = config_digest(model_path)
    config = json.loads((model_path / "genai_config.json").read_text(encoding="utf-8"))
    if sequence_capacity > config["model"]["context_length"]:
        raise ValueError("Requested capacity exceeds the ORT artifact's context limit")
    with tempfile.TemporaryDirectory(prefix="ort-trial-", dir=model_path.parent) as temporary:
        variant = Path(temporary) / model_path.name
        shutil.copytree(model_path, variant, copy_function=os.link)
        config_path = variant / "genai_config.json"
        config["model"]["context_length"] = sequence_capacity
        config["search"]["max_length"] = sequence_capacity
        if use_device_allocator:
            config["model"]["decoder"]["session_options"][ALLOCATOR_KEY] = "1"
        config_path.unlink()  # Break the hardlink before writing the per-trial config.
        write_json(config_path, config)
        try:
            yield variant
        finally:
            config_digest(model_path, original_hash)


def build_prompt(context_label: int) -> str:
    seed = "Benchmark context sentence for Phi-4 runtime comparison. "
    repetitions = max(1, context_label // 9)
    return (seed * repetitions)[: max(100, context_label * 5)] + "\nAnswer briefly:"


def requested_sequence_capacity(context_label: int, output_tokens: int) -> int:
    if context_label < 1 or output_tokens < 1:
        raise ValueError("Context label and output limit must be positive")
    # llama.cpp 0.3.35 pads n_ctx and n_ctx_seq to 256, even without flash attention.
    return ((context_label + output_tokens + 255) // 256) * 256


def as_mib(value: int) -> float:
    return round(value / (1024 * 1024), 2)


@contextmanager
def nvml_device() -> Generator[tuple[Any, Any], None, None]:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        expected = os.environ.get(GPU_UUID_ENV)
        if expected is not None and normalize_gpu_uuid(pynvml.nvmlDeviceGetUUID(handle)) != normalize_gpu_uuid(expected):
            raise RuntimeError("NVML GPU 0 no longer matches the launcher-selected physical GPU UUID")
        yield pynvml, handle
    finally:
        pynvml.nvmlShutdown()


def device_snapshot(nvml: Any, handle: Any) -> dict[str, Any]:
    started = time.perf_counter_ns()
    memory = nvml.nvmlDeviceGetMemoryInfo(handle, version=nvml.nvmlMemory_v2)
    processes = [
        *nvml.nvmlDeviceGetComputeRunningProcesses(handle),
        *nvml.nvmlDeviceGetGraphicsRunningProcesses(handle),
    ]
    return {
        "time_ns": started,
        "query_finished_ns": time.perf_counter_ns(),
        "vram_bytes": int(memory.used),
        "reserved_bytes": int(memory.reserved),
        "gpu_process_pids": sorted({int(process.pid) for process in processes}),
    }


def check_gpu_processes(snapshot: dict[str, Any], allowed_pids: set[int]) -> None:
    unrelated = set(snapshot["gpu_process_pids"]) - allowed_pids
    if unrelated:
        raise RuntimeError(f"GPU 0 has unrelated processes {sorted(unrelated)}; refusing to proceed")


def idle_gpu_snapshot() -> dict[str, Any]:
    with nvml_device() as (nvml, handle):
        snapshot = device_snapshot(nvml, handle)
        check_gpu_processes(snapshot, set())
        snapshot["gpu_uuid"] = nvml.nvmlDeviceGetUUID(handle)
        snapshot["gpu_name"] = nvml.nvmlDeviceGetName(handle)
        return snapshot


def sample_worker(directory: Path, worker_pid: int, interval_ms: float) -> int:
    import psutil

    gpu_zero_environment()
    process = psutil.Process(worker_pid)
    with nvml_device() as (nvml, handle):
        first = device_snapshot(nvml, handle)
        check_gpu_processes(first, set())
        fields = ["time_ns", "query_finished_ns", "vram_bytes", "reserved_bytes", "ram_bytes", "gpu_process_pids"]
        with (directory / "samples.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()

            def record(snapshot: dict[str, Any]) -> None:
                check_gpu_processes(snapshot, {worker_pid})
                writer.writerow({
                    **snapshot,
                    "ram_bytes": process.memory_info().rss,
                    "gpu_process_pids": json.dumps(snapshot["gpu_process_pids"]),
                })
                stream.flush()

            record(first)
            print(json.dumps({
                **first, "sampler_pid": os.getpid(),
                "gpu_uuid": nvml.nvmlDeviceGetUUID(handle),
            }), flush=True)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], interval_ms / 1000)
                if ready:
                    command = sys.stdin.readline()
                    if command not in ("", "stop\n"):
                        raise RuntimeError(f"Invalid sampler command: {command!r}")
                    if not process.is_running():
                        raise RuntimeError("Trial worker exited before sampler shutdown")
                    record(device_snapshot(nvml, handle))
                    break
                record(device_snapshot(nvml, handle))
    return 0


def stop_owned_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)


def stop_owned_worker(process: subprocess.Popen[str]) -> None:
    import psutil

    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    stop_owned_process(process)
    for child in children:
        try:
            child.wait(timeout=5)
        except psutil.TimeoutExpired:
            try:
                child.terminate()
                child.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
            except psutil.TimeoutExpired:
                child.kill()
                child.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
            except psutil.NoSuchProcess:
                continue


class ResourceSampler:
    def __init__(self, directory: Path, interval_ms: float) -> None:
        self.directory = directory
        self.interval_ms = interval_ms
        self.baseline: dict[str, Any] = {}
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> ResourceSampler:
        self.stderr = (self.directory / "sampler.stderr.log").open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--sample-worker", str(self.directory),
                 str(os.getpid()), str(self.interval_ms)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True,
            )
            self.baseline = self.read_baseline()
            if self.baseline["sampler_pid"] != self.process.pid:
                raise RuntimeError("Sampler handshake PID does not match the owned process")
            check_gpu_processes(self.baseline, set())
        except BaseException:
            try:
                if self.process is not None:
                    stop_owned_process(self.process)
            finally:
                self.close_streams()
            raise
        return self

    def read_baseline(self) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Sampler stdout pipe was not created")
        deadline = time.monotonic() + 30
        data = bytearray()
        while b"\n" not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Sampler did not provide a complete baseline within 30 seconds")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError("Sampler did not provide a complete baseline within 30 seconds")
            chunk = os.read(self.process.stdout.fileno(), 4096)
            if not chunk:
                error = (self.directory / "sampler.stderr.log").read_text(encoding="utf-8")
                raise RuntimeError(f"Sampler exited without a baseline:\n{error}")
            data.extend(chunk)
            if len(data) > 65536:
                raise RuntimeError("Sampler baseline exceeded the expected message size")
        return json.loads(data)

    def close_streams(self) -> None:
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        self.stderr.close()

    def ensure_running(self) -> None:
        if self.process is None or self.process.poll() is not None:
            error = (self.directory / "sampler.stderr.log").read_text(encoding="utf-8")
            raise RuntimeError(f"Independent sampler is not running:\n{error}")

    def stop(self) -> None:
        if self.process is None:
            raise RuntimeError("Sampler was not started")
        try:
            self.process.communicate("stop\n", timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            stop_owned_process(self.process)
            raise TimeoutError("Independent sampler shutdown timed out; owned sampler was stopped") from error
        except BaseException:
            stop_owned_process(self.process)
            raise
        finally:
            self.close_streams()
        if self.process.returncode != 0:
            error = (self.directory / "sampler.stderr.log").read_text(encoding="utf-8")
            raise RuntimeError(f"Independent sampler failed:\n{error}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


class CudaSynchronizer:
    def __init__(self) -> None:
        library = ctypes.util.find_library("cudart")
        if library is None:
            raise RuntimeError("CUDA runtime library not found; explicit synchronization is required")
        self.library = library
        self.runtime = ctypes.CDLL(library)
        self.runtime.cudaDeviceSynchronize.argtypes = []
        self.runtime.cudaDeviceSynchronize.restype = ctypes.c_int
        self.runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.runtime.cudaGetErrorString.restype = ctypes.c_char_p

    def __call__(self) -> None:
        status = self.runtime.cudaDeviceSynchronize()
        if status != 0:
            message = self.runtime.cudaGetErrorString(status)
            raise RuntimeError(f"cudaDeviceSynchronize failed ({status}): {message!r}")


def validate_input_capacity(input_ids: Sequence[int], capacity: int, output_tokens: int) -> None:
    if not input_ids:
        raise ValueError("Tokenizer produced an empty prompt")
    if len(input_ids) + output_tokens > capacity:
        raise ValueError(
            f"Actual input ({len(input_ids)}) + output limit ({output_tokens}) exceeds "
            f"shared requested capacity ({capacity}); the context label is not a token count"
        )


def token_metrics(started_ns: int, ids: Sequence[int], times_ns: Sequence[int]) -> dict[str, Any]:
    if not ids or len(ids) != len(times_ns):
        raise ValueError("Each generated token must have one CPU-availability timestamp")
    if times_ns[0] <= started_ns or any(b <= a for a, b in zip(times_ns, times_ns[1:])):
        raise ValueError("Token timestamps must strictly increase after inference starts")
    decode_seconds = (times_ns[-1] - times_ns[0]) / 1e9
    return {
        "completion_tokens": len(ids),
        "output_token_ids": list(ids),
        "token_timestamps_ns": list(times_ns),
        "ttft_ms": (times_ns[0] - started_ns) / 1e6,
        "total_latency_ms": (times_ns[-1] - started_ns) / 1e6,
        "decode_tokens_per_second": (len(ids) - 1) / decode_seconds if len(ids) >= 2 else None,
    }


def measure_tokens(
    prefill: Callable[[], None],
    next_token: Callable[[], int],
    is_eos: Callable[[int], bool],
    synchronize: Callable[[], None],
    output_tokens: int,
    phases: dict[str, int],
) -> dict[str, Any]:
    if output_tokens < 1:
        raise ValueError("The output token limit must be positive")
    ids: list[int] = []
    timestamps: list[int] = []
    synchronize()
    phases["generation_start_ns"] = time.perf_counter_ns()
    prefill()
    synchronize()
    phases["prefill_end_ns"] = time.perf_counter_ns()
    stop_reason = "length"
    for _ in range(output_tokens):
        try:
            token = int(next_token())
        except StopIteration as error:
            raise RuntimeError("Runtime stopped without returning an EOS token or reaching the limit") from error
        available_ns = time.perf_counter_ns()
        ids.append(token)
        timestamps.append(available_ns)
        if is_eos(token):
            stop_reason = "eos"
            break
    phases["first_token_ns"] = timestamps[0]
    phases["last_token_ns"] = timestamps[-1]
    request_start = phases.get("request_start_ns", phases["generation_start_ns"])
    return {
        **token_metrics(phases["generation_start_ns"], ids, timestamps),
        "prompt_processing_ms": (phases["prefill_end_ns"] - phases["generation_start_ns"]) / 1e6,
        "prefill_to_first_token_ms": (timestamps[0] - phases["prefill_end_ns"]) / 1e6,
        "request_setup_ms": (phases["generation_start_ns"] - request_start) / 1e6,
        "request_ttft_ms": (timestamps[0] - request_start) / 1e6,
        "request_total_latency_ms": (timestamps[-1] - request_start) / 1e6,
        "stop_reason": stop_reason,
        "eos_included_in_completion_tokens": stop_reason == "eos",
    }


def input_metadata(input_ids: Sequence[int], policy: str) -> dict[str, Any]:
    return {
        "prompt_tokens": len(input_ids),
        "input_token_ids_sha256": hashlib.sha256(json.dumps(list(input_ids)).encode()).hexdigest(),
        "special_token_policy": policy,
        "chat_template_applied": False,
    }


def inspect_ort_cache(generator: Any, config: dict[str, Any], capacity: int) -> dict[str, Any]:
    decoder = config["model"]["decoder"]
    tensors = []
    for layer in range(decoder["num_hidden_layers"]):
        for kind in ("past_key_names", "past_value_names"):
            name = decoder["inputs"][kind] % layer
            tensor = generator.get_input(name)
            shape = list(tensor.shape)
            if len(shape) != 4 or shape[0] != 1 or shape[2] != capacity or str(tensor.dtype) != "float16":
                raise RuntimeError(
                    f"Unsupported KV allocation for {name}: shape={shape}, dtype={tensor.dtype}; "
                    f"expected batch 1, capacity {capacity}, FP16"
                )
            tensors.append({"name": name, "shape": shape, "dtype": str(tensor.dtype), "bytes": tensor.nbytes})
            del tensor
    if not tensors:
        raise RuntimeError("No ORT KV tensors were inspected")
    return {
        "effective_sequence_capacity": capacity,
        "kv_cache_dtype": "float16",
        "kv_cache_tensor_bytes": sum(tensor["bytes"] for tensor in tensors),
        "kv_cache_tensors": tensors,
        "kv_cache_observation": "All past K/V input tensor shapes inspected after timing; shared outputs not double-counted",
    }


def prepare_request(
    trial: Trial, sampler: ResourceSampler, worker_phases: dict[str, int],
    index: int, label: str,
) -> tuple[Path, dict[str, int], dict[str, Any]]:
    sampler.ensure_running()
    config_digest(Path(trial.ort_model), trial.original_config_sha256)
    with nvml_device() as (nvml, handle):
        baseline = device_snapshot(nvml, handle)
        check_gpu_processes(baseline, {os.getpid()})
    directory = Path(trial.directory) / f"request-{index}-{label}"
    directory.mkdir(exist_ok=False)
    metadata = {
        "request_index": index, "request_kind": label,
        "request_temperature": "cold" if index == 0 else "warm",
        "requests_per_worker": len(request_labels(trial)),
        "request_directory": str(directory),
        "pre_request_vram_mib": as_mib(baseline["vram_bytes"]),
        "pre_request_snapshot": baseline,
    }
    return directory, dict(worker_phases), metadata


def finish_request(
    trial: Trial, sampler: ResourceSampler, directory: Path, phases: dict[str, int],
    input_ids: Sequence[int], result: dict[str, Any],
) -> dict[str, Any]:
    sampler.ensure_running()
    digest = config_digest(Path(trial.ort_model), trial.original_config_sha256)
    result.update({
        "phases_ns": phases, "original_config_sha256_request": digest,
        "inspection_ms": (phases["inspection_end_ns"] - phases["inspection_start_ns"]) / 1e6,
        "request_cleanup_ms": (phases["request_cleanup_end_ns"] - phases["request_cleanup_start_ns"]) / 1e6,
    })
    write_json(directory / "tokens.json", {
        "input_token_ids": list(input_ids), "output_token_ids": result["output_token_ids"],
    })
    write_json(directory / "timing.json", result)
    return result


def ort_request(
    og: Any, model: Any, config: dict[str, Any], input_ids: list[int], eos_ids: set[int],
    trial: Trial, sampler: ResourceSampler, worker_phases: dict[str, int],
    synchronize: CudaSynchronizer, index: int, label: str,
) -> dict[str, Any]:
    directory, phases, metadata = prepare_request(trial, sampler, worker_phases, index, label)
    phases["request_start_ns"] = time.perf_counter_ns()
    phases["generator_setup_start_ns"] = phases["request_start_ns"]
    params = og.GeneratorParams(model)
    options: dict[str, Any] = {
        "max_length": trial.sequence_capacity, "batch_size": 1, "do_sample": False,
    }
    if trial.chunk_size is not None:
        options["chunk_size"] = trial.chunk_size
    params.set_search_options(**options)
    generator = og.Generator(model, params)
    try:
        if trial.profile_dir is not None:
            profile_dir = Path(trial.profile_dir)
            profile_dir.mkdir(parents=True, exist_ok=False)
            generator.set_runtime_option("enable_profiling", str(profile_dir / "ort-profile"))
        synchronize()
        phases["generator_setup_end_ns"] = time.perf_counter_ns()
        initial_tokens = int(generator.token_count())
        if initial_tokens != 0:
            raise RuntimeError(f"New ORT generator contains {initial_tokens} tokens; refusing prefix reuse")

        def next_token() -> int:
            if generator.is_done():
                raise RuntimeError("ORT ended before an observed EOS token or the requested output limit")
            generator.generate_next_token()
            tokens = generator.get_next_tokens()
            if len(tokens) != 1:
                raise RuntimeError(f"Expected one ORT token ID, got {len(tokens)}")
            return int(tokens[0])

        timing = measure_tokens(
            lambda: generator.append_tokens(input_ids), next_token, eos_ids.__contains__,
            synchronize, trial.output_tokens, phases,
        )
        phases["inspection_start_ns"] = time.perf_counter_ns()
        sequence = [int(token) for token in generator.get_sequence(0)]
        if sequence != input_ids + timing["output_token_ids"]:
            raise RuntimeError("ORT sequence does not match this request's full prompt and generated IDs")
        cache = inspect_ort_cache(generator, config, trial.sequence_capacity)
        write_json(directory / "kv-cache.json", cache)
        phases["inspection_end_ns"] = time.perf_counter_ns()
        result = {
            **metadata, **timing, **input_metadata(input_ids, "og.Tokenizer.encode with artifact tokenizer defaults"),
            "effective_sequence_capacity": cache["effective_sequence_capacity"],
            "kv_cache_dtype": cache["kv_cache_dtype"],
            "kv_cache_tensor_bytes": cache["kv_cache_tensor_bytes"],
            "past_present_share_buffer": config["search"]["past_present_share_buffer"],
            "ort_provider_options": config["model"]["decoder"]["session_options"]["provider_options"],
            "runtime_version": og.__version__, "runtime_commit": og.__commit__,
            "cuda_runtime_library": synchronize.library,
            "chunk_size": trial.chunk_size,
            "fresh_state_method": "New og.Generator on the retained model for every request",
            "state_tokens_before": initial_tokens,
            "state_tokens_after": int(generator.token_count()),
            "full_sequence_verified": True, "no_prefix_reuse_verified": True,
        }
    finally:
        phases["request_cleanup_start_ns"] = time.perf_counter_ns()
        del generator, params
        synchronize()
        phases["request_cleanup_end_ns"] = time.perf_counter_ns()
    return finish_request(trial, sampler, directory, phases, input_ids, result)


def run_ort(
    trial: Trial, phases: dict[str, int], sampler: ResourceSampler
) -> list[dict[str, Any]]:
    import onnxruntime_genai as og

    synchronize = CudaSynchronizer()
    with ort_model_variant(
        Path(trial.ort_model), trial.ort_allocator_mode == "device", trial.sequence_capacity
    ) as active_model_path:
        config = json.loads((active_model_path / "genai_config.json").read_text(encoding="utf-8"))
        phases["model_load_start_ns"] = time.perf_counter_ns()
        model = og.Model(str(active_model_path))
        tokenizer = None
        try:
            synchronize()
            phases["model_load_end_ns"] = time.perf_counter_ns()
            if model.device_type.lower() != "cuda":
                raise RuntimeError(f"ORT is not using CUDA: {model.device_type}")
            phases["tokenization_start_ns"] = time.perf_counter_ns()
            tokenizer = og.Tokenizer(model)
            input_ids = [int(token) for token in tokenizer.encode(trial.prompt)]
            eos_ids = {int(token) for token in tokenizer.eos_token_ids}
            if not eos_ids:
                raise RuntimeError("ORT tokenizer did not expose EOS IDs")
            validate_input_capacity(input_ids, trial.sequence_capacity, trial.output_tokens)
            phases["tokenization_end_ns"] = time.perf_counter_ns()
            results = [
                ort_request(
                    og, model, config, input_ids, eos_ids, trial, sampler, phases,
                    synchronize, index, label,
                )
                for index, label in enumerate(request_labels(trial))
            ]
        finally:
            phases["cleanup_start_ns"] = time.perf_counter_ns()
            del tokenizer, model
    phases["cleanup_end_ns"] = time.perf_counter_ns()
    return results


def reset_llama_state(model: Any, api: Any, synchronize: Callable[[], None]) -> Any:
    model.set_cache(None)
    model.reset()
    memory = api.llama_get_memory(model.ctx)
    if memory is None:
        raise RuntimeError("llama.cpp has no inspectable KV memory; fresh state cannot be verified")
    api.llama_memory_clear(memory, True)
    synchronize()
    positions = (api.llama_memory_seq_pos_min(memory, 0), api.llama_memory_seq_pos_max(memory, 0))
    if model.n_tokens != 0 or positions != (-1, -1) or model.cache is not None:
        raise RuntimeError(f"llama.cpp state was not cleared: n_tokens={model.n_tokens}, positions={positions}")
    return memory


def llama_request(
    model: Any, api: Any, vocab: Any, cache: dict[str, Any], input_ids: list[int],
    trial: Trial, sampler: ResourceSampler, worker_phases: dict[str, int],
    synchronize: CudaSynchronizer, index: int, label: str,
) -> dict[str, Any]:
    directory, phases, metadata = prepare_request(trial, sampler, worker_phases, index, label)
    phases["request_start_ns"] = time.perf_counter_ns()
    phases["generator_setup_start_ns"] = phases["request_start_ns"]
    memory = reset_llama_state(model, api, synchronize)
    # Explicit full eval plus reset=False bypasses generate()'s prefix-match optimization.
    tokens = model.generate([], reset=False, temp=0.0, repeat_penalty=1.0)
    phases["generator_setup_end_ns"] = time.perf_counter_ns()
    try:
        timing = measure_tokens(
            lambda: model.eval(input_ids), lambda: next(tokens),
            lambda token: api.llama_vocab_is_eog(vocab, token),
            synchronize, trial.output_tokens, phases,
        )
        phases["inspection_start_ns"] = time.perf_counter_ns()
        expected_evaluated = len(input_ids) + timing["completion_tokens"] - 1
        positions = (api.llama_memory_seq_pos_min(memory, 0), api.llama_memory_seq_pos_max(memory, 0))
        if model.n_tokens != expected_evaluated or positions != (0, expected_evaluated - 1):
            raise RuntimeError(
                f"llama.cpp evaluated state does not match this full request: "
                f"n_tokens={model.n_tokens}, positions={positions}, expected={expected_evaluated}"
            )
        write_json(directory / "kv-cache.json", {**cache, "native_sequence_positions_after": positions})
        phases["inspection_end_ns"] = time.perf_counter_ns()
        result = {
            **metadata, **timing, **input_metadata(input_ids, "Llama.tokenize(add_bos=False, special=False)"),
            "effective_sequence_capacity": cache["effective_sequence_capacity"],
            "kv_cache_dtype": cache["kv_cache_dtype"],
            "runtime_version": api.__version__, "cuda_runtime_library": synchronize.library,
            "fresh_state_method": "Llama.reset + llama_memory_clear(data=True), no completion cache; explicit full eval",
            "state_tokens_before": 0, "state_tokens_after": int(model.n_tokens),
            "native_sequence_positions_before": [-1, -1], "native_sequence_positions_after": list(positions),
            "no_prefix_reuse_verified": True,
            "artifact_file_type": model.metadata["general.file_type"],
            "flash_attn_type_requested": cache["flash_attn_type_requested"],
        }
    finally:
        phases["request_cleanup_start_ns"] = time.perf_counter_ns()
        tokens.close()
        synchronize()
        phases["request_cleanup_end_ns"] = time.perf_counter_ns()
    return finish_request(trial, sampler, directory, phases, input_ids, result)


def run_llama(
    trial: Trial, phases: dict[str, int], sampler: ResourceSampler
) -> list[dict[str, Any]]:
    import llama_cpp
    from llama_cpp import Llama

    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("Installed llama.cpp does not support GPU offload")
    synchronize = CudaSynchronizer()
    phases["model_load_start_ns"] = time.perf_counter_ns()
    model = Llama(
        model_path=trial.gguf_model, n_gpu_layers=-1, main_gpu=0,
        split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, n_ctx=trial.sequence_capacity,
        type_k=llama_cpp.GGML_TYPE_F16, type_v=llama_cpp.GGML_TYPE_F16, verbose=True,
    )
    try:
        synchronize()
        phases["model_load_end_ns"] = time.perf_counter_ns()
        effective_capacity = int(llama_cpp.llama_n_ctx_seq(model.ctx))
        if effective_capacity != trial.sequence_capacity:
            raise RuntimeError(
                f"llama.cpp effective capacity {effective_capacity} differs from shared requested "
                f"capacity {trial.sequence_capacity}; choose a capacity supported by both runtimes"
            )
        native_model = llama_cpp.llama_get_model(model.ctx)
        if native_model is None:
            raise RuntimeError("llama.cpp did not expose its model handle")
        vocab = llama_cpp.llama_model_get_vocab(native_model)
        if vocab is None:
            raise RuntimeError("llama.cpp did not expose its vocabulary")
        phases["tokenization_start_ns"] = time.perf_counter_ns()
        input_ids = model.tokenize(trial.prompt.encode("utf-8"), add_bos=False, special=False)
        validate_input_capacity(input_ids, trial.sequence_capacity, trial.output_tokens)
        phases["tokenization_end_ns"] = time.perf_counter_ns()
        cache = {
            "effective_sequence_capacity": effective_capacity,
            "effective_total_context_capacity": model.n_ctx(),
            "kv_cache_dtype": "float16",
            "type_k": model.context_params.type_k,
            "type_v": model.context_params.type_v,
            "n_layers": llama_cpp.llama_model_n_layer(native_model),
            "n_kv_heads": llama_cpp.llama_model_n_head_kv(native_model),
            "kv_cache_observation": "Native n_ctx_seq and explicit FP16 configuration; native allocation log in worker.stderr.log",
            "flash_attn_type_requested": model.context_params.flash_attn_type,
            "offload_kqv": bool(model.context_params.offload_kqv),
        }
        results = [
            llama_request(
                model, llama_cpp, vocab, cache, input_ids, trial, sampler, phases,
                synchronize, index, label,
            )
            for index, label in enumerate(request_labels(trial))
        ]
    finally:
        phases["cleanup_start_ns"] = time.perf_counter_ns()
        model.close()
        phases["cleanup_end_ns"] = time.perf_counter_ns()
    return results


def summarize_samples(path: Path, phases: dict[str, int]) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        samples = [
            {key: int(row[key]) for key in ("time_ns", "query_finished_ns", "vram_bytes", "reserved_bytes", "ram_bytes")}
            for row in csv.DictReader(stream)
        ]
    if len(samples) < 2:
        raise RuntimeError("Independent sampler produced fewer than two samples")
    timestamps = [sample["time_ns"] for sample in samples]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise RuntimeError("Sampler timestamps are not strictly increasing")
    if any(sample["query_finished_ns"] < sample["time_ns"] for sample in samples):
        raise RuntimeError("Sample acquisition ended before it started")
    if samples[0]["query_finished_ns"] >= phases["model_load_start_ns"] or timestamps[-1] <= phases["last_token_ns"]:
        raise RuntimeError("Samples do not bracket model loading and inference")

    def in_window(start: int, end: int) -> list[dict[str, int]]:
        return [sample for sample in samples if start <= sample["time_ns"] and sample["query_finished_ns"] <= end]

    request_start = phases.get("request_start_ns", timestamps[0])
    request_samples = in_window(request_start, phases["last_token_ns"])
    preparation_end = phases.get("tokenization_end_ns", phases["model_load_end_ns"])
    benchmark_samples = in_window(timestamps[0], preparation_end) + request_samples
    result: dict[str, Any] = {
        "pre_load_vram_mib": as_mib(samples[0]["vram_bytes"]),
        "pre_load_reserved_vram_mib": as_mib(samples[0]["reserved_bytes"]),
        "nvml_memory_api": "v2; used excludes driver-reserved memory, recorded separately",
        "peak_vram_mib": as_mib(max(sample["vram_bytes"] for sample in benchmark_samples)),
        "peak_ram_mib": as_mib(max(sample["ram_bytes"] for sample in benchmark_samples)),
        "sample_count": len(samples),
        "max_sample_gap_ms": max(b - a for a, b in zip(timestamps, timestamps[1:])) / 1e6,
        "median_sample_gap_ms": statistics.median(b - a for a, b in zip(timestamps, timestamps[1:])) / 1e6,
        "request_peak_vram_mib": as_mib(max(sample["vram_bytes"] for sample in request_samples)) if request_samples else None,
        "request_sample_count": len(request_samples),
        "sample_window_policy": "Acquisition start and finish must both be inside the selected interval",
    }
    for name, start, end in (
        ("load", "model_load_start_ns", "model_load_end_ns"),
        ("prefill", "generation_start_ns", "prefill_end_ns"),
        ("decode", "first_token_ns", "last_token_ns"),
        ("inference", "generation_start_ns", "last_token_ns"),
    ):
        selected = [sample["vram_bytes"] for sample in in_window(phases[start], phases[end])]
        result[f"{name}_sample_count"] = len(selected)
        result[f"{name}_peak_vram_mib"] = as_mib(max(selected)) if selected else None
    return result


def trial_worker(request: Path) -> int:
    gpu_zero_environment()
    trial = Trial(**json.loads(request.read_text(encoding="utf-8")))
    if trial.cold_warm and trial.profile_dir is not None:
        raise ValueError("Cold/warm diagnostics must be unprofiled")
    model_path = Path(trial.ort_model)
    config_digest(model_path, trial.original_config_sha256)
    phases = {"worker_start_ns": time.perf_counter_ns()}
    execution_gpu_uuid = validate_execution_gpu_identity()
    try:
        with ResourceSampler(Path(trial.directory), trial.sampling_interval_ms) as sampler:
            if trial.runtime == "ort":
                results = run_ort(trial, phases, sampler)
            elif trial.runtime == "llama_cpp":
                results = run_llama(trial, phases, sampler)
            else:
                raise ValueError(f"Unsupported runtime {trial.runtime}")
        for result in results:
            result.update(summarize_samples(Path(trial.directory) / "samples.csv", result["phases_ns"]))
            result.update({
                "schema_version": SCHEMA_VERSION, "status": "complete",
                "runtime": trial.runtime, "ort_allocator_mode": trial.ort_allocator_mode,
                "context_label": trial.context_label, "repetition": trial.repetition,
                "max_output_tokens": trial.output_tokens, "requested_sequence_capacity": trial.sequence_capacity,
                "prompt_text_sha256": hashlib.sha256(trial.prompt.encode("utf-8")).hexdigest(),
                "worker_pid": os.getpid(), "sampler_pid": sampler.baseline["sampler_pid"],
                "pre_load_gpu_process_pids": sampler.baseline["gpu_process_pids"],
                "gpu_uuid": sampler.baseline["gpu_uuid"], "gpu_index": 0,
                "execution_gpu_uuid": execution_gpu_uuid,
                "gpu_identity_policy": "Physical NVML GPU 0; UUID visibility; CUDA driver identity checked before runtime import",
                "sampling_interval_ms": trial.sampling_interval_ms,
                "profiled": trial.profile_dir is not None, "worker_phases_ns": phases,
                "model_load_ms": (phases["model_load_end_ns"] - phases["model_load_start_ns"]) / 1e6,
                "tokenization_ms": (phases["tokenization_end_ns"] - phases["tokenization_start_ns"]) / 1e6,
                "model_ready_ms": (phases["tokenization_end_ns"] - phases["model_load_start_ns"]) / 1e6,
                "synchronization": "cudaDeviceSynchronize before timing and after prefill; each token returned as a CPU ID",
                "trial_directory": trial.directory,
            })
    finally:
        config_digest(model_path, trial.original_config_sha256)
    write_json(Path(trial.directory) / "result.json", {"schema_version": SCHEMA_VERSION, "requests": results})
    return 0


def run_trial(trial: Trial) -> list[dict[str, Any]]:
    directory = Path(trial.directory)
    directory.mkdir(parents=True, exist_ok=False)
    config_digest(Path(trial.ort_model), trial.original_config_sha256)
    before = idle_gpu_snapshot()
    write_json(directory / "request.json", asdict(trial))
    worker_error = ""
    with (directory / "worker.stdout.log").open("x") as stdout, (directory / "worker.stderr.log").open("x") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--trial-worker", str(directory / "request.json")],
            stdout=stdout, stderr=stderr, text=True,
        )
        try:
            exit_code = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            stop_owned_worker(process)
            exit_code = None
            worker_error = f"Trial worker exceeded {WORKER_TIMEOUT_SECONDS} seconds and was stopped"
        except BaseException:
            stop_owned_worker(process)
            raise
    integrity = True
    integrity_error = ""
    after_hash = None
    try:
        after_hash = config_digest(Path(trial.ort_model), trial.original_config_sha256)
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        integrity = False
        integrity_error = str(error)
    if exit_code != 0 or not integrity:
        raw_error = (directory / "worker.stderr.log").read_text(encoding="utf-8")
        print(raw_error, file=sys.stderr)
        results: list[dict[str, Any]] = [{
            "schema_version": SCHEMA_VERSION, "status": "error", "runtime": trial.runtime,
            "ort_allocator_mode": trial.ort_allocator_mode, "context_label": trial.context_label,
            "repetition": trial.repetition,
            "error": integrity_error or worker_error or f"Worker exited with {exit_code}",
            "trial_directory": trial.directory,
        }]
    else:
        results = json.loads((directory / "result.json").read_text(encoding="utf-8"))["requests"]
        if [result["request_kind"] for result in results] != list(request_labels(trial)):
            raise RuntimeError("Worker returned missing, duplicate, or out-of-order requests")
    after = idle_gpu_snapshot()
    for result in results:
        result.update({
            "worker_exit_code": exit_code,
            "parent_pre_load_vram_mib": as_mib(before["vram_bytes"]),
            "post_exit_vram_mib": as_mib(after["vram_bytes"]),
            "post_exit_gpu_process_pids": after["gpu_process_pids"],
            "original_config_sha256_before": trial.original_config_sha256,
            "original_config_sha256_after": after_hash,
            "original_config_integrity": integrity,
        })
    return results


def reserve_output_directory(output_dir: Path) -> None:
    for name in ("benchmark-run.json", "results.jsonl", "results.csv", "report.md", "trials"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing benchmark output: {output_dir / name}")
    output_dir.mkdir(parents=True, exist_ok=True)


def file_provenance(path: Path, hash_contents: bool = True) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        with path.open("rb") as stream:
            record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return record


def collect_package_versions() -> dict[str, str | None]:
    required = (
        "onnxruntime-genai-cuda", "onnxruntime-gpu", "llama-cpp-python",
        "numpy", "psutil", "nvidia-ml-py",
    )
    versions: dict[str, str | None] = {name: importlib.metadata.version(name) for name in required}
    try:
        versions["pynvml"] = importlib.metadata.version("pynvml")
    except importlib.metadata.PackageNotFoundError:
        versions["pynvml"] = None
    return versions


def collect_run_provenance(ort_model: Path, gguf_model: Path) -> dict[str, Any]:
    root = SCRIPT.parent.parent
    ort_model, gguf_model = ort_model.resolve(), gguf_model.resolve()
    versions = collect_package_versions()
    genai_distribution = importlib.metadata.distribution("onnxruntime-genai-cuda")
    llama_distribution = importlib.metadata.distribution("llama-cpp-python")
    genai_init = Path(genai_distribution.locate_file("onnxruntime_genai/__init__.py"))
    commits = [
        node.value.value for node in ast.parse(genai_init.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and any(isinstance(target, ast.Name) and target.id == "__commit__" for target in node.targets)
    ]
    if len(commits) != 1 or not isinstance(commits[0], str):
        raise RuntimeError("Installed ORT GenAI source commit is not identifiable")
    source_paths = (
        SCRIPT, SCRIPT.parent / "evidence.py", root / "README.md",
        root / "FINDINGS.md", root / "REPRODUCING.md",
        root / "cuda" / "requirements.txt",
        genai_init, Path(llama_distribution.locate_file("llama_cpp/llama.py")),
        Path(llama_distribution.locate_file("llama_cpp/_internals.py")),
    )
    metadata_symbols = {"file_provenance", "collect_run_provenance", "write_report", "main", "run_matrix"}
    measurement_nodes = [
        node for node in ast.parse(SCRIPT.read_bytes()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in metadata_symbols
    ]
    measurement_ast = "\n".join(ast.dump(node, include_attributes=False) for node in measurement_nodes)
    recipe_path = ort_model.parent / "config.json"
    receipt_path = gguf_model.parent / ".cache" / "huggingface" / "download" / f"{gguf_model.name}.metadata"
    discovery_path = root / "results" / gguf_model.parent.parent.name / "discovered.env"
    warnings = []
    recipe = None
    if recipe_path.is_file():
        recipe = {"file": file_provenance(recipe_path), "config": json.loads(recipe_path.read_text(encoding="utf-8"))}
    else:
        warnings.append(f"ORT conversion recipe is unavailable at {recipe_path}")
    receipt = None
    if receipt_path.is_file():
        lines = receipt_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 3:
            raise ValueError(f"Unexpected Hugging Face download receipt: {receipt_path}")
        receipt = {
            "file": file_provenance(receipt_path), "repository_revision": lines[0],
            "recorded_etag": lines[1], "download_timestamp": float(lines[2]),
        }
    else:
        warnings.append(f"GGUF download receipt is unavailable at {receipt_path}")
    gguf_repo = None
    if discovery_path.is_file():
        values = [
            line.partition("=")[2] for line in discovery_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("GGUF_REPO=")
        ]
        if len(values) != 1 or not values[0]:
            raise ValueError(f"GGUF repository is not identifiable in {discovery_path}")
        gguf_repo = {"repository": values[0], "source": file_provenance(discovery_path)}
    else:
        warnings.append(f"GGUF repository discovery record is unavailable at {discovery_path}")
    for warning in warnings:
        print(f"PROVENANCE WARNING: {warning}", file=sys.stderr, flush=True)
    reference = None
    if reference_path := os.environ.get("BENCHMARK_REFERENCE_RUN"):
        reference_dir = Path(reference_path).resolve()
        reference_manifest = json.loads((reference_dir / "benchmark-run.json").read_text(encoding="utf-8"))
        if reference_manifest["original_config_sha256"] != config_digest(ort_model):
            raise RuntimeError("Original config no longer matches the specified reference run")
        reference = {
            "directory": str(reference_dir),
            "files": [file_provenance(reference_dir / name) for name in ("benchmark-run.json", "results.jsonl", "results.csv", "report.md")],
            "runner_sha256": reference_manifest["runner_sha256"],
        }
    with nvml_device() as (nvml, _handle):
        driver_version = nvml.nvmlSystemGetDriverVersion()
    return {
        "python": {"executable": sys.executable, "version": sys.version},
        "package_versions": versions, "nvidia_driver_version": driver_version,
        "nvml_python_provider": {"module": "pynvml", "required_distribution": "nvidia-ml-py",
                                 "optional_legacy_distribution": "pynvml"},
        "ort_genai_source_commit": commits[0],
        "source_files": [file_provenance(path) for path in source_paths],
        "measurement_ast_sha256": hashlib.sha256(measurement_ast.encode()).hexdigest(),
        "measurement_symbols": [node.name for node in measurement_nodes],
        "models": {
            "ort": {
                "directory": str(ort_model), "conversion_recipe": recipe,
                "configurations": [file_provenance(path) for path in sorted(ort_model.glob("*.json"))],
                "artifacts": [file_provenance(path, hash_contents=path.suffix == ".onnx")
                              for path in sorted(ort_model.glob("*.onnx*")) if path.is_file()],
            },
            "gguf": {"artifact": file_provenance(gguf_model, hash_contents=False),
                     "repository_record": gguf_repo, "download_receipt": receipt},
        },
        "large_weight_hash_policy": "Large weight files are inventoried by path/size/mtime, not rehashed. The GGUF etag is recorded download metadata, not a fresh content verification.",
        "reference_run": reference, "provenance_warnings": warnings,
    }


def write_report(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        } for row in rows)
    lines = [
        "# Isolated ORT GenAI versus llama.cpp requests (schema 3)",
        "",
        "Context labels are prompt-construction targets, not measured input lengths.",
        "TTFT includes prompt evaluation through the first CPU-available token ID; tokenization and model setup are excluded.",
        "Request TTFT additionally includes fresh generation/KV setup and its state checks. Model loading and tokenization are reported separately.",
        "Decode = (N - 1) / (last-token time - first-token time); unavailable for N < 2. Counts include a generated EOS ID.",
        "NVML samples run in an independent process. Request peaks cover state setup and inference; load-inclusive peaks add shared model preparation.",
        "Acquisition intervals must fit wholly inside a phase. Post-timing KV inspection and cleanup are excluded from request peaks.",
        "Cold = first request in a fresh worker; warm_1 and warm_2 retain that worker's model with fresh generation/KV state and no prompt-prefix reuse.",
        "No requests are discarded as hidden warmups. Native kernel/allocator choices and different quantized artifacts remain comparability limits.",
        "Profiled rows are diagnostic only. See benchmark-run.json, trial metadata, raw samples and native logs.",
        "",
        "| Runtime | Allocator | Context label | Worker rep | Request | Input IDs | Output IDs | Stop | Prompt ms | TTFT ms | Request TTFT ms | Decode tok/s | Pre-load MiB | Pre-request MiB | Request peak MiB | Config intact | Status |",
        "|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        names = (
            "runtime", "ort_allocator_mode", "context_label", "repetition", "request_kind", "prompt_tokens",
            "completion_tokens", "stop_reason", "prompt_processing_ms", "ttft_ms", "request_ttft_ms",
            "decode_tokens_per_second", "pre_load_vram_mib", "pre_request_vram_mib",
            "request_peak_vram_mib", "original_config_integrity", "status",
        )
        lines.append("| " + " | ".join(str(row.get(name)) if row.get(name) is not None else "n/a" for name in names) + " |")
        if row.get("status") == "error":
            lines.append(f"\nTrial failed: {row['error']}\n")
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "complete":
            key = (row["runtime"], row["ort_allocator_mode"], row["context_label"], row["request_kind"])
            groups.setdefault(key, []).append(row)
    if groups:
        lines += [
            "", "## Medians by request kind (never pooled across cold/warm)",
            "",
            "Each metric is median [minimum, maximum] across independent worker repetitions. Partial groups show the number of completed workers.",
            "",
            "| Runtime | Allocator | Label | Request | Workers | Prompt ms | TTFT ms | Request TTFT ms | Decode tok/s | Pre-request MiB | Request peak MiB | Actual input/output IDs and stop per worker |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for key, group in sorted(groups.items()):
            if len({row["repetition"] for row in group}) != len(group):
                raise RuntimeError(f"Duplicate request records within a worker repetition: {key}")
            cells = [*(str(value) for value in key), str(len(group))]
            for name in ("prompt_processing_ms", "ttft_ms", "request_ttft_ms", "decode_tokens_per_second", "pre_request_vram_mib", "request_peak_vram_mib"):
                values = [row[name] for row in group]
                cells.append(
                    f"{statistics.median(values):.3f} [{min(values):.3f}, {max(values):.3f}]"
                    if all(value is not None for value in values) else "n/a"
                )
            cells.append("; ".join(
                f"r{row['repetition']}: {row['prompt_tokens']}/{row['completion_tokens']} {row['stop_reason']}"
                for row in sorted(group, key=lambda row: row["repetition"])
            ))
            lines.append("| " + " | ".join(cells) + " |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--sample-worker"]:
        if len(argv) != 4:
            raise ValueError("Sampler requires directory, worker PID, and sampling interval")
        return sample_worker(Path(argv[1]), int(argv[2]), float(argv[3]))
    if argv[:1] == ["--trial-worker"]:
        if len(argv) != 2:
            raise ValueError("Trial worker requires one request file")
        def terminate_worker(signum: int, _frame: Any) -> None:
            raise InterruptedError(f"Owned trial worker received signal {signum}")

        signal.signal(signal.SIGTERM, terminate_worker)
        return trial_worker(Path(argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort-model", type=Path, required=True)
    parser.add_argument("--gguf-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contexts", default="2048,4096,8192,16384,32768", help="Nominal prompt labels, not token counts")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--runtime", choices=("ort", "llama_cpp", "both"), default="both")
    parser.add_argument("--sampling-interval-ms", type=float, default=10.0)
    parser.add_argument("--ort-device-allocator", action="store_true")
    parser.add_argument("--ort-profile-dir", type=Path)
    parser.add_argument("--ort-chunk-size", type=int)
    parser.add_argument("--cold-warm", action="store_true", help="One cold request followed by two warm requests per fresh worker; unprofiled only")
    args = parser.parse_args(argv)
    contexts = [int(value) for value in args.contexts.split(",")]
    if not contexts or len(set(contexts)) != len(contexts) or any(value < 1 for value in contexts):
        parser.error("Context labels must be positive and unique")
    if args.output_tokens < 1 or args.repetitions < 1:
        parser.error("Output tokens and repetitions must be positive")
    if not math.isfinite(args.sampling_interval_ms) or args.sampling_interval_ms <= 0:
        parser.error("Sampling interval must be finite and positive")
    if args.ort_chunk_size is not None and args.ort_chunk_size < 1:
        parser.error("ORT chunk size must be positive")
    if args.cold_warm and args.ort_profile_dir is not None:
        parser.error("--cold-warm cannot be combined with profiling")
    gpu_zero_environment()
    with gpu_run_lock():
        return run_matrix(args, contexts, argv)


def run_matrix(args: argparse.Namespace, contexts: list[int], argv: list[str]) -> int:
    original_hash = config_digest(args.ort_model)
    initial_gpu = idle_gpu_snapshot()
    output_dir = args.output_dir.resolve()
    reserve_output_directory(output_dir)
    variants = []
    if args.runtime in ("ort", "both"):
        variants.append(("ort", "baseline"))
        if args.ort_device_allocator:
            variants.append(("ort", "device"))
    if args.runtime in ("llama_cpp", "both"):
        variants.append(("llama_cpp", "n/a"))
    command = [sys.executable, str(SCRIPT), *argv]
    requests_per_worker = 3 if args.cold_warm else 1
    write_json(output_dir / "benchmark-run.json", {
        "schema_version": SCHEMA_VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "command_shell": shlex.join(command),
        "working_directory": str(Path.cwd()), "initial_gpu": initial_gpu,
        "environment": {name: os.environ[name] for name in ("BENCHMARK_GPU_INDEX", "CUDA_VISIBLE_DEVICES")},
        "tmux_session": os.environ.get("BENCHMARK_TMUX_SESSION"),
        "exact_log_path": os.environ.get("BENCHMARK_LOG_PATH"),
        "launcher_command": os.environ.get("BENCHMARK_LAUNCH_COMMAND"),
        "provenance": collect_run_provenance(args.ort_model, args.gguf_model),
        "matrix": {
            "context_labels": contexts, "variants": variants,
            "worker_repetitions": args.repetitions, "requests_per_worker": requests_per_worker,
            "expected_workers": len(contexts) * len(variants) * args.repetitions,
            "expected_request_records": len(contexts) * len(variants) * args.repetitions * requests_per_worker,
            "requested_capacities": {str(context): requested_sequence_capacity(context, args.output_tokens) for context in contexts},
            "max_generated_ids": args.output_tokens,
        },
        "runner_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "original_config_sha256": original_hash,
        "capacity_policy": "Same requested capacity = round_up(context label + maximum output tokens, 256); each actual input must fit",
        "eos_policy": "Count every returned token ID, including a terminal EOS/EOG ID",
        "request_mode": "cold_warm" if args.cold_warm else "cold_only",
        "requests_per_worker": requests_per_worker,
        "profiled": args.ort_profile_dir is not None,
        "ttft_policy": "Inference TTFT includes full prompt eval; request TTFT also includes fresh generation/KV setup",
        "prefix_policy": "Fresh ORT Generator; llama.cpp reset and physical KV clear, explicit full prompt eval, no prefix matching",
    })
    rows: list[dict[str, Any]] = []
    prompt_hashes: dict[int, str] = {}
    with (output_dir / "results.jsonl").open("x", encoding="utf-8") as journal:
        for context in contexts:
            prompt = build_prompt(context)
            for repetition in range(1, args.repetitions + 1):
                for runtime, allocator in variants:
                    variant = f"ort-{allocator}" if runtime == "ort" else "llama_cpp"
                    relative_dir = Path(f"context-{context}") / f"repetition-{repetition}" / variant
                    profile_dir = str(args.ort_profile_dir.resolve() / relative_dir) if runtime == "ort" and args.ort_profile_dir else None
                    trial = Trial(
                        runtime, allocator, context, repetition, args.output_tokens,
                        requested_sequence_capacity(context, args.output_tokens), prompt, str(args.ort_model.resolve()),
                        str(args.gguf_model.resolve()), str(output_dir / "trials" / relative_dir),
                        original_hash, args.sampling_interval_ms, profile_dir, args.ort_chunk_size, args.cold_warm,
                    )
                    for result in run_trial(trial):
                        if result["status"] == "complete":
                            token_hash = result["input_token_ids_sha256"]
                            if context in prompt_hashes and prompt_hashes[context] != token_hash:
                                result.update(status="error", error="Runtime input token IDs differ; comparison stopped")
                            prompt_hashes[context] = token_hash
                        rows.append(result)
                        encoded = json.dumps(result, sort_keys=True, allow_nan=False)
                        journal.write(encoded + "\n")
                        journal.flush()
                        write_report(rows, output_dir)
                        print(encoded, flush=True)
                        if result["status"] != "complete":
                            return 1
    config_digest(args.ort_model, original_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
