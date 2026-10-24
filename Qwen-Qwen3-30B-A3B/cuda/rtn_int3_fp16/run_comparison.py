"""Sequential, matched-input PR33555/PR33569 benchmark driver for jiafa-dev."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import site
import subprocess
import sys


HERE = Path(__file__).resolve().parent
COMMITS = {
    "33555": "d75ac16f377c0c783d289046f5cf2efe6bb5f9bc",
    "33569": "4ba6c38384cf5b12d4aa4570cd15af2461c4d515",
}
ARTIFACT_COMMIT = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
SPEC = importlib.util.spec_from_file_location(
    "comparison_benchmark", HERE / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def plan_runs(lengths, rounds):
    if (
        rounds < 1
        or not lengths
        or len(set(lengths)) != len(lengths)
        or min(lengths) < 1
    ):
        raise ValueError("Require positive rounds and distinct positive lengths")
    return [
        {
            "round": round_index + 1,
            "input_length": length,
            "runtime": runtime,
            "label": f"round{round_index + 1}-length{length}-{runtime}",
        }
        for round_index in range(rounds)
        for length_index, length in enumerate(lengths)
        for runtime in (
            ("33555", "33569")
            if (round_index + length_index) % 2 == 0
            else ("33569", "33555")
        )
    ]


def artifact_identity(files):
    identity = {
        Path(item["path"]).name: (item["bytes"], item["sha256"]) for item in files
    }
    if not identity or len(identity) != len(files):
        raise ValueError("Missing or duplicate artifact files")
    return identity


def verify_identity(report, expected, prompt_hash=None, forced_tokens=None):
    metadata = report["metadata"]
    if metadata["artifact_commit"] != ARTIFACT_COMMIT:
        raise ValueError("Artifact commit mismatch")
    if metadata["runtime_commit"] != expected["commit"]:
        raise ValueError("Runtime commit mismatch")
    if artifact_identity(metadata["artifact_files"]) != expected["artifacts"]:
        raise ValueError("Artifact full hashes mismatch")
    if not metadata["gpu_uuid"] or metadata["gpu_uuid"] != expected["gpu_uuid"]:
        raise ValueError("GPU UUID mismatch")
    if metadata["providers"][0] != "CUDAExecutionProvider":
        raise ValueError("CUDA provider is not primary")
    binaries = metadata["runtime_binaries"]
    actual = {
        str(Path(item["path"]).resolve()): (item["bytes"], item["sha256"])
        for item in binaries
    }
    if (
        not actual
        or len(actual) != len(binaries)
        or any(
            expected["binaries"].get(path) != digest for path, digest in actual.items()
        )
    ):
        raise ValueError("Loaded runtime binary path/hash mismatch")
    if not any(
        Path(path).name == "libonnxruntime_providers_cuda.so" for path in actual
    ):
        raise ValueError("Expected CUDA provider binary not loaded")
    if metadata["prompt_sha256"] != benchmark.token_hash(metadata["prompt_ids"]):
        raise ValueError("Invalid prompt hash")
    if prompt_hash is not None and metadata["prompt_sha256"] != prompt_hash:
        raise ValueError("Prompt mismatch")
    forced_hash = None if forced_tokens is None else benchmark.token_hash(forced_tokens)
    if (
        metadata["forced_tokens"] != forced_tokens
        or metadata["forced_token_sha256"] != forced_hash
    ):
        raise ValueError("Forced token stream/hash mismatch")


def validate_comparison(report):
    if (
        report.get("error")
        or report.get("shape_equal") is not True
        or report.get("finite") is not True
    ):
        raise ValueError(f"Invalid logits comparison: {report}")
    if any(
        not isinstance(report.get(key), (int, float)) or not math.isfinite(report[key])
        for key in ("max_abs", "mean_abs", "rmse", "top1_agreement")
    ):
        raise ValueError("Invalid/nonfinite comparison metrics")
    return report


def summarize_runs(records, output_length):
    def aggregate(items, length):
        rows = [row for item in items for row in item["report"]["trials"]]
        result = benchmark.summarize(rows, length, output_length)
        result.update(
            {
                "trials": len(rows),
                "sessions": len(items),
                "total_prefill_seconds": sum(row["prefill_seconds"] for row in rows),
                "total_decode_seconds": sum(row["decode_seconds"] for row in rows),
                "total_input_tokens": length * len(rows),
                "total_decode_tokens": (output_length - 1) * len(rows),
            }
        )
        summaries = [item["report"]["summary"] for item in items]
        for metric in (
            "measurement_peak_process_gib",
            "measurement_peak_device_gib",
            "lifetime_peak_process_gib",
        ):
            result[metric] = max(summary[metric] for summary in summaries)
        initializations = [summary["session_init_seconds"] for summary in summaries]
        if any(not math.isfinite(value) or value < 0 for value in initializations):
            raise ValueError("Invalid session initialization time")
        result["session_init_seconds_min"] = min(initializations)
        result["session_init_seconds_max"] = max(initializations)
        return result

    def group(items):
        result = {}
        for length in sorted({item["input_length"] for item in items}):
            sides = {
                runtime: aggregate(
                    [
                        item
                        for item in items
                        if item["input_length"] == length and item["runtime"] == runtime
                    ],
                    length,
                )
                for runtime in COMMITS
            }
            sides["speed_ratio_33569_over_33555"] = {
                metric: sides["33569"][metric] / sides["33555"][metric]
                for metric in ("prefill_tps", "decode_tps")
            }
            result[str(length)] = sides
        return result

    return {
        "aggregate": group(records),
        "per_round": {
            str(round_id): group(
                [item for item in records if item["round"] == round_id]
            )
            for round_id in sorted({item["round"] for item in records})
        },
        "interpretation": "Matched-input timing; finite logits differences are not equivalence.",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    for runtime in COMMITS:
        parser.add_argument(
            f"--runtime{runtime}",
            type=Path,
            default=Path(f"/tmp/int3-pr{runtime}-compare/python-runtime"),
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[128, 512, 2048, 4096]
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-length", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        plan_runs(args.lengths, args.rounds)
        benchmark.token_counts(128, args.output_length)
        if args.physical_gpu < 0 or args.warmups < 0 or args.repeats < 1:
            raise ValueError("Invalid GPU, warmups or repeats")
        for path in (args.model_dir, args.runtime33555, args.runtime33569):
            if not path.is_dir():
                raise ValueError(f"Required directory missing: {path}")
        if args.output_dir.exists() and (
            not args.output_dir.is_dir() or any(args.output_dir.iterdir())
        ):
            raise ValueError(
                "Output directory must be new or empty; resume is not supported"
            )
    except ValueError as error:
        parser.error(str(error))
    return args


def library_environment(runtime, physical_gpu):
    capi = runtime / "onnxruntime" / "capi"
    cuda = Path("/usr/local/cuda-13/lib64")
    if not (cuda / "libcudart.so").is_file():
        cuda = Path("/usr/local/cuda/lib64")
    candidates = [
        Path(value)
        for value in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if value
    ]
    candidates += [Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/local/cuda/lib64")]
    candidates += [Path(base) / "nvidia/cudnn/lib" for base in site.getsitepackages()]
    cudnn = [path.resolve() for path in candidates if list(path.glob("libcudnn.so*"))]
    if not capi.is_dir() or not (cuda / "libcudart.so").is_file() or not cudnn:
        raise ValueError("Runtime capi, CUDA 13 and cuDNN libraries must be available")
    env = {
        key: value
        for key, value in os.environ.items()
        if "QMOE" not in key.upper() and key != "ORT_DISABLE_MOE_GEMV"
    }
    env.update(
        {
            "PYTHONPATH": str(runtime),
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "LD_LIBRARY_PATH": os.pathsep.join(
                dict.fromkeys(str(path) for path in [capi, cuda, *cudnn])
            ),
            "PYTHONNOUSERSITE": "1",
        }
    )
    binaries = [
        benchmark.file_hash(path)
        for path in sorted(capi.glob("*.so*"))
        if path.is_file()
    ]
    if not any(
        Path(item["path"]).name == "libonnxruntime_providers_cuda.so"
        for item in binaries
    ):
        raise ValueError("Runtime package has no CUDA provider binary")
    return env, cuda / "libcudart.so", binaries


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory is not empty")
    manifest = {
        "status": "running",
        "python": sys.executable,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_commits": COMMITS,
        "artifact_commit": ARTIFACT_COMMIT,
        "commands": [],
        "expected_runtime_binaries": {},
    }
    records = []
    manifest_path = output / "manifest.json"

    def save_manifest():
        write_json(manifest_path, manifest)

    def execute(label, command, env):
        entry = {
            "label": label,
            "command": command,
            "environment": {
                key: env[key]
                for key in (
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONNOUSERSITE",
                )
            },
            "log": str(output / f"{label}.log"),
            "status": "running",
        }
        manifest["commands"].append(entry)
        save_manifest()
        with Path(entry["log"]).open("x") as log:
            try:
                subprocess.run(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    check=True,
                )
            except subprocess.CalledProcessError as error:
                entry.update(status="failed", returncode=error.returncode)
                save_manifest()
                raise
        entry.update(status="completed", returncode=0)
        save_manifest()
        return entry

    try:
        environments, cudarts, expected = {}, {}, {}
        artifacts = [
            benchmark.file_hash(path)
            for path in sorted(args.model_dir.iterdir())
            if path.is_file()
        ]
        gpu_uuid = subprocess.run(
            [
                "nvidia-smi",
                f"--id={args.physical_gpu}",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            start_new_session=True,
        ).stdout.strip()
        for runtime in COMMITS:
            root = getattr(args, f"runtime{runtime}").resolve()
            env, cudart, binaries = library_environment(root, args.physical_gpu)
            environments[runtime], cudarts[runtime] = env, cudart
            manifest["expected_runtime_binaries"][runtime] = binaries
            expected[runtime] = {
                "commit": COMMITS[runtime],
                "artifacts": artifact_identity(artifacts),
                "gpu_uuid": gpu_uuid,
                "binaries": {
                    item["path"]: (item["bytes"], item["sha256"]) for item in binaries
                },
            }
        manifest["artifact_files"] = artifacts
        manifest["gpu_uuid"] = gpu_uuid
        save_manifest()
        token_path = output / "reference-tokens.json"
        forced_tokens = None
        prompts = {}

        def measure(
            label, runtime, length, warmups, repeats, reference=False, capture=False
        ):
            raw_path = output / f"{label}.json"
            command = [
                sys.executable,
                str(HERE / "benchmark.py"),
                "--model-dir",
                str(args.model_dir.resolve()),
                "--input-length",
                str(length),
                "--output-length",
                str(args.output_length),
                "--warmups",
                str(warmups),
                "--repeats",
                str(repeats),
                "--physical-gpu",
                str(args.physical_gpu),
                "--artifact-commit",
                ARTIFACT_COMMIT,
                "--runtime-commit",
                COMMITS[runtime],
                "--cudart",
                str(cudarts[runtime]),
                "--output",
                str(raw_path),
            ]
            command += [
                "--write-token-stream" if reference else "--token-stream",
                str(token_path),
            ]
            if capture:
                command += ["--logits-output", str(output / f"{label}.npz")]
            execute(label, command, environments[runtime])
            report = json.loads(raw_path.read_text())
            verify_identity(
                report,
                expected[runtime],
                prompts.get(length),
                None if reference else forced_tokens,
            )
            if (
                len(report["metadata"]["prompt_ids"]) != length
                or len(report["trials"]) != repeats
            ):
                raise ValueError("Unexpected prompt or trial count")
            if report["summary"]["output_tokens"] != args.output_length:
                raise ValueError("Unexpected output token count")
            prompts[length] = report["metadata"]["prompt_sha256"]
            print(f"Completed {label}", flush=True)
            return report

        measure("reference33569", "33569", 128, 0, 1, reference=True, capture=True)
        forced_tokens = benchmark.validate_tokens(
            json.loads(token_path.read_text()), args.output_length
        )
        reference_report = json.loads((output / "reference33569.json").read_text())
        if forced_tokens != reference_report["trials"][0]["generated_tokens"]:
            raise ValueError(
                "Reference token file does not match natural greedy predictions"
            )
        measure("reference33555", "33555", 128, 0, 1, capture=True)
        comparison_path = output / "logits-comparison.json"
        command = [
            sys.executable,
            str(HERE / "compare_logits.py"),
            str(output / "reference33569.npz"),
            str(output / "reference33555.npz"),
            "--output",
            str(comparison_path),
        ]
        try:
            entry = execute("compare-logits", command, environments["33569"])
        except subprocess.CalledProcessError as error:
            if error.returncode != 1 or not comparison_path.is_file():
                raise
            comparison = validate_comparison(json.loads(comparison_path.read_text()))
            if comparison["allclose"] is not False:
                raise ValueError("Unexpected comparison exit status") from error
            entry = manifest["commands"][-1]
            entry["status"] = "completed-finite-differences"
        else:
            comparison = validate_comparison(json.loads(comparison_path.read_text()))
        manifest["logits_comparison"] = comparison
        save_manifest()
        print(
            f"Completed compare-logits: allclose={comparison['allclose']}, "
            f"max_abs={comparison['max_abs']}, top1={comparison['top1_agreement']}; "
            "not a claim of equivalence",
            flush=True,
        )
        for item in plan_runs(args.lengths, args.rounds):
            report = measure(
                item["label"],
                item["runtime"],
                item["input_length"],
                args.warmups,
                args.repeats,
            )
            records.append({**item, "report": report})
        write_json(output / "summary.json", summarize_runs(records, args.output_length))
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["result_files"] = [
            benchmark.file_hash(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path != manifest_path
        ]
        save_manifest()


if __name__ == "__main__":
    run(parse_args())
