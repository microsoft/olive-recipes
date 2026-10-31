# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Mocked CPU regressions for reproducibility boundaries; never call GPU APIs."""
from __future__ import annotations

import builtins
import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evidence  # noqa: E402

UUID_A = "GPU-12345678-1234-1234-1234-123456789abc"
UUID_B = "GPU-87654321-4321-4321-4321-cba987654321"


def load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/run_ort_llama_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner("reproducibility_runner")


class ProviderTests(unittest.TestCase):
    def versions(self, legacy: str | None):
        def version(name):
            if name == "pynvml":
                if legacy is None:
                    raise importlib.metadata.PackageNotFoundError(name)
                return legacy
            return "required-provider-present"
        return version

    def test_declared_provider_present_legacy_absent_is_explicitly_optional(self):
        with patch.object(RUNNER.importlib.metadata, "version", side_effect=self.versions(None)):
            result = RUNNER.collect_package_versions()
        self.assertEqual(result["nvidia-ml-py"], "required-provider-present")
        self.assertIsNone(result["pynvml"])

    def test_legacy_metadata_is_recorded_when_present(self):
        with patch.object(RUNNER.importlib.metadata, "version", side_effect=self.versions("13.0.1")):
            result = RUNNER.collect_package_versions()
        self.assertEqual(result["pynvml"], "13.0.1")

    def test_missing_required_provider_still_raises(self):
        def missing(name):
            if name == "nvidia-ml-py":
                raise importlib.metadata.PackageNotFoundError(name)
            return "present"
        with patch.object(RUNNER.importlib.metadata, "version", side_effect=missing):
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                RUNNER.collect_package_versions()

    def test_non_missing_metadata_error_is_not_suppressed(self):
        def invalid(name):
            if name == "pynvml":
                raise OSError("invalid package metadata")
            return "present"
        with patch.object(RUNNER.importlib.metadata, "version", side_effect=invalid):
            with self.assertRaisesRegex(OSError, "invalid package metadata"):
                RUNNER.collect_package_versions()


class IdentityTests(unittest.TestCase):
    def fake_nvml(self, uuid=UUID_A):
        nvml = Mock()
        nvml.nvmlDeviceGetUUID.return_value = uuid
        return nvml

    def test_unset_numeric_and_matching_uuid_visibility_are_canonicalized(self):
        for visibility in (None, "0", UUID_A):
            with self.subTest(visibility=visibility):
                environment = {"BENCHMARK_GPU_INDEX": "0"}
                if visibility is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = visibility
                nvml = self.fake_nvml(UUID_A.encode())
                with patch.dict(os.environ, environment, clear=True), patch.object(
                    RUNNER.sys, "platform", "linux"
                ), patch.object(RUNNER, "nvml_device", return_value=nullcontext((nvml, object()))):
                    RUNNER.gpu_zero_environment()
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], UUID_A)
                    self.assertEqual(os.environ[RUNNER.GPU_UUID_ENV], UUID_A)
                    self.assertEqual(os.environ["BENCHMARK_GPU_INDEX"], "0")

    def test_conflicting_and_multiple_device_visibility_is_rejected(self):
        for visibility in (UUID_B, "1", "0,1", "", UUID_A + "," + UUID_B):
            with self.subTest(visibility=visibility), patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": visibility}, clear=True
            ), patch.object(RUNNER.sys, "platform", "linux"), patch.object(
                RUNNER, "nvml_device", return_value=nullcontext((self.fake_nvml(), object()))
            ):
                with self.assertRaises(RuntimeError):
                    RUNNER.gpu_zero_environment()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], visibility)

    def test_physical_gpu_index_other_than_zero_is_rejected(self):
        with patch.dict(os.environ, {"BENCHMARK_GPU_INDEX": "1"}, clear=True), patch.object(
            RUNNER.sys, "platform", "linux"
        ), patch.object(RUNNER, "nvml_device") as nvml:
            with self.assertRaisesRegex(RuntimeError, "must be 0"):
                RUNNER.gpu_zero_environment()
            nvml.assert_not_called()

    def test_different_enumeration_orders_use_uuid_not_ordinal_assumption(self):
        # In this fixture CUDA ordinal 0 would originally be B; NVML index 0 is A.
        cuda_order, nvml_order = [UUID_B, UUID_A], [UUID_A, UUID_B]
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear=True), patch.object(
            RUNNER.sys, "platform", "linux"
        ), patch.object(RUNNER, "nvml_device", return_value=nullcontext((self.fake_nvml(nvml_order[0]), object()))):
            RUNNER.gpu_zero_environment()
            selected = next(value for value in cuda_order if value == os.environ["CUDA_VISIBLE_DEVICES"])
            with patch.object(RUNNER, "cuda_driver_gpu_uuid", return_value=selected):
                self.assertEqual(RUNNER.validate_execution_gpu_identity(), UUID_A)
            self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "FASTEST_FIRST")

    def test_execution_uuid_mismatch_is_rejected(self):
        with patch.dict(os.environ, {RUNNER.GPU_UUID_ENV: UUID_A, "CUDA_VISIBLE_DEVICES": UUID_A}, clear=True), patch.object(
            RUNNER, "cuda_driver_gpu_uuid", return_value=UUID_B
        ):
            with self.assertRaisesRegex(RuntimeError, "differs from NVML"):
                RUNNER.validate_execution_gpu_identity()

    def test_monitoring_uuid_mismatch_is_rejected_and_nvml_shuts_down(self):
        nvml = self.fake_nvml(UUID_B)
        with patch.dict(os.environ, {RUNNER.GPU_UUID_ENV: UUID_A}, clear=True), patch.dict(
            sys.modules, {"pynvml": nvml}
        ):
            with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                with RUNNER.nvml_device():
                    self.fail("Mismatched NVML handle must not be yielded")
        nvml.nvmlShutdown.assert_called_once()
        nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)

    def test_empty_preinit_and_cleanup_process_lists_remain_valid(self):
        RUNNER.check_gpu_processes({"gpu_process_pids": []}, set())
        RUNNER.check_gpu_processes({"gpu_process_pids": []}, {123})
        RUNNER.check_gpu_processes({"gpu_process_pids": [123]}, {123})
        with self.assertRaisesRegex(RuntimeError, "unrelated"):
            RUNNER.check_gpu_processes({"gpu_process_pids": [456]}, {123})

    def test_driver_query_uses_identity_apis_without_context_or_memory_calls(self):
        def count(pointer):
            pointer._obj.value = 1
            return 0
        def device(pointer, ordinal):
            self.assertEqual(ordinal, 0)
            pointer._obj.value = 0
            return 0
        def identifier(pointer, ordinal):
            self.assertEqual(ordinal, 0)
            pointer._obj.data[:] = UUID(UUID_A[4:]).bytes
            return 0
        driver = types.SimpleNamespace(
            cuInit=Mock(return_value=0), cuDeviceGetCount=Mock(side_effect=count),
            cuDeviceGet=Mock(side_effect=device), cuDeviceGetUuid_v2=Mock(side_effect=identifier),
        )
        with patch.object(RUNNER.sys, "platform", "linux"), patch.object(
            RUNNER.ctypes.util, "find_library", return_value="mock-cuda-driver"
        ), patch.object(RUNNER.ctypes, "CDLL", return_value=driver):
            self.assertEqual(RUNNER.cuda_driver_gpu_uuid(), UUID_A)
        driver.cuInit.assert_called_once_with(0)
        driver.cuDeviceGetCount.assert_called_once()
        driver.cuDeviceGet.assert_called_once()
        driver.cuDeviceGetUuid_v2.assert_called_once()

    def test_driver_query_rejects_more_than_one_visible_device(self):
        def count(pointer):
            pointer._obj.value = 2
            return 0
        driver = types.SimpleNamespace(
            cuInit=Mock(return_value=0), cuDeviceGetCount=Mock(side_effect=count),
            cuDeviceGet=Mock(return_value=0), cuDeviceGetUuid_v2=Mock(return_value=0),
        )
        with patch.object(RUNNER.sys, "platform", "linux"), patch.object(
            RUNNER.ctypes.util, "find_library", return_value="mock-cuda-driver"
        ), patch.object(RUNNER.ctypes, "CDLL", return_value=driver):
            with self.assertRaisesRegex(RuntimeError, "Expected one"):
                RUNNER.cuda_driver_gpu_uuid()
        driver.cuDeviceGet.assert_not_called()

    def test_driver_error_is_explicit_and_stops_later_queries(self):
        driver = types.SimpleNamespace(
            cuInit=Mock(return_value=100), cuDeviceGetCount=Mock(return_value=0),
            cuDeviceGet=Mock(return_value=0), cuDeviceGetUuid=Mock(return_value=0),
        )
        with patch.object(RUNNER.sys, "platform", "linux"), patch.object(
            RUNNER.ctypes.util, "find_library", return_value="mock-cuda-driver"
        ), patch.object(RUNNER.ctypes, "CDLL", return_value=driver):
            with self.assertRaisesRegex(RuntimeError, "cuInit failed with status 100"):
                RUNNER.cuda_driver_gpu_uuid()
        driver.cuDeviceGetCount.assert_not_called()

    def test_worker_and_sampler_revalidate_inherited_matching_uuid(self):
        # The shared resolver is used by launcher, trial worker and sampler entry paths.
        nvml = self.fake_nvml(UUID_A)
        with patch.dict(os.environ, {
            "BENCHMARK_GPU_INDEX": "0", "CUDA_VISIBLE_DEVICES": UUID_A, RUNNER.GPU_UUID_ENV: UUID_A,
        }, clear=True), patch.object(RUNNER.sys, "platform", "linux"), patch.dict(
            sys.modules, {"pynvml": nvml}
        ):
            RUNNER.gpu_zero_environment()
            with RUNNER.nvml_device() as (_, handle):
                self.assertIs(handle, nvml.nvmlDeviceGetHandleByIndex.return_value)
            self.assertEqual(os.environ[RUNNER.GPU_UUID_ENV], UUID_A)
        self.assertTrue(all(call.args == (0,) for call in nvml.nvmlDeviceGetHandleByIndex.call_args_list))

    def test_worker_identity_failure_precedes_sampler_and_runtime(self):
        trial = RUNNER.Trial("ort", "device", 8192, 1, 64, 8448, "unused", "unused-model",
                             "unused-gguf", "unused-output", "0" * 64, 10)
        request = json.dumps(RUNNER.asdict(trial))
        with patch.object(RUNNER, "gpu_zero_environment"), patch.object(
            RUNNER.Path, "read_text", return_value=request
        ), patch.object(RUNNER, "config_digest", return_value="0" * 64), patch.object(
            RUNNER, "validate_execution_gpu_identity", side_effect=RuntimeError("identity mismatch")
        ), patch.object(RUNNER, "ResourceSampler") as sampler, patch.object(RUNNER, "run_ort") as runtime:
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                RUNNER.trial_worker(Path("unused-request"))
        sampler.assert_not_called()
        runtime.assert_not_called()


class PlatformTests(unittest.TestCase):
    def test_cpu_import_and_help_do_not_need_unix_or_gpu_modules(self):
        real_import = builtins.__import__
        def portable_import(name, *args, **kwargs):
            if name in {"fcntl", "pynvml", "onnxruntime_genai", "llama_cpp"}:
                raise ModuleNotFoundError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=portable_import):
            module = load_runner("portable_cpu_runner")
            self.assertEqual(module.requested_sequence_capacity(8192, 64), 8448)
            module.validate_input_capacity(range(7189), 8448, 64)
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
                module.main(["--help"])
            self.assertEqual(exit.exception.code, 0)

    def test_unsupported_gpu_execution_fails_before_unix_operations(self):
        with patch.object(RUNNER.sys, "platform", "win32"), patch.object(
            RUNNER.os, "getuid", create=True
        ) as getuid, patch.object(RUNNER, "nvml_device") as nvml:
            with self.assertRaisesRegex(RuntimeError, "requires Linux"):
                with RUNNER.gpu_run_lock():
                    self.fail("Linux-only execution must not enter")
            with self.assertRaisesRegex(RuntimeError, "requires Linux"):
                RUNNER.gpu_zero_environment()
        getuid.assert_not_called()
        nvml.assert_not_called()

    def test_linux_locking_preserves_exclusive_nonblocking_and_busy_failure(self):
        fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=Mock())
        with tempfile.TemporaryDirectory(prefix="qwen-lock-cpu-") as directory, patch.object(
            RUNNER.sys, "platform", "linux"
        ), patch.object(RUNNER.tempfile, "gettempdir", return_value=directory), patch.object(
            RUNNER.os, "getuid", return_value=123, create=True
        ), patch.dict(sys.modules, {"fcntl": fcntl}):
            with RUNNER.gpu_run_lock():
                pass
            self.assertEqual(fcntl.flock.call_args.args[1], 3)
            fcntl.flock.side_effect = BlockingIOError()
            with self.assertRaisesRegex(RuntimeError, "holds the GPU 0"):
                with RUNNER.gpu_run_lock():
                    pass


class ProvenanceTests(unittest.TestCase):
    def test_publication_lineage_matches_runner_and_retains_historical_identities(self):
        lineage = evidence.read_json(ROOT / "scripts/runner_lineage.json")
        historical = evidence.read_json(evidence.EVIDENCE / "provenance.json")["accepted"]
        runner = ROOT / "scripts/run_ort_llama_benchmark.py"
        self.assertEqual(hashlib.sha256(runner.read_bytes()).hexdigest(), lineage["revised_publication_runner_sha256"])
        self.assertEqual(lineage["historical_reference"]["runner_sha256"], historical["measurement_runner_sha256"])
        self.assertEqual(lineage["historical_reference"]["recorded_whole_measurement_ast_sha256"], historical["measurement_ast_sha256"])
        definitions = {node.name: node for node in ast.parse(runner.read_bytes()).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name, expected in lineage["unchanged_named_algorithm_ast_sha256"].items():
            self.assertEqual(hashlib.sha256(ast.dump(definitions[name], include_attributes=False).encode()).hexdigest(), expected)
        self.assertIn("trial_worker", lineage["changed_existing_definitions"])
        self.assertNotIn("measure_tokens", lineage["changed_existing_definitions"])

    def test_checksum_consistent_single_condition_mutations_are_rejected(self):
        mutations = [
            ("record_count", ("record_count",), 134),
            ("independent_workers", ("independent_workers",), 44),
            ("profiled", ("profiled",), True),
            ("gpu_index", ("gpu_index",), 1),
            ("benchmark_exit_code", ("benchmark_exit_status", "benchmark_exit_code"), 1),
            ("tee_exit_code", ("benchmark_exit_status", "tee_exit_code"), 1),
            ("launcher_exit_code", ("benchmark_exit_status", "launcher_exit_code"), 1),
            ("context_labels", ("matrix", "context_labels"), [8192]),
            ("variants", ("matrix", "variants"), [["ort", "device"]]),
            ("worker_repetitions", ("matrix", "worker_repetitions"), 2),
            ("requests_per_worker", ("matrix", "requests_per_worker"), 2),
            ("expected_workers", ("matrix", "expected_workers"), 44),
            ("expected_request_records", ("matrix", "expected_request_records"), 134),
            ("max_generated_ids", ("matrix", "max_generated_ids"), 63),
            ("requested_capacities", ("matrix", "requested_capacities", "8192"), 8192),
        ]
        original = evidence.read_json(evidence.EVIDENCE / "provenance.json")
        with tempfile.TemporaryDirectory(prefix="qwen-provenance-cpu-") as directory:
            copy_dir = Path(directory) / "evidence"
            shutil.copytree(evidence.EVIDENCE, copy_dir)
            for message, keys, value in mutations:
                with self.subTest(condition=message):
                    mutated = copy.deepcopy(original)
                    target = mutated["accepted"]
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = value
                    path = copy_dir / "provenance.json"
                    path.write_text(json.dumps(mutated, indent=2) + "\n", encoding="utf-8")
                    # Only disposable copied evidence gets a new checksum. Historical files are untouched.
                    entries = [
                        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
                        for p in sorted(copy_dir.iterdir()) if p.is_file() and p.name != "SHA256SUMS"
                    ]
                    (copy_dir / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
                    evidence.verify_checksums(copy_dir)
                    with self.assertRaisesRegex(ValueError, "Accepted provenance"):
                        evidence.calculate(copy_dir)

    def test_missing_required_provenance_is_rejected_explicitly(self):
        records = evidence.load_requests(evidence.EVIDENCE)
        original = evidence.read_json(evidence.EVIDENCE / "provenance.json")
        for keys in (("accepted",), ("accepted", "matrix"), ("accepted", "gpu_index"),
                     ("accepted", "matrix", "requested_capacities"), ("accepted", "benchmark_exit_status")):
            with self.subTest(keys=keys):
                mutated = copy.deepcopy(original)
                target = mutated
                for key in keys[:-1]:
                    target = target[key]
                del target[keys[-1]]
                with self.assertRaisesRegex(ValueError, "Accepted provenance"):
                    evidence.validate_accepted_provenance(mutated, records)

    def test_provenance_requires_actual_integer_and_boolean_types(self):
        records = evidence.load_requests(evidence.EVIDENCE)
        for key, value in (("gpu_index", False), ("record_count", "135"), ("profiled", "false")):
            with self.subTest(key=key):
                mutated = copy.deepcopy(evidence.read_json(evidence.EVIDENCE / "provenance.json"))
                mutated["accepted"][key] = value
                with self.assertRaisesRegex(ValueError, "Accepted provenance"):
                    evidence.validate_accepted_provenance(mutated, records)


class GitLineEndingTests(unittest.TestCase):
    @staticmethod
    def pinned_paths():
        """Model-relative paths whose raw bytes are pinned, with their expected SHA-256."""
        manifest = evidence.EVIDENCE / "SHA256SUMS"
        pinned = []
        for line in manifest.read_text().splitlines():
            expected, name = line.split("  ", 1)
            pinned.append(("evidence/" + name, expected))
        pinned.append(("evidence/SHA256SUMS", hashlib.sha256(manifest.read_bytes()).hexdigest()))
        lineage = evidence.read_json(ROOT / "scripts/runner_lineage.json")
        pinned.append(("scripts/run_ort_llama_benchmark.py", lineage["revised_publication_runner_sha256"]))
        return pinned

    @unittest.skipUnless(shutil.which("git"), "Git is required only for checkout/filter regression")
    def test_autocrlf_checkout_filter_preserves_all_pinned_files(self):
        pinned = self.pinned_paths()
        self.assertEqual(len({path for path, _ in pinned}), len(pinned))
        # A disposable repository keeps this hermetic: no normal index/objects and no enclosing .git needed.
        with tempfile.TemporaryDirectory(prefix="qwen-git-filter-cpu-") as directory:
            temporary = Path(directory)
            subprocess.run(["git", "init", "-q", str(temporary)], check=True, capture_output=True)
            (temporary / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
            for path, expected in pinned:
                with self.subTest(path=path):
                    data = (ROOT / path).read_bytes()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
                    oid = subprocess.run(["git", "-C", str(temporary), "hash-object", "-w", "--stdin"],
                                         input=data, capture_output=True, check=True).stdout.decode().strip()
                    filtered = subprocess.run(
                        ["git", "-C", str(temporary), "-c", "core.autocrlf=true", "cat-file", "--filters",
                         "--path=" + path, oid], capture_output=True, check=True
                    ).stdout
                    self.assertEqual(filtered, data)
                    self.assertEqual(hashlib.sha256(filtered).hexdigest(), expected)
                    attrs = subprocess.run(
                        ["git", "-C", str(temporary), "check-attr", "text", "eol", "--", path],
                        capture_output=True, check=True, text=True).stdout
                    self.assertIn("text: set", attrs)
                    self.assertIn("eol: lf", attrs)


if __name__ == "__main__":
    unittest.main()
