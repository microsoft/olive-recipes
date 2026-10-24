import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).with_name("benchmark.py")
SPEC = importlib.util.spec_from_file_location("standalone_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class FakeTokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(
            ids={
                benchmark.PREFIX: [1, 2],
                benchmark.SUFFIX: [8, 9, 10],
                benchmark.CONTEXT: [3, 4, 5],
            }[text]
        )


def test_exact_prompt_template():
    assert benchmark.PREFIX == "<|im_start|>user\n"
    assert benchmark.SUFFIX == (
        "Summarize the context in detail. /no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert (
        benchmark.CONTEXT
        == "Paris is the capital of France. Its museums and public parks attract visitors. "
    )
    assert benchmark.build_prompt(FakeTokenizer(), 10) == [
        1,
        2,
        3,
        4,
        5,
        3,
        4,
        8,
        9,
        10,
    ]
    assert benchmark.build_prompt(FakeTokenizer(), 5) == [1, 2, 8, 9, 10]
    with pytest.raises(ValueError):
        benchmark.build_prompt(FakeTokenizer(), 4)


@pytest.mark.parametrize(
    "tokens", [None, {}, [0], [0, True], [0, -1], [0, 1.0], [0, 10]]
)
def test_invalid_tokens(tokens):
    with pytest.raises(ValueError):
        benchmark.validate_tokens(tokens, 2, 10)


def test_forced_and_natural_decode():
    stream = benchmark.validate_tokens([2, 3, 4, 5], 4, 6)
    assert [benchmark.decode_input(step, 99, stream) for step in range(1, 4)] == [
        2,
        3,
        4,
    ]
    assert [
        benchmark.decode_input(step, prediction, None)
        for step, prediction in enumerate([9, 8, 7], 1)
    ] == [9, 8, 7]
    with pytest.raises(ValueError):
        benchmark.decode_input(0, 99, stream)
    assert benchmark.token_hash(stream) == benchmark.token_hash(list(stream))
    assert benchmark.token_hash(stream) != benchmark.token_hash([2, 3, 5, 4])


def test_token_counts_and_masks():
    counts = benchmark.token_counts(128, 4)
    assert counts == {
        "input_tokens": 128,
        "output_tokens": 4,
        "decode_steps": 3,
        "final_cache_tokens": 131,
    }
    assert [128 + step for step in range(1, 4)][-1] == counts["final_cache_tokens"]
    for lengths in [(0, 2), (128, 1)]:
        with pytest.raises(ValueError):
            benchmark.token_counts(*lengths)


def test_metrics():
    rows = [
        {
            "ttft_ms": value * 1000,
            "prefill_seconds": value,
            "decode_seconds": value * 2,
            "decode_run_seconds": [value / 2, value / 2],
        }
        for value in (1, 3)
    ]
    summary = benchmark.summarize(rows, 128, 3)
    assert summary["ttft_ms_p50"] == 2000
    assert summary["ttft_ms_p95"] == 2900
    assert summary["prefill_tps"] == 64
    assert summary["decode_tps"] == 0.5
    assert summary["decode_ort_run_p50_ms"] == 1000
    with pytest.raises(ValueError):
        benchmark.summarize([], 128, 3)
    rows[0]["decode_seconds"] = 0
    with pytest.raises(ValueError):
        benchmark.summarize(rows, 128, 3)


@pytest.mark.parametrize("values,percent", [([], 50), ([float("nan")], 50), ([1], 101)])
def test_invalid_percentile(values, percent):
    with pytest.raises(ValueError):
        benchmark.percentile(values, percent)


def test_clean_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("ORT_ENABLE_QMOE_INT2_GEMV", "1")
    monkeypatch.setenv("ORT_QMOE_INT_DEQUANT_MAX_SCRATCH_BYTES", "1")
    monkeypatch.setenv("ORT_DISABLE_MOE_GEMV", "1")
    removed = benchmark.clean_environment(2)
    assert set(removed) >= {
        "ORT_ENABLE_QMOE_INT2_GEMV",
        "ORT_QMOE_INT_DEQUANT_MAX_SCRATCH_BYTES",
        "ORT_DISABLE_MOE_GEMV",
    }
    with pytest.raises(ValueError):
        benchmark.clean_environment(0)


def test_import_is_dependency_free():
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('benchmark', {str(SCRIPT)!r}); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "assert not {'onnxruntime', 'pynvml', 'numpy', 'tokenizers'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-S", "-c", code], check=True)


def test_parser_defaults_and_required_labels(tmp_path):
    argv = [
        "--model-dir",
        "/model",
        "--input-length",
        "128",
        "--output",
        str(tmp_path / "result.json"),
        "--artifact-commit",
        "artifact-sha",
        "--runtime-commit",
        "runtime-sha",
    ]
    args = benchmark.parse_args(argv)
    assert (args.output_length, args.warmups, args.repeats) == (128, 3, 10)
    assert args.int_dequant_max_scratch_bytes == 67108864
    assert str(args.cudart) == "/usr/local/cuda/lib64/libcudart.so"
    with pytest.raises(SystemExit):
        benchmark.parse_args(argv[:-2])
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            argv + ["--token-stream", "/tokens", "--write-token-stream", "/other"]
        )


def test_parser_rejects_output_collisions(tmp_path):
    output = tmp_path / "result.json"
    argv = [
        "--model-dir",
        "/model",
        "--input-length",
        "128",
        "--output",
        str(output),
        "--artifact-commit",
        "artifact-sha",
        "--runtime-commit",
        "runtime-sha",
    ]
    with pytest.raises(SystemExit):
        benchmark.parse_args(argv + ["--logits-output", str(output)])
    output.touch()
    with pytest.raises(SystemExit):
        benchmark.parse_args(argv)
