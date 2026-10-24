import copy
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "comparison_driver", Path(__file__).with_name("run_comparison.py")
)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_plan_alternates_and_defaults():
    plan = driver.plan_runs([128, 512, 2048, 4096], 2)
    assert len(plan) == 16
    assert len({item["label"] for item in plan}) == 16
    assert [item["runtime"] for item in plan] == [
        "33555",
        "33569",
        "33569",
        "33555",
        "33555",
        "33569",
        "33569",
        "33555",
        "33569",
        "33555",
        "33555",
        "33569",
        "33569",
        "33555",
        "33555",
        "33569",
    ]
    assert [item["input_length"] for item in plan[:8]] == [
        128,
        128,
        512,
        512,
        2048,
        2048,
        4096,
        4096,
    ]
    for lengths, rounds in (([], 1), ([128, 128], 1), ([0], 1), ([128], 0)):
        with pytest.raises(ValueError):
            driver.plan_runs(lengths, rounds)


def make_report(value):
    return {
        "trials": [
            {
                "ttft_ms": value * 1000,
                "prefill_seconds": value,
                "decode_seconds": value * 2,
                "decode_run_seconds": [value, value],
            }
        ],
        "summary": {
            "session_init_seconds": value,
            "measurement_peak_process_gib": value + 1,
            "measurement_peak_device_gib": value + 2,
            "lifetime_peak_process_gib": value + 3,
        },
    }


def test_summary_uses_raw_trials_and_rounds():
    records = [
        {
            **item,
            "report": make_report(
                item["round"] * (2 if item["runtime"] == "33555" else 1)
            ),
        }
        for item in driver.plan_runs([128], 2)
    ]
    result = driver.summarize_runs(records, 3)
    slower = result["aggregate"]["128"]["33555"]
    assert slower["ttft_ms_p50"] == 3000
    assert slower["ttft_ms_p95"] == 3900
    assert slower["prefill_tps"] == pytest.approx(256 / 6)
    assert slower["decode_tps"] == pytest.approx(4 / 12)
    assert slower["total_input_tokens"] == 256
    assert slower["total_decode_tokens"] == 4
    assert slower["total_prefill_seconds"] == 6
    assert slower["session_init_seconds_min"] == 2
    assert slower["session_init_seconds_max"] == 4
    assert slower["measurement_peak_device_gib"] == 6
    assert result["aggregate"]["128"]["speed_ratio_33569_over_33555"] == {
        "prefill_tps": 2,
        "decode_tps": 2,
    }
    assert result["per_round"]["1"]["128"]["33555"]["trials"] == 1


def identity_fixture():
    binary = {
        "path": "/runtime/onnxruntime/capi/libonnxruntime_providers_cuda.so",
        "bytes": 4,
        "sha256": "binary-hash",
    }
    files = [{"path": "/model/model.onnx", "bytes": 8, "sha256": "full-artifact-hash"}]
    metadata = {
        "artifact_commit": driver.ARTIFACT_COMMIT,
        "runtime_commit": driver.COMMITS["33555"],
        "artifact_files": files,
        "gpu_uuid": "GPU-test",
        "providers": ["CUDAExecutionProvider"],
        "runtime_binaries": [binary],
        "prompt_ids": [1, 2],
        "prompt_sha256": driver.benchmark.token_hash([1, 2]),
        "forced_tokens": [3, 4],
        "forced_token_sha256": driver.benchmark.token_hash([3, 4]),
    }
    expected = {
        "commit": driver.COMMITS["33555"],
        "artifacts": driver.artifact_identity(files),
        "gpu_uuid": "GPU-test",
        "binaries": {binary["path"]: (4, "binary-hash")},
    }
    return {"metadata": metadata}, expected


def test_identity_accepts_matching_pair():
    report, expected = identity_fixture()
    driver.verify_identity(
        report, expected, driver.benchmark.token_hash([1, 2]), [3, 4]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_uuid", "GPU-other"),
        ("prompt_sha256", "other"),
        ("forced_tokens", [4, 3]),
        ("forced_token_sha256", "other"),
        ("runtime_commit", "wrong"),
        ("artifact_commit", "wrong"),
        ("providers", ["CPUExecutionProvider"]),
    ],
)
def test_identity_rejects_mismatch(field, value):
    report, expected = identity_fixture()
    report["metadata"][field] = value
    with pytest.raises(ValueError):
        driver.verify_identity(
            report, expected, driver.benchmark.token_hash([1, 2]), [3, 4]
        )


@pytest.mark.parametrize("field", ["runtime_binaries", "artifact_files"])
def test_identity_rejects_full_hash_changes(field):
    report, expected = identity_fixture()
    report = copy.deepcopy(report)
    report["metadata"][field][0]["sha256"] = "different"
    with pytest.raises(ValueError):
        driver.verify_identity(report, expected, forced_tokens=[3, 4])


def test_finite_differences_permitted_not_equivalence():
    report = {
        "shape_equal": True,
        "finite": True,
        "allclose": False,
        "max_abs": 0.225,
        "mean_abs": 0.01,
        "rmse": 0.02,
        "top1_agreement": 1.0,
    }
    assert driver.validate_comparison(report)["allclose"] is False
    for changes in (
        {"finite": False},
        {"shape_equal": False},
        {"error": "invalid archive"},
        {"max_abs": float("nan")},
        {"rmse": None},
    ):
        with pytest.raises(ValueError):
            driver.validate_comparison({**report, **changes})


def test_parser_defaults_and_rejects_nonempty_output(tmp_path):
    argv = [
        "--model-dir",
        str(tmp_path),
        "--runtime33555",
        str(tmp_path),
        "--runtime33569",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    args = driver.parse_args(argv)
    assert (
        args.lengths,
        args.rounds,
        args.warmups,
        args.repeats,
        args.output_length,
    ) == (
        [128, 512, 2048, 4096],
        2,
        3,
        5,
        128,
    )
    assert args.physical_gpu == 0
    args.output_dir.mkdir()
    (args.output_dir / "failed.json").touch()
    with pytest.raises(SystemExit):
        driver.parse_args(argv)
