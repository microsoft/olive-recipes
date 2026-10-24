import json

import numpy as np
import pytest

import compare_logits


def save_logits(tmp_path, name, values, tokens=None):
    path = tmp_path / name
    logits = np.asarray(values, dtype=np.float16)
    if tokens is None:
        tokens = np.argmax(logits, axis=-1)
    np.savez(path, logits=logits, generated_tokens=np.asarray(tokens))
    return path


def test_exact_and_predicted_steps(tmp_path):
    path = save_logits(tmp_path, "reference.npz", [[1, 2], [4, 3], [2, 1]])
    report = compare_logits.compare(path, path)
    assert report["allclose"] and report["finite"] and report["shape_equal"]
    assert report["max_abs"] == report["mean_abs"] == report["rmse"] == 0
    assert report["reference"]["predicted_steps"] == 3
    assert report["reference"]["decode_steps"] == 2
    assert report["top1_agreement_count"] == 3
    assert report["top1_mismatch_indices"] == []


def test_numeric_difference_and_top1(tmp_path):
    reference = save_logits(tmp_path, "reference.npz", [[1, 2], [3, 1]])
    candidate = save_logits(tmp_path, "candidate.npz", [[2, 1], [5, 1]])
    report = compare_logits.compare(reference, candidate)
    assert not report["allclose"]
    assert report["max_abs"] == 2
    assert report["mean_abs"] == 1
    assert report["rmse"] == pytest.approx(np.sqrt(1.5))
    assert report["top1_agreement"] == 0.5
    assert report["top1_agreement_count"] == report["top1_mismatch_count"] == 1
    assert report["top1_mismatch_indices"] == [0]
    assert report["top1_reference"] == [1, 0]
    assert "isolated-layer" in report["scope"]


def test_default_tolerance_and_float16_no_square_overflow(tmp_path):
    reference = save_logits(tmp_path, "reference.npz", [[0, 1]])
    candidate = save_logits(tmp_path, "near.npz", [[0.005, 1]])
    assert compare_logits.compare(reference, candidate)["allclose"]
    far = save_logits(tmp_path, "far.npz", [[1000, 1]])
    assert compare_logits.compare(reference, far)["rmse"] == pytest.approx(
        np.sqrt(500000)
    )


@pytest.mark.parametrize("values", [[[1, 2, 3]], [[1, 2], [2, 1]]])
def test_shape_mismatch_report(tmp_path, values):
    reference = save_logits(tmp_path, "reference.npz", [[1, 2]])
    candidate = save_logits(tmp_path, "candidate.npz", values)
    report = compare_logits.compare(reference, candidate)
    assert report["shape_equal"] is False
    assert report["allclose"] is False
    assert report["rmse"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_json(tmp_path, bad):
    reference = save_logits(tmp_path, "reference.npz", [[1, 2]])
    candidate = save_logits(tmp_path, "candidate.npz", [[bad, 2]])
    report = compare_logits.compare(reference, candidate)
    assert not report["finite"] and not report["allclose"]
    assert report["candidate"]["nonfinite_count"] == 1
    assert report["top1_agreement"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("atol,rtol", [(-1, 0), (0, -1), (np.nan, 0), (0, np.inf)])
def test_bad_tolerances(tmp_path, atol, rtol):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_logits.compare(tmp_path / "missing", tmp_path / "missing", atol, rtol)


@pytest.mark.parametrize(
    "defect", ["missing", "rank", "empty", "integer", "tokens", "token_count", "object"]
)
def test_malformed_npz(tmp_path, defect):
    path = tmp_path / "bad.npz"
    payload = {
        "logits": np.asarray([[1, 2]], dtype=np.float16),
        "generated_tokens": np.asarray([1]),
    }
    if defect == "missing":
        del payload["logits"]
    elif defect == "rank":
        payload["logits"] = np.asarray([1, 2], dtype=np.float16)
    elif defect == "empty":
        payload["logits"] = np.empty((0, 2), dtype=np.float16)
    elif defect == "integer":
        payload["logits"] = np.asarray([[1, 2]])
    elif defect == "tokens":
        payload["generated_tokens"] = np.asarray([0])
    elif defect == "token_count":
        payload["generated_tokens"] = np.asarray([1, 1])
    else:
        payload["logits"] = np.asarray([[object()]], dtype=object)
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        compare_logits.load_logits(path)


def test_cli_output_and_failure_status(tmp_path, capsys):
    reference = save_logits(tmp_path, "reference.npz", [[1, 2]])
    candidate = save_logits(tmp_path, "candidate.npz", [[2, 1]])
    output = tmp_path / "report.json"
    assert (
        compare_logits.main([str(reference), str(candidate), "--output", str(output)])
        == 1
    )
    assert json.loads(output.read_text())["allclose"] is False
    assert json.loads(capsys.readouterr().out)["allclose"] is False
    assert compare_logits.main([str(reference), str(reference)]) == 0
    assert json.loads(capsys.readouterr().out)["allclose"] is True


def test_invalid_archive_cli(tmp_path, capsys):
    path = tmp_path / "invalid.npz"
    path.write_bytes(b"invalid")
    assert compare_logits.main([str(path), str(path)]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
