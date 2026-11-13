"""Offline Olive contracts; no model payloads, real exports or CUDA execution."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from lightning_builder import MODEL_ID, MODEL_REVISION, NemotronLightningBuilder
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.package_config import OlivePackageConfig
from olive.passes.olive_pass import create_pass_from_dict
from olive.workflows.run.config import RunConfig

DIRECTORY = Path(__file__).parent
CUDA = AcceleratorSpec(Device.GPU, "CUDAExecutionProvider")


def model(**load_kwargs):
    return HfModelHandler(
        model_path=MODEL_ID,
        task="text-generation",
        load_kwargs={"revision": MODEL_REVISION, "trust_remote_code": False, **load_kwargs},
    )


def exporter(accelerator=CUDA, **config):
    return create_pass_from_dict(
        NemotronLightningBuilder,
        {"allow_unvalidated_export": True, **config},
        disable_search=True,
        accelerator_spec=accelerator,
    )


@pytest.fixture(autouse=True)
def offline_metadata(monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.repo_info",
        lambda *args, **kwargs: types.SimpleNamespace(sha=MODEL_REVISION),
    )
    monkeypatch.setattr(
        HfModelHandler,
        "get_hf_model_config",
        lambda self: types.SimpleNamespace(to_dict=lambda: {"model_type": "nemotron_h"}),
    )


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("lightning_recipe_runner", DIRECTORY / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_mobius(monkeypatch):
    class FakePackage(dict):
        def save(self, directory, **kwargs):
            assert kwargs == {"external_data": "onnx"}
            root = Path(directory)
            for component in self:
                path = root / component
                path.mkdir(parents=True)
                (path / "model.onnx").write_bytes(b"fake graph, not an ONNX validation fixture")
                (path / "model.onnx.data").write_bytes(b"fake external data")
            (root / "weight-loading-report.json").write_text("{}\n", encoding="utf-8")

    module = types.ModuleType("mobius")
    module.build = Mock(return_value=FakePackage(decoder=None, mtp=None))
    monkeypatch.setitem(sys.modules, "mobius", module)
    return module.build


def test_export_preserves_paired_paths_pins_and_unvalidated_status(tmp_path, fake_mobius):
    output = exporter().run(model(), str(tmp_path / "export"))
    fake_mobius.assert_called_once_with(
        MODEL_ID,
        revision=MODEL_REVISION,
        task="nemotron-h-mtp",
        dtype="bf16",
        execution_provider="cuda",
        load_weights=True,
        trust_remote_code=False,
    )
    assert output.model_component_names == ["decoder", "mtp"]
    assert output.model_attributes["no_flatten"] is True
    assert output.model_attributes["validation_status"] == "full_model_parity_blocked"
    assert output.model_attributes["speculative_runtime"] == "not_implemented"
    assert output.model_attributes["additional_files"] == [str(tmp_path / "export/weight-loading-report.json")]
    for name, component in output.get_model_components():
        assert component.model_path == str(tmp_path / "export" / name / "model.onnx")
        assert Path(component.model_path).with_name("model.onnx.data").is_file()
    assert not (tmp_path / "export/genai_config.json").exists()


def test_export_refuses_default_gate_before_build(tmp_path, fake_mobius):
    with pytest.raises(ValueError, match="Full-model parity is blocked"):
        exporter(allow_unvalidated_export=False).run(model(), str(tmp_path / "export"))
    fake_mobius.assert_not_called()
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize(
    "accelerator",
    [
        AcceleratorSpec(Device.CPU, "CPUExecutionProvider"),
        AcceleratorSpec(Device.GPU, "NvTensorRTRTXExecutionProvider"),
    ],
)
def test_export_refuses_other_providers(tmp_path, fake_mobius, accelerator):
    with pytest.raises(ValueError, match="no provider fallback"):
        exporter(accelerator).run(model(), str(tmp_path / "export"))
    fake_mobius.assert_not_called()


@pytest.mark.parametrize("revision", [None, "main", "different-revision"])
def test_export_refuses_unpinned_revision(tmp_path, fake_mobius, revision):
    with pytest.raises(ValueError, match="exact pinned"):
        exporter().run(model(revision=revision), str(tmp_path / "export"))
    fake_mobius.assert_not_called()


def test_export_refuses_remote_model_code(tmp_path, fake_mobius):
    with pytest.raises(ValueError, match="Remote model code"):
        exporter().run(model(trust_remote_code=True), str(tmp_path / "export"))
    fake_mobius.assert_not_called()


@pytest.mark.parametrize("precision", ["fp16", None])
def test_export_refuses_non_bf16_precision(tmp_path, fake_mobius, precision):
    with pytest.raises(ValueError):
        exporter(precision=precision).run(model(), str(tmp_path / "export"))
    fake_mobius.assert_not_called()


def test_export_refuses_other_checkpoint(tmp_path, fake_mobius):
    wrong = HfModelHandler(model_path="nvidia/unsupported-model", load_kwargs={"revision": MODEL_REVISION})
    with pytest.raises(ValueError, match="exact pinned"):
        exporter().run(wrong, str(tmp_path / "export"))
    fake_mobius.assert_not_called()


def test_export_preserves_existing_output(tmp_path, fake_mobius):
    output = tmp_path / "export"
    output.mkdir()
    sentinel = output / "previous.onnx"
    sentinel.write_bytes(b"previous artifact")
    with pytest.raises(FileExistsError, match="fresh, empty"):
        exporter().run(model(), str(output))
    assert sentinel.read_bytes() == b"previous artifact"
    fake_mobius.assert_not_called()


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_export_refuses_non_directory_or_symlink(tmp_path, fake_mobius, kind):
    output = tmp_path / "export"
    if kind == "file":
        output.write_bytes(b"previous artifact")
    else:
        target = tmp_path / "previous"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="fresh, empty"):
        exporter().run(model(), str(output))
    fake_mobius.assert_not_called()


def test_export_refuses_non_hf_input(tmp_path, fake_mobius):
    from olive.model import ONNXModelHandler

    previous = tmp_path / "previous.onnx"
    previous.write_bytes(b"fake graph")
    with pytest.raises(ValueError, match="HfModel input"):
        exporter().run(ONNXModelHandler(model_path=str(previous)), str(tmp_path / "export"))
    fake_mobius.assert_not_called()


def test_export_rejects_incomplete_package(tmp_path, fake_mobius):
    del fake_mobius.return_value["mtp"]
    with pytest.raises(RuntimeError, match="both decoder and mtp"):
        exporter().run(model(), str(tmp_path / "export"))
    assert not (tmp_path / "export").exists()


def test_runner_registers_pass_and_preserves_default_gate(monkeypatch, runner):
    import olive

    invoke = Mock()
    monkeypatch.setattr(olive, "run", invoke)
    monkeypatch.setattr(sys, "argv", ["run.py", "--allow-unvalidated-export"])
    runner.main()
    config = invoke.call_args.args[0]
    assert config["passes"]["export"]["allow_unvalidated_export"] is True
    parsed = RunConfig.model_validate(config)
    assert parsed.input_model.config["model_path"] == MODEL_ID
    package = OlivePackageConfig.model_validate(invoke.call_args.kwargs["package_config"])
    assert package.import_pass_module("NemotronLightningBuilder") is NemotronLightningBuilder
    assert NemotronLightningBuilder.supported_providers == {"CUDAExecutionProvider"}
    assert NemotronLightningBuilder.supported_precisions == {"bf16"}
    source = json.loads((DIRECTORY / "config.json").read_text(encoding="utf-8"))
    assert source["passes"]["export"]["allow_unvalidated_export"] is False
    for path in (DIRECTORY / "info.yml", DIRECTORY.parent / "info.yml"):
        metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert metadata["arch"] == "nemotron_h"
        assert all(entry["eps"] == "CUDAExecutionProvider" for entry in metadata["recipes"])


def test_runner_requires_explicit_opt_in_before_importing_olive(monkeypatch, runner):
    monkeypatch.setattr(sys, "argv", ["run.py"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_real_olive_workflow_preserves_composite_and_reports(tmp_path, monkeypatch, fake_mobius, runner):
    from olive.model.config import ModelConfig
    from olive.systems.local import LocalSystem

    # Advertise the mocked provider only; this exercises packaging, not CUDA kernels.
    monkeypatch.setattr(LocalSystem, "get_supported_execution_providers", lambda self: ["CUDAExecutionProvider"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run.py", "--allow-unvalidated-export"])
    runner.main()
    fake_mobius.assert_called_once()
    artifacts = list((tmp_path / "cuda/models").rglob("model_config.json"))
    assert len(artifacts) == 1
    published = ModelConfig.model_validate(json.loads(artifacts[0].read_text(encoding="utf-8"))).create_model()
    assert published.model_component_names == ["decoder", "mtp"]
    assert published.model_attributes["validation_status"] == "full_model_parity_blocked"
    assert published.model_attributes["speculative_runtime"] == "not_implemented"
    assert published.model_attributes["additional_files"] == [str(artifacts[0].parent / "weight-loading-report.json")]
    for name, component in published.get_model_components():
        graph = artifacts[0].parent / name / "model.onnx"
        assert Path(component.model_path) == graph
        assert graph.is_file()
        assert graph.with_name("model.onnx.data").read_bytes() == b"fake external data"
    assert (artifacts[0].parent / "weight-loading-report.json").is_file()
    assert not (artifacts[0].parent / "genai_config.json").exists()


def test_real_olive_workflow_reports_failed_export(tmp_path, monkeypatch, fake_mobius, runner):
    from olive.systems.local import LocalSystem

    monkeypatch.setattr(LocalSystem, "get_supported_execution_providers", lambda self: ["CUDAExecutionProvider"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run.py", "--allow-unvalidated-export"])
    fake_mobius.side_effect = RuntimeError("mock export failure")
    with pytest.raises(RuntimeError, match="Olive did not produce the paired export"):
        runner.main()
    fake_mobius.assert_called_once()
    assert not list((tmp_path / "cuda/models").rglob("model_config.json"))
