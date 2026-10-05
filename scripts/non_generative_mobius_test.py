"""Tests for the shared CLM and KEV Olive exporter."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import scripts.non_generative_mobius as exporter
from scripts.non_generative_mobius import (
    RECIPES,
    apply_component_session_options,
    apply_execution_provider_metadata,
    olive_export,
    reusable_package,
    save_package_metadata,
)


def _write_kev_config(path, threads: int = 16) -> None:
    path.write_text(
        json.dumps(
            {
                "model": {
                    "decoder": {
                        "session_options": {
                            "intra_op_num_threads": threads,
                            "inter_op_num_threads": 1,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_mixed_kev_metadata_selects_accelerator_thread_defaults(tmp_path):
    _write_kev_config(tmp_path / "genai_config.json")

    save_package_metadata(RECIPES["kev"], tmp_path, precision="mixed")

    config = json.loads((tmp_path / "genai_config.json").read_text())
    assert config["model"]["decoder"]["session_options"] == {
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
    }


def test_recipe_session_options_override_generated_config(tmp_path):
    config_path = tmp_path / "genai_config.json"
    _write_kev_config(config_path)
    options = {
        "intra_op_num_threads": 24,
        "inter_op_num_threads": 1,
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0",
    }

    apply_component_session_options(tmp_path, options)

    config = json.loads(config_path.read_text())
    assert config["model"]["decoder"]["session_options"] == options


def test_recipe_provider_overrides_published_metadata(tmp_path):
    (tmp_path / "inference_model.json").write_text(
        json.dumps(
            {
                "Name": "kev-0.8b-fp16-cuda:1",
                "Provider": {
                    "execution_provider": "cuda",
                    "variant": "fp16",
                },
            }
        )
    )

    apply_execution_provider_metadata(tmp_path, "webgpu")

    metadata = json.loads((tmp_path / "inference_model.json").read_text())
    assert metadata["Name"] == "kev-0.8b-fp16-webgpu:1"
    assert metadata["Provider"]["execution_provider"] == "webgpu"

    apply_execution_provider_metadata(tmp_path, "cuda")
    metadata = json.loads((tmp_path / "inference_model.json").read_text())
    assert metadata["Name"] == "kev-0.8b-fp16-cuda:1"
    assert metadata["Provider"]["execution_provider"] == "cuda"


@pytest.mark.parametrize(
    ("config_path", "expected_threads"),
    [
        ("kev-4b/cpu/kev-4b_cpu_fp32.json", 16),
        ("kev-4b/cuda/kev-4b_cuda_mixed.json", 1),
        ("kev-0.8b/cpu/kev-0.8b_cpu_fp32.json", 16),
        ("kev-0.8b/cuda/kev-0.8b_cuda_fp16.json", 1),
        ("kev-0.8b/cuda/kev-0.8b_cuda_mixed.json", 1),
    ],
)
def test_kev_recipe_json_declares_component_session_options(
    config_path, expected_threads
):
    config = json.loads((Path(__file__).parents[1] / config_path).read_text())
    options = config["passes"]["m"]["exporter_config"]["component_session_options"]
    assert options == {
        "intra_op_num_threads": expected_threads,
        "inter_op_num_threads": 1,
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0",
    }


@pytest.mark.parametrize(
    "config_path",
    [
        "Contrastive-LM-CLM-v0.1-8B/webgpu/CLM-v0.1-8B_webgpu_fp32.json",
        "Contrastive-LM-CLM-v0.1-8B/webgpu/CLM-v0.1-8B_webgpu_mixed.json",
        "kev-4b/webgpu/kev-4b_webgpu_fp32.json",
        "kev-4b/webgpu/kev-4b_webgpu_mixed.json",
        "kev-0.8b/webgpu/kev-0.8b_webgpu_fp32.json",
        "kev-0.8b/webgpu/kev-0.8b_webgpu_fp16.json",
        "kev-0.8b/webgpu/kev-0.8b_webgpu_mixed.json",
    ],
)
def test_webgpu_recipe_targets_webgpu_provider(config_path):
    config = json.loads((Path(__file__).parents[1] / config_path).read_text())
    accelerator = config["systems"]["local_system"]["accelerators"][0]
    assert accelerator == {
        "device": "gpu",
        "execution_providers": ["WebGpuExecutionProvider"],
    }


def test_webgpu_publication_stamps_provider_metadata(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    package = staging / RECIPES["kev08"].fp16_directory
    for component in ("backbone", "pointer_head"):
        (package / component).mkdir(parents=True)
    (package / "inference_model.json").write_text(
        json.dumps(
            {
                "Name": "kev-0.8b-fp16-cuda:1",
                "Provider": {
                    "execution_provider": "cuda",
                    "variant": "fp16",
                },
            }
        )
    )
    _write_kev_config(package / "genai_config.json")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(exporter, "run_recipe", lambda *args: None)

    result = olive_export(
        model_name="kev08",
        output_dir=output,
        execution_provider="webgpu",
        exporter_config={
            "artifact_path": "artifact",
            "recipe_precision": "fp16",
            "staging_path": str(staging),
            "component_session_options": {
                "intra_op_num_threads": 1,
                "inter_op_num_threads": 1,
                "session.intra_op.allow_spinning": "0",
                "session.inter_op.allow_spinning": "0",
            },
        },
    )

    metadata = json.loads((output / "inference_model.json").read_text())
    assert result == {"components": ["backbone", "pointer_head"]}
    assert metadata["Name"] == "kev-0.8b-fp16-webgpu:1"
    assert metadata["Provider"]["execution_provider"] == "webgpu"


def test_run_recipe_forwards_webgpu_to_mobius_build(tmp_path, monkeypatch):
    captured = {}

    def fake_export(recipe, artifact, output_root, execution_provider):
        captured.update(
            recipe=recipe.name,
            artifact=artifact,
            output_root=output_root,
            execution_provider=execution_provider,
        )

    monkeypatch.setattr(exporter, "export_fp32", fake_export)
    artifact = tmp_path / "artifact"
    output_root = tmp_path / "output"

    exporter.run_recipe(
        "kev08",
        artifact,
        output_root,
        "fp32",
        "webgpu",
    )

    assert captured == {
        "recipe": "kev08",
        "artifact": artifact.resolve(),
        "output_root": output_root.resolve(),
        "execution_provider": "webgpu",
    }


def test_reusable_kev_package_requires_genai_config(tmp_path):
    recipe = RECIPES["kev"]
    for filename in (
        "component_manifest.json",
        "inference_model.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (tmp_path / filename).touch()
    for component in ("backbone", "pointer_head"):
        directory = tmp_path / component
        directory.mkdir()
        (directory / "model.onnx").touch()
        (directory / "model.onnx.data").touch()

    assert not reusable_package(tmp_path, recipe)
    (tmp_path / "genai_config.json").touch()
    assert reusable_package(tmp_path, recipe)


def test_kev_08_recipe_uses_immutable_model_contract():
    recipe = RECIPES["kev08"]

    assert recipe.base == "Qwen/Qwen3.5-0.8B-Base"
    assert recipe.base_revision == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    assert recipe.artifact_revision == "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"
    assert recipe.fp32_directory == "kev-0.8b-fp32"
    assert recipe.mixed_directory == "kev-0.8b-mixed-middle-mlp"


def test_olive_export_rejects_provider_precision_mismatch(tmp_path):
    with pytest.raises(ValueError, match="requires execution_provider in"):
        olive_export(
            model_name="kev",
            output_dir=tmp_path,
            execution_provider="cpu",
            exporter_config={
                "artifact_path": "artifact",
                "recipe_precision": "mixed",
                "staging_path": "staging",
            },
        )


def test_olive_export_requires_explicit_kev_session_options(tmp_path):
    with pytest.raises(ValueError, match="component_session_options is required"):
        olive_export(
            model_name="kev",
            output_dir=tmp_path,
            execution_provider="cpu",
            exporter_config={
                "artifact_path": "artifact",
                "recipe_precision": "fp32",
                "staging_path": "staging",
            },
        )


@pytest.mark.parametrize(
    ("script", "function_name", "model_name"),
    [
        ("kev-4b/user_script.py", "export_kev_package", "kev"),
        ("kev-0.8b/user_script.py", "export_kev_package", "kev08"),
        (
            "Contrastive-LM-CLM-v0.1-8B/user_script.py",
            "export_clm_package",
            "clm",
        ),
    ],
)
def test_user_script_forwards_execution_provider(
    script, function_name, model_name, tmp_path
):
    namespace = runpy.run_path(Path(__file__).parents[1] / script)
    function = namespace[function_name]
    captured = {}

    def fake_export(**kwargs):
        captured.update(kwargs)
        return {"components": ["component"]}

    function.__globals__["olive_export"] = fake_export
    result = function(
        output_dir=tmp_path,
        execution_provider="cuda",
        exporter_config={"recipe_precision": "mixed"},
    )

    assert result == {"components": ["component"]}
    assert captured == {
        "model_name": model_name,
        "output_dir": tmp_path,
        "execution_provider": "cuda",
        "exporter_config": {"recipe_precision": "mixed"},
    }
