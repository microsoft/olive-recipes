"""Shared Mobius exporter for CLM and KEV Olive recipes."""

from __future__ import annotations

import dataclasses
import gc
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import onnx
import onnx_ir as ir
import torch
from mobius.integrations.ort_genai import write_ort_genai_config
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    QuantFormat,
)
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mobius import ArchitectureConfig, build_clm_package, build_kev_package
from non_generative_runtime_optimizations import finalize_clm_package

ModelName = Literal["clm", "kev", "kev08"]
Precision = Literal["fp32", "fp16", "mixed", "both"]


@dataclass(frozen=True)
class Recipe:
    """Immutable model identity and package policy."""

    name: ModelName
    base: str
    base_revision: str
    artifact_revision: str
    fp32_directory: str
    mixed_directory: str
    fp16_directory: str
    component: str
    mixed_model_id: str
    int4_accuracy_level: int


RECIPES = {
    "clm": Recipe(
        name="clm",
        base="Qwen/Qwen3-8B",
        base_revision="b968826d9c46dd6066d109eabc6255188de91218",
        artifact_revision="e939398d4556fcd9400c76fa8c5a513202f42b0a",
        fp32_directory="clm-v0.1-8b-fp32",
        mixed_directory="clm-v0.1-8b-mixed-middle-mlp",
        fp16_directory="clm-v0.1-8b-fp16-backbone",
        component="encoder",
        mixed_model_id="clm-v0.1-8b-mixed-cuda:1",
        int4_accuracy_level=4,
    ),
    "kev": Recipe(
        name="kev",
        base="Qwen/Qwen3.5-4B-Base",
        base_revision="1001bb4d826a52d1f399e183466143f4da7b741b",
        artifact_revision="139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
        fp32_directory="kev-4b-fp32",
        mixed_directory="kev-4b-mixed-middle-mlp",
        fp16_directory="kev-4b-fp16-backbone",
        component="backbone",
        mixed_model_id="kev-4b-mixed-cuda:1",
        int4_accuracy_level=3,
    ),
    "kev08": Recipe(
        name="kev08",
        base="Qwen/Qwen3.5-0.8B-Base",
        base_revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        artifact_revision="bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
        fp32_directory="kev-0.8b-fp32",
        mixed_directory="kev-0.8b-mixed-middle-mlp",
        fp16_directory="kev-0.8b-fp16-backbone",
        component="backbone",
        mixed_model_id="kev-0.8b-mixed-cuda:1",
        int4_accuracy_level=3,
    ),
}


def write_json(path: Path, value: dict) -> None:
    """Write stable human-readable JSON."""
    path.write_text(f"{json.dumps(value, indent=2)}\n", encoding="utf-8")


def architecture_config(recipe: Recipe, dtype: ir.DataType) -> ArchitectureConfig:
    """Load a pinned base architecture in the requested graph dtype."""
    parent = AutoConfig.from_pretrained(
        recipe.base,
        revision=recipe.base_revision,
    )
    primary = getattr(parent, "text_config", parent)
    config = ArchitectureConfig.from_transformers(
        primary,
        parent_config=parent if primary is not parent else None,
    )
    return dataclasses.replace(config, dtype=dtype)


def component_names(recipe: Recipe) -> tuple[str, ...]:
    """Return the expected Mobius FP32 component set."""
    if recipe.name == "clm":
        return ("encoder", "state_head", "action_head", "scorer")
    return ("backbone", "pointer_head")


def reusable_package(path: Path, recipe: Recipe) -> bool:
    """Return whether a package can be reused after an interrupted run."""
    required = [
        path / "component_manifest.json",
        path / "inference_model.json",
        path / "tokenizer.json",
        path / "tokenizer_config.json",
        *(path / name / "model.onnx" for name in component_names(recipe)),
        *(path / name / "model.onnx.data" for name in component_names(recipe)),
    ]
    if recipe.name != "clm":
        required.append(path / "genai_config.json")
    return all(item.is_file() for item in required)


def load_clm_head(artifact: Path):
    """Load the released CLM head checkpoint."""
    return torch.load(
        artifact / "CLM_v0.1-8B.pt",
        map_location="cpu",
        weights_only=True,
    )


def load_kev(recipe: Recipe, artifact: Path, dtype: torch.dtype):
    """Load the pinned KEV base, merge its adapter, and return dense weights."""
    checkpoint = torch.load(
        artifact / "head.pt",
        map_location="cpu",
        weights_only=True,
    )
    container = AutoModelForCausalLM.from_pretrained(
        recipe.base,
        revision=recipe.base_revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    adapted = PeftModel.from_pretrained(container.model, artifact)
    merged = adapted.merge_and_unload()
    weights = {f"model.{name}": value for name, value in merged.state_dict().items()}
    return checkpoint, container, adapted, merged, weights


def manifest(recipe: Recipe) -> dict:
    """Create the runtime component manifest."""
    if recipe.name == "clm":
        components = {
            "encoder": {"role": "backbone", "filename": "encoder/model.onnx"},
            "state_head": {"role": "head", "filename": "state_head/model.onnx"},
            "action_head": {"role": "head", "filename": "action_head/model.onnx"},
            "scorer": {"role": "scorer", "filename": "scorer/model.onnx"},
        }
        model_type = "clm-v0.1-8b"
    else:
        components = {
            "backbone": {
                "role": "backbone",
                "filename": "backbone/model.onnx",
            },
            "pointer_head": {
                "role": "head",
                "filename": "pointer_head/model.onnx",
            },
        }
        model_type = "kev-4b"
    return {
        "schema_version": 1,
        "model_type": model_type,
        "components": components,
    }


def inference_metadata(
    recipe: Recipe,
    *,
    precision: Literal["fp32", "fp16", "mixed"],
) -> dict:
    """Create Foundry metadata for one exported package."""
    if recipe.name == "clm":
        source = "Contrastive-LM/CLM-v0.1-8B"
        alias = "clm"
        task = "text-ranking"
        capabilities = ["structured-input", "candidate-ranking", "action-cache"]
        fp32_id = "clm-v0.1-8b-generic-cpu:1"
        fp16_id = "clm-v0.1-8b-fp16-cuda:1"
    else:
        size = "0.8b" if recipe.name == "kev08" else "4b"
        source = f"jaredpalmer/kev-{size}"
        alias = "kev"
        task = "typed-decision"
        capabilities = ["structured-input", "noul", "choice", "score"]
        fp32_id = f"kev-{size}-generic-cpu:1"
        fp16_id = f"kev-{size}-fp16-cuda:1"
    return {
        "Name": (
            recipe.mixed_model_id
            if precision == "mixed"
            else (fp16_id if precision == "fp16" else fp32_id)
        ),
        "Alias": alias,
        "Task": task,
        "ComponentManifest": "component_manifest.json",
        "License": "Apache-2.0",
        "Provenance": {
            "source": source,
            "artifact_revision": recipe.artifact_revision,
            "base_model": recipe.base,
            "base_revision": recipe.base_revision,
        },
        "Provider": {
            "execution_provider": "cuda" if precision != "fp32" else "cpu",
            "variant": (
                "fp16-int4-middle-mlp-edge4" if precision == "mixed" else precision
            ),
        },
        "Capabilities": capabilities,
    }


def save_package_metadata(
    recipe: Recipe,
    output: Path,
    *,
    precision: Literal["fp32", "fp16", "mixed"],
) -> None:
    """Write runtime and Foundry manifests."""
    write_json(output / "component_manifest.json", manifest(recipe))
    write_json(
        output / "inference_model.json",
        inference_metadata(recipe, precision=precision),
    )
    if recipe.name != "clm" and precision != "fp32":
        config_path = output / "genai_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        session_options = config["model"]["decoder"]["session_options"]
        session_options["intra_op_num_threads"] = 1
        session_options["inter_op_num_threads"] = 1
        write_json(config_path, config)


def apply_component_session_options(
    package: Path,
    options: dict,
) -> None:
    """Apply recipe-owned ORT session options to a generated component config."""
    expected = {
        "intra_op_num_threads",
        "inter_op_num_threads",
        "session.intra_op.allow_spinning",
        "session.inter_op.allow_spinning",
    }
    if set(options) != expected:
        raise ValueError(
            "component_session_options must contain exactly "
            f"{sorted(expected)}, got {sorted(options)}"
        )
    for name in ("intra_op_num_threads", "inter_op_num_threads"):
        value = options[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"component_session_options.{name} must be a positive integer"
            )
    for name in (
        "session.intra_op.allow_spinning",
        "session.inter_op.allow_spinning",
    ):
        if options[name] not in {"0", "1"}:
            raise ValueError(f"component_session_options.{name} must be '0' or '1'")

    config_path = package / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["decoder"]["session_options"] = dict(options)
    write_json(config_path, config)


def apply_execution_provider_metadata(package: Path, execution_provider: str) -> None:
    """Stamp the provider selected by the Olive recipe onto published metadata."""
    path = package / "inference_model.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["Provider"]["execution_provider"] = execution_provider
    name = metadata["Name"]
    if execution_provider == "webgpu":
        name = name.replace("-generic-cpu:", "-generic-webgpu:")
        name = name.replace("-cuda:", "-webgpu:")
    elif execution_provider == "cpu":
        name = name.replace("-generic-webgpu:", "-generic-cpu:")
    elif execution_provider == "cuda":
        name = name.replace("-webgpu:", "-cuda:")
    metadata["Name"] = name
    write_json(path, metadata)


def apply_component_runtime(package: Path, component_runtime: dict) -> None:
    """Write provider/runtime policies that are independent of model graphs."""
    if not component_runtime:
        raise ValueError("component_runtime must be a non-empty object")
    components = {}
    for component, policy in component_runtime.items():
        if not isinstance(component, str) or not component:
            raise ValueError("component_runtime keys must be non-empty strings")
        allowed = {"cuda_graph_max_signatures", "cuda_graph_max_bytes"}
        if (
            not isinstance(policy, dict)
            or "cuda_graph_max_signatures" not in policy
            or not set(policy) <= allowed
        ):
            raise ValueError(
                "each component_runtime policy must define "
                "cuda_graph_max_signatures and may define cuda_graph_max_bytes"
            )
        limit = policy["cuda_graph_max_signatures"]
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError(
                "cuda_graph_max_signatures must be a non-negative integer"
            )
        result = {"cuda_graph_max_signatures": limit}
        if "cuda_graph_max_bytes" in policy:
            max_bytes = policy["cuda_graph_max_bytes"]
            if (
                not isinstance(max_bytes, int)
                or isinstance(max_bytes, bool)
                or max_bytes < 0
            ):
                raise ValueError(
                    "cuda_graph_max_bytes must be a non-negative integer"
                )
            result["cuda_graph_max_bytes"] = max_bytes
        components[component] = result
    write_json(
        package / "component_runtime.json",
        {"schema_version": 1, "components": components},
    )


def export_fp32(
    recipe: Recipe,
    artifact: Path,
    output_root: Path,
    execution_provider: str,
) -> None:
    """Export one complete FP32 Mobius package."""
    output = output_root / recipe.fp32_directory
    if output.exists():
        raise FileExistsError(output)
    config = architecture_config(recipe, ir.DataType.FLOAT)
    if recipe.name == "clm":
        checkpoint = load_clm_head(artifact)
        base = AutoModelForCausalLM.from_pretrained(
            recipe.base,
            revision=recipe.base_revision,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        package = build_clm_package(
            config,
            base_weights=base.state_dict(),
            head_checkpoint=checkpoint,
            base_revision=recipe.base_revision,
            execution_provider=execution_provider,
        )
        owners = (package, base, checkpoint)
    else:
        checkpoint, container, adapted, merged, weights = load_kev(
            recipe,
            artifact,
            torch.float32,
        )
        package = build_kev_package(
            config,
            head_checkpoint=checkpoint,
            merged_base_weights=weights,
            execution_provider=execution_provider,
        )
        owners = (package, weights, merged, adapted, container, checkpoint)
    package.save(output, external_data="onnx", max_workers=1)
    if recipe.name != "clm":
        write_ort_genai_config(
            package,
            str(output),
            ep=execution_provider,
            context_length=8192,
        )
    AutoTokenizer.from_pretrained(
        recipe.base,
        revision=recipe.base_revision,
    ).save_pretrained(output)
    save_package_metadata(recipe, output, precision="fp32")
    del package, owners
    gc.collect()


def combine_fp16(
    recipe: Recipe,
    source: Path,
    temporary: Path,
    output: Path,
) -> None:
    """Copy FP32 heads and replace only the backbone with its FP16 graph."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for child in source.iterdir():
        if child.name == recipe.component:
            continue
        destination = output / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)
    saved = temporary / recipe.component
    if not saved.is_dir():
        saved = temporary
    target = output / recipe.component
    target.mkdir()
    for child in saved.iterdir():
        if child.name.startswith("model.onnx"):
            shutil.copy2(child, target / child.name)


def export_fp16_backbone(
    recipe: Recipe,
    artifact: Path,
    output_root: Path,
    execution_provider: str,
) -> None:
    """Export an FP16 backbone while preserving FP32 decision heads."""
    config = architecture_config(recipe, ir.DataType.FLOAT16)
    if recipe.name == "clm":
        checkpoint = load_clm_head(artifact)
        base = AutoModelForCausalLM.from_pretrained(
            recipe.base,
            revision=recipe.base_revision,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        package = build_clm_package(
            config,
            base_weights=base.state_dict(),
            head_checkpoint=checkpoint,
            base_revision=recipe.base_revision,
            execution_provider=execution_provider,
        )
        owners = (package, base, checkpoint)
    else:
        checkpoint, container, adapted, merged, weights = load_kev(
            recipe,
            artifact,
            torch.float16,
        )
        package = build_kev_package(
            config,
            head_checkpoint=checkpoint,
            merged_base_weights=weights,
            execution_provider=execution_provider,
        )
        owners = (package, weights, merged, adapted, container, checkpoint)
    with tempfile.TemporaryDirectory(prefix=f"{recipe.name}-fp16-") as directory:
        temporary = Path(directory)
        package.save(
            temporary,
            external_data="onnx",
            max_workers=1,
            components=lambda name: name == recipe.component,
        )
        combine_fp16(
            recipe,
            output_root / recipe.fp32_directory,
            temporary,
            output_root / recipe.fp16_directory,
        )
    save_package_metadata(
        recipe,
        output_root / recipe.fp16_directory,
        precision="fp16",
    )
    del package, owners
    gc.collect()


def export_clm_safe_encoder(
    recipe: Recipe,
    artifact: Path,
    package_path: Path,
    execution_provider: str,
) -> None:
    """Add a BF16 CLM encoder used only after non-finite FP16 output."""
    if recipe.name != "clm":
        raise ValueError("safe encoder export is supported only for CLM")
    checkpoint = load_clm_head(artifact)
    base = AutoModelForCausalLM.from_pretrained(
        recipe.base,
        revision=recipe.base_revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    package = build_clm_package(
        architecture_config(recipe, ir.DataType.BFLOAT16),
        base_weights=base.state_dict(),
        head_checkpoint=checkpoint,
        base_revision=recipe.base_revision,
        execution_provider=execution_provider,
    )
    with tempfile.TemporaryDirectory(prefix="clm-safe-bf16-") as directory:
        temporary = Path(directory)
        package.save(
            temporary,
            external_data="onnx",
            max_workers=1,
            components=lambda name: name == "encoder",
        )
        source = temporary / "encoder"
        if not source.is_dir():
            source = temporary
        destination = package_path / "safe_encoder"
        destination.mkdir()
        for child in source.iterdir():
            if child.name.startswith("model.onnx"):
                shutil.copy2(child, destination / child.name)
    del package, base, checkpoint
    gc.collect()


def quantize_mixed(recipe: Recipe, output_root: Path) -> None:
    """Apply the selected middle-MLP INT4 block-128 policy."""
    source = output_root / recipe.fp16_directory
    output = output_root / recipe.mixed_directory
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for child in source.iterdir():
        if child.name == recipe.component:
            continue
        destination = output / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)
    source_model = source / recipe.component / "model.onnx"
    output_model = output / recipe.component / "model.onnx"
    output_model.parent.mkdir(parents=True)
    model = onnx.load(source_model, load_external_data=False)
    matmuls = [node for node in model.graph.node if node.op_type == "MatMul"]
    layers = [
        int(match.group(1))
        for node in matmuls
        if (match := re.search(r"/layers\.(\d+)/", node.name))
    ]
    last_layer = max(layers)
    excluded = []
    for node in matmuls:
        match = re.search(r"/layers\.(\d+)/", node.name)
        layer = int(match.group(1)) if match else -1
        if not (4 <= layer <= last_layer - 4 and "/mlp/" in node.name):
            excluded.append(node.name)
    quantizer = MatMulNBitsQuantizer(
        str(source_model),
        bits=4,
        block_size=128,
        is_symmetric=True,
        accuracy_level=recipe.int4_accuracy_level,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        nodes_to_exclude=excluded,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(
        str(output_model),
        use_external_data_format=True,
    )
    quantized = onnx.load(output_model, load_external_data=False)
    count = sum(
        node.domain == "com.microsoft" and node.op_type == "MatMulNBits"
        for node in quantized.graph.node
    )
    if count == 0:
        raise RuntimeError("mixed graph contains no MatMulNBits nodes")
    onnx.checker.check_model(str(output_model))
    save_package_metadata(recipe, output, precision="mixed")


def run_recipe(
    model: ModelName,
    artifact: Path,
    output_root: Path,
    precision: Precision,
    execution_provider: str = "default",
) -> None:
    """Run one model-specific recipe."""
    recipe = RECIPES[model]
    output_root = output_root.resolve()
    artifact = artifact.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fp32 = output_root / recipe.fp32_directory
    mixed = output_root / recipe.mixed_directory
    fp32_ready = False
    mixed_ready = False
    if fp32.exists():
        if reusable_package(fp32, recipe):
            fp32_ready = True
            print(f"reusing existing FP32 package: {fp32}")
        else:
            print(f"removing incomplete FP32 package: {fp32}")
            shutil.rmtree(fp32)
    if precision in {"mixed", "both"} and mixed.exists():
        if reusable_package(mixed, recipe):
            mixed_ready = True
            print(f"reusing existing mixed package: {mixed}")
        else:
            print(f"removing incomplete mixed package: {mixed}")
            shutil.rmtree(mixed)
    if mixed_ready and precision == "mixed":
        return
    if not fp32_ready:
        export_fp32(recipe, artifact, output_root, execution_provider)
    if precision == "fp16":
        fp16 = output_root / recipe.fp16_directory
        if fp16.exists():
            shutil.rmtree(fp16)
        export_fp16_backbone(recipe, artifact, output_root, execution_provider)
        shutil.rmtree(fp32)
        return
    if precision in {"mixed", "both"} and not mixed_ready:
        fp16 = output_root / recipe.fp16_directory
        if fp16.exists():
            print(f"removing incomplete FP16 package: {fp16}")
            shutil.rmtree(fp16)
        export_fp16_backbone(recipe, artifact, output_root, execution_provider)
        quantize_mixed(recipe, output_root)
        shutil.rmtree(fp16)
    if precision in {"mixed", "both"}:
        if precision == "mixed":
            shutil.rmtree(fp32)


def olive_export(
    *,
    model_name: ModelName,
    output_dir: Path,
    execution_provider: str,
    exporter_config: dict,
) -> dict[str, list[str]]:
    """Export one recipe into an Olive-owned output directory."""
    recipe = RECIPES[model_name]
    artifact_value = exporter_config.get("artifact_path")
    if not isinstance(artifact_value, str) or not artifact_value:
        raise ValueError("exporter_config.artifact_path is required")
    precision = exporter_config.get("recipe_precision", "fp32")
    if precision not in {"fp32", "fp16", "mixed"}:
        raise ValueError("recipe_precision must be fp32, fp16, or mixed")
    allowed_providers = {
        "fp32": {"cpu", "webgpu"},
        "fp16": {"cuda", "webgpu"},
        "mixed": {"cuda", "webgpu"},
    }[precision]
    if execution_provider not in allowed_providers:
        raise ValueError(
            f"recipe_precision={precision!r} requires execution_provider in "
            f"{sorted(allowed_providers)}, got {execution_provider!r}"
        )
    component_options = exporter_config.get("component_session_options")
    if recipe.name != "clm" and not isinstance(component_options, dict):
        raise ValueError(
            "exporter_config.component_session_options is required for KEV"
        )
    staging_value = exporter_config.get("staging_path")
    if not isinstance(staging_value, str) or not staging_value:
        raise ValueError("exporter_config.staging_path is required")

    output_dir = Path(output_dir)
    if any(output_dir.iterdir()):
        raise ValueError(f"Olive output directory must be empty: {output_dir}")
    staging = Path(staging_value)
    run_recipe(
        model_name,
        Path(artifact_value),
        staging,
        precision,
        execution_provider,
    )
    selected = (
        staging / recipe.mixed_directory
        if precision == "mixed"
        else (
            staging / recipe.fp16_directory
            if precision == "fp16"
            else staging / recipe.fp32_directory
        )
    )
    if recipe.name != "clm":
        assert isinstance(component_options, dict)
        apply_component_session_options(selected, component_options)
    apply_execution_provider_metadata(selected, execution_provider)
    component_runtime = exporter_config.get("component_runtime")
    if component_runtime is not None:
        if not isinstance(component_runtime, dict):
            raise ValueError("component_runtime must be an object")
        apply_component_runtime(selected, component_runtime)
    runtime_optimizations = exporter_config.get("runtime_optimizations")
    if runtime_optimizations is not None:
        if recipe.name != "clm" or not isinstance(runtime_optimizations, dict):
            raise ValueError(
                "runtime_optimizations is supported only for CLM and must be an object"
            )
        export_clm_safe_encoder(
            recipe,
            Path(artifact_value).resolve(),
            selected,
            execution_provider,
        )
        finalize_clm_package(
            selected, runtime_optimizations, execution_provider
        )
    publishing = output_dir.with_name(f".{output_dir.name}.publishing")
    if publishing.exists():
        shutil.rmtree(publishing)
    shutil.copytree(selected, publishing)
    output_dir.rmdir()
    publishing.replace(output_dir)
    return {"components": list(component_names(recipe))}
