import importlib.util
import json
import sys
import tempfile
import types
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parent
RECIPE_ROOT = SCRIPTS_DIR.parent
CPU_DIR = RECIPE_ROOT / "cpu"
TRT_RTX_DIR = RECIPE_ROOT / "NvTensorRtRtx"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    original_sys_path = sys.path.copy()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


cpu_optimize = _load_module("nemotron_cpu_optimize", CPU_DIR / "optimize.py")
trt_rtx_optimize = _load_module("nemotron_trt_rtx_optimize", TRT_RTX_DIR / "optimize.py")


def _load_export_tokenizer_module():
    return _load_module("nemotron_export_tokenizer", SCRIPTS_DIR / "export_tokenizer.py")


def _load_model_load_module(provider="NvTensorRtRtx"):
    provider_dir = CPU_DIR if provider == "cpu" else TRT_RTX_DIR
    return _load_module(f"nemotron_{provider}_model_load", provider_dir / "nemotron_model_load.py")


class NemotronOptimizePlanTest(unittest.TestCase):
    def test_cpu_encoder_precision_selects_provider_local_config(self):
        self.assertEqual(
            cpu_optimize.resolve_encoder_config("int4"),
            "nemotron_encoder_int4_cpu.json",
        )
        self.assertEqual(
            cpu_optimize.resolve_encoder_config("int8"),
            "nemotron_encoder_int8_cpu.json",
        )

    def test_cpu_configs_use_cpu_execution_provider(self):
        for config_name in (
            "nemotron_encoder_int4_cpu.json",
            "nemotron_encoder_int8_cpu.json",
            "nemotron_decoder_fp32_cpu.json",
            "nemotron_joint_fp32_cpu.json",
        ):
            with self.subTest(config_name=config_name):
                config = json.loads((CPU_DIR / config_name).read_text())
                self.assertEqual(
                    config["systems"]["local_system"]["accelerators"][0]["execution_providers"],
                    ["CPUExecutionProvider"],
                )
                self.assertEqual(
                    config["input_model"]["model_script"],
                    "cpu/nemotron_model_load.py",
                )

    def test_trtrtx_configs_use_fp16_opset23(self):
        expected_loaders = {
            "nemotron_encoder_fp16_trtrtx.json": (
                "encoder_fp16_model_loader",
                "encoder_fp16_dummy_inputs",
            ),
            "nemotron_decoder_fp16_trtrtx.json": (
                "decoder_fp16_model_loader",
                "decoder_fp16_dummy_inputs",
            ),
            "nemotron_joint_fp16_trtrtx.json": (
                "joint_fp16_model_loader",
                "joint_fp16_dummy_inputs",
            ),
        }
        for config_name, (model_loader, dummy_inputs_func) in expected_loaders.items():
            with self.subTest(config_name=config_name):
                config = json.loads((TRT_RTX_DIR / config_name).read_text())
                self.assertEqual(
                    config["systems"]["local_system"]["accelerators"][0]["execution_providers"],
                    ["NvTensorRTRTXExecutionProvider"],
                )
                self.assertEqual(config["passes"]["convert"]["target_opset"], 23)
                self.assertEqual(config["input_model"]["model_loader"], model_loader)
                self.assertEqual(config["input_model"]["dummy_inputs_func"], dummy_inputs_func)
                self.assertEqual(
                    config["input_model"]["model_script"],
                    "NvTensorRtRtx/nemotron_model_load.py",
                )

    def test_cpu_pipeline_uses_only_cpu_configs(self):
        with patch.object(cpu_optimize, "_run_olive_pipeline") as run_pipeline:
            cpu_optimize.run_olive_pipelines("output", "model", "int8")

        self.assertEqual(
            [call.args[0] for call in run_pipeline.call_args_list],
            [
                "nemotron_encoder_int8_cpu.json",
                "nemotron_decoder_fp32_cpu.json",
                "nemotron_joint_fp32_cpu.json",
            ],
        )

    def test_trtrtx_pipeline_uses_only_trtrtx_configs(self):
        with patch.object(trt_rtx_optimize, "_run_olive_pipeline") as run_pipeline:
            trt_rtx_optimize.run_olive_pipelines("output", "model")

        self.assertEqual(
            [call.args[0] for call in run_pipeline.call_args_list],
            [
                "nemotron_encoder_fp16_trtrtx.json",
                "nemotron_decoder_fp16_trtrtx.json",
                "nemotron_joint_fp16_trtrtx.json",
            ],
        )

    def test_cpu_entry_point_runs_only_cpu_pipeline(self):
        model_load = types.ModuleType("cpu.nemotron_model_load")
        model_load.MODEL_NAME = "nvidia/test-model"
        model_load.CHUNK_SIZE = 0.16

        with (
            patch.dict(sys.modules, {"cpu.nemotron_model_load": model_load}),
            patch.object(sys, "argv", ["optimize.py"]),
            patch.object(cpu_optimize, "run_olive_pipelines") as run_olive_pipelines,
            patch.object(cpu_optimize, "run_tokenizer_export"),
            patch.object(cpu_optimize, "generate_configs") as generate_configs,
            patch.object(cpu_optimize, "download_silero_vad"),
        ):
            cpu_optimize.main()

        self.assertEqual(
            run_olive_pipelines.call_args.kwargs["encoder_precision"],
            "int4",
        )
        self.assertTrue(generate_configs.call_args.kwargs["include_vad"])

    def test_trtrtx_entry_point_runs_only_trtrtx_pipeline(self):
        model_load = types.ModuleType("NvTensorRtRtx.nemotron_model_load")
        model_load.MODEL_NAME = "nvidia/test-model"
        model_load.CHUNK_SIZE = 0.16

        with (
            patch.dict(sys.modules, {"NvTensorRtRtx.nemotron_model_load": model_load}),
            patch.object(sys, "argv", ["optimize.py"]),
            patch.object(trt_rtx_optimize, "run_olive_pipelines") as run_olive_pipelines,
            patch.object(trt_rtx_optimize, "run_tokenizer_export"),
            patch.object(trt_rtx_optimize, "generate_configs") as generate_configs,
        ):
            trt_rtx_optimize.main()

        self.assertEqual(
            run_olive_pipelines.call_args.kwargs,
            {
                "output_dir": trt_rtx_optimize.DEFAULT_OUTPUT_DIR,
                "model_path": "nvidia/test-model",
            },
        )
        self.assertFalse(generate_configs.call_args.kwargs["include_vad"])

    def test_provider_readmes_document_vad_behavior(self):
        cpu_readme = (CPU_DIR / "README.md").read_text()
        trtrtx_readme = (TRT_RTX_DIR / "README.md").read_text()

        self.assertIn("downloads Silero VAD", cpu_readme)
        self.assertIn("Silero VAD is omitted", trtrtx_readme)

    def test_tokenizer_export_restores_local_checkpoint_on_cpu(self):
        export_tokenizer = _load_export_tokenizer_module()

        calls = []

        module_names = [
            "nemo",
            "nemo.collections",
            "nemo.collections.asr",
        ]
        saved_modules = {name: sys.modules.get(name) for name in module_names}
        for name in module_names:
            sys.modules.pop(name, None)

        try:
            nemo_module = types.ModuleType("nemo")
            nemo_module.__path__ = []
            collections_module = types.ModuleType("nemo.collections")
            collections_module.__path__ = []
            asr_module = types.ModuleType("nemo.collections.asr")

            class DummyTokenizer:
                def ids_to_tokens(self, ids):
                    return [f"tok_{ids[0]}"]

            class DummyModel:
                tokenizer = DummyTokenizer()
                cfg = SimpleNamespace(joint=SimpleNamespace(num_classes=1))

            class ASRModel:
                @staticmethod
                def restore_from(model_name, **kwargs):
                    calls.append(("restore", model_name, kwargs))
                    return DummyModel()

            asr_module.models = SimpleNamespace(ASRModel=ASRModel)
            nemo_module.collections = collections_module
            collections_module.asr = asr_module
            sys.modules["nemo"] = nemo_module
            sys.modules["nemo.collections"] = collections_module
            sys.modules["nemo.collections.asr"] = asr_module

            with tempfile.TemporaryDirectory() as tmpdir:
                tokens = export_tokenizer.extract_vocab("local.nemo", Path(tmpdir))

            self.assertEqual(calls, [("restore", "local.nemo", {"map_location": "cpu"})])
            self.assertEqual(tokens, ["tok_0", "<blank>"])
        finally:
            for name in module_names:
                if saved_modules[name] is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = saved_modules[name]

    def test_fp16_encoder_dummy_inputs_expose_fp16_floating_tensors(self):
        import torch

        model_load = _load_model_load_module()
        inputs = model_load.encoder_fp16_dummy_inputs(None)

        self.assertEqual(inputs[0].dtype, torch.float16)
        self.assertEqual(inputs[2].dtype, torch.float16)
        self.assertEqual(inputs[3].dtype, torch.float16)
        self.assertEqual(inputs[1].dtype, torch.int64)
        self.assertEqual(inputs[4].dtype, torch.int64)
        self.assertEqual(inputs[5].dtype, torch.int64)

    def test_fp16_decoder_and_joint_dummy_inputs_expose_fp16_floating_tensors(self):
        import torch

        model_load = _load_model_load_module()
        decoder_inputs = model_load.decoder_fp16_dummy_inputs(None)
        joint_inputs = model_load.joint_fp16_dummy_inputs(None)

        self.assertEqual(decoder_inputs[0].dtype, torch.int64)
        self.assertEqual(decoder_inputs[1].dtype, torch.float16)
        self.assertEqual(decoder_inputs[2].dtype, torch.float16)
        self.assertTrue(all(value.dtype == torch.float16 for value in joint_inputs))

    def test_fp16_component_loaders_convert_decoder_and_joint(self):
        import torch
        import torch.nn as nn

        model_load = _load_model_load_module()
        decoder_wrapper = nn.Linear(2, 2)
        joint_wrapper = nn.Linear(2, 2)
        with (
            patch.object(model_load, "decoder_model_loader", return_value=decoder_wrapper),
            patch.object(model_load, "joint_model_loader", return_value=joint_wrapper),
        ):
            decoder = model_load.decoder_fp16_model_loader("unused")
            joint = model_load.joint_fp16_model_loader("unused")

        self.assertEqual(next(decoder.parameters()).dtype, torch.float16)
        self.assertEqual(next(joint.parameters()).dtype, torch.float16)

    def test_configure_encoder_for_sdpa_enables_each_attention_module(self):
        model_load = _load_model_load_module()
        attentions = [SimpleNamespace(use_pytorch_sdpa=False) for _ in range(2)]
        encoder = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention) for attention in attentions])

        model_load.configure_encoder_for_sdpa(encoder)

        self.assertTrue(all(attention.use_pytorch_sdpa for attention in attentions))

    def test_fp16_encoder_wrapper_preserves_fp16_public_outputs(self):
        import torch
        import torch.nn as nn

        model_load = _load_model_load_module()

        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones((), dtype=torch.float16))

            def forward_for_export(self, audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len):
                return audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len

        class FakePromptKernel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones((), dtype=torch.float16))

            def forward(self, value):
                return value[..., :model_load.D_MODEL]

        wrapper = model_load.StreamingEncoderWrapper(FakeEncoder(), FakePromptKernel())
        outputs = wrapper(
            torch.zeros(1, 1, model_load.D_MODEL, dtype=torch.float16),
            torch.ones(1, dtype=torch.int64),
            torch.zeros(1, 1, 1, model_load.D_MODEL, dtype=torch.float16),
            torch.zeros(1, 1, model_load.D_MODEL, 1, dtype=torch.float16),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        )

        self.assertEqual(outputs[0].dtype, torch.float16)
        self.assertEqual(outputs[2].dtype, torch.float16)
        self.assertEqual(outputs[3].dtype, torch.float16)

if __name__ == "__main__":
    unittest.main()
