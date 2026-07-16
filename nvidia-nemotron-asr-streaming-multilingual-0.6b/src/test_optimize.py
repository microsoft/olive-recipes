import importlib.util
import json
import sys
import tempfile
import types
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nemotron_optimize", SRC_DIR / "optimize.py")
optimize = importlib.util.module_from_spec(spec)
original_sys_path = sys.path.copy()
try:
    spec.loader.exec_module(optimize)
finally:
    sys.path[:] = original_sys_path


def _load_export_tokenizer_module():
    script = SRC_DIR.parent / "scripts" / "export_tokenizer.py"
    spec = importlib.util.spec_from_file_location("nemotron_export_tokenizer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model_load_module():
    spec = importlib.util.spec_from_file_location("nemotron_model_load", SRC_DIR / "nemotron_model_load.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NemotronOptimizePlanTest(unittest.TestCase):
    def test_cpu_int8_keeps_quantized_encoder_config(self):
        plan = optimize.resolve_encoder_export_plan("cpu", "int8")

        self.assertEqual(plan.config_name, "nemotron_encoder_int8_cpu.json")
        self.assertEqual(plan.encoder_precision, "int8")
        self.assertEqual(tuple(plan.__dataclass_fields__), ("config_name", "encoder_precision"))


    def test_cpu_fp16_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fp16"):
            optimize.resolve_encoder_export_plan("cpu", "fp16")

    def test_nvtensorrtrtx_uses_fp16_opset23_encoder_config(self):
        plan = optimize.resolve_encoder_export_plan("NvTensorRtRtx", "int4")

        self.assertEqual(plan.config_name, "nemotron_encoder_fp16_trtrtx.json")
        self.assertEqual(plan.encoder_precision, "fp16")
        self.assertTrue(optimize._is_trt_rtx_execution_provider("trt-rtx"))

        config = json.loads((SRC_DIR / plan.config_name).read_text())
        self.assertEqual(
            config["systems"]["local_system"]["accelerators"][0]["execution_providers"],
            ["NvTensorRTRTXExecutionProvider"],
        )
        self.assertEqual(config["passes"]["convert"]["target_opset"], 23)
        self.assertNotIn("quantization", config["passes"])
        self.assertEqual(config["input_model"]["model_loader"], "encoder_fp16_model_loader")
        self.assertEqual(config["input_model"]["dummy_inputs_func"], "encoder_fp16_dummy_inputs")

    def test_nvtensorrtrtx_uses_fp16_decoder_and_joint_configs(self):
        decoder_name = optimize._decoder_config_name("NvTensorRtRtx")
        joint_name = optimize._joint_config_name("NvTensorRtRtx")
        decoder_config = json.loads((SRC_DIR / decoder_name).read_text())
        joint_config = json.loads((SRC_DIR / joint_name).read_text())

        self.assertEqual(decoder_name, "nemotron_decoder_fp16_trtrtx.json")
        self.assertEqual(joint_name, "nemotron_joint_fp16_trtrtx.json")
        self.assertEqual(decoder_config["input_model"]["model_loader"], "decoder_fp16_model_loader")
        self.assertEqual(decoder_config["input_model"]["dummy_inputs_func"], "decoder_fp16_dummy_inputs")
        self.assertEqual(joint_config["input_model"]["model_loader"], "joint_fp16_model_loader")
        self.assertEqual(joint_config["input_model"]["dummy_inputs_func"], "joint_fp16_dummy_inputs")

    def test_nvtensorrtrtx_omits_vad_from_generated_config(self):
        self.assertTrue(optimize.should_include_vad("cpu"))
        self.assertFalse(optimize.should_include_vad("NvTensorRtRtx"))
        self.assertFalse(optimize.should_include_vad("NvTensorRTRTXExecutionProvider"))

    def test_cuda_execution_provider_aliases_are_accepted(self):
        model_load = types.ModuleType("src.nemotron_model_load")
        model_load.MODEL_NAME = "nvidia/test-model"
        model_load.CHUNK_SIZE = 0.16

        for execution_provider in ("cuda", "CUDAExecutionProvider"):
            with self.subTest(execution_provider=execution_provider):
                with (
                    patch.dict(sys.modules, {"src.nemotron_model_load": model_load}),
                    patch.object(sys, "argv", ["optimize.py", "--execution-provider", execution_provider]),
                    patch.object(optimize, "run_olive_pipelines") as run_olive_pipelines,
                    patch.object(optimize, "run_tokenizer_export"),
                    patch.object(optimize, "generate_configs"),
                    patch.object(optimize, "download_silero_vad"),
                ):
                    optimize.main()

                self.assertEqual(
                    run_olive_pipelines.call_args.kwargs["execution_provider"],
                    execution_provider,
                )

    def test_readme_documents_trt_rtx_vad_skip(self):
        readme = (SRC_DIR / "README.md").read_text()

        self.assertIn(
            "**VAD** — downloads Silero VAD ONNX model (skipped for NvTensorRtRtx)",
            readme,
        )

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
