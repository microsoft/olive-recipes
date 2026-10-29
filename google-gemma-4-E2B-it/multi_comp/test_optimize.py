import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "gemma4_multi_comp_optimize", Path(__file__).with_name("optimize.py")
)
optimize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(optimize)


class Gemma4OptimizeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.recipe_dir = Path(temporary.name)
        directory_patch = patch.object(optimize, "RECIPE_DIR", self.recipe_dir)
        directory_patch.start()
        self.addCleanup(directory_patch.stop)
        subprocess_patch = patch.object(optimize.subprocess, "run")
        self.subprocess_run = subprocess_patch.start()
        self.addCleanup(subprocess_patch.stop)
        for ep in ("qnn", "ov"):
            self.write_config(
                ep, {"model_attributes": {"matmul_nbits_accuracy_level": 1}}
            )

    def write_config(self, ep, model_config):
        config = {"input_model": {"type": "CompositeModel", "config": model_config}}
        (self.recipe_dir / f"{ep}.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

    def invoke(self, ep):
        with (
            patch.object(sys, "argv", ["optimize.py", "--ep", ep]),
            patch("sys.stdout", new=io.StringIO()),
        ):
            optimize.main()

    def assert_pipeline(self, ep, accuracy_level):
        capture = [
            "capture-onnx-graph",
            "--model_name_or_path",
            "gemma4_quantized_hf",
            "--use_mobius_builder",
            "--precision",
            "fp32",
            "--output_path",
            "gemma4_onnx",
        ]
        if accuracy_level is not None:
            capture.extend(["--int4_accuracy_level", str(accuracy_level)])
        self.assertEqual(
            [call.args[0] for call in self.subprocess_run.call_args_list],
            [
                [
                    sys.executable,
                    "-m",
                    "olive",
                    "run",
                    "--config",
                    "gemma4_quantize.json",
                ],
                [sys.executable, "-m", "olive", *capture],
                [sys.executable, "-m", "olive", "run", "--config", f"{ep}.json"],
            ],
        )
        for call in self.subprocess_run.call_args_list:
            self.assertEqual(call.kwargs, {"cwd": self.recipe_dir, "check": True})

    def test_main_forwards_selected_recipe_accuracy_to_original_capture_command(self):
        for ep, alias in (
            ("qnn", "QNNExecutionProvider"),
            ("ov", "OpenVINOExecutionProvider"),
        ):
            for level in (0, 2, 4):
                with self.subTest(ep=ep, level=level):
                    self.subprocess_run.reset_mock()
                    self.write_config(
                        ep, {"model_attributes": {"matmul_nbits_accuracy_level": level}}
                    )
                    self.invoke(alias)
                    self.assert_pipeline(ep, level)

    def test_main_omits_capture_override_when_recipe_setting_is_absent_or_null(self):
        configs = (
            {},
            {"model_attributes": None},
            {"model_attributes": {}},
            {"model_attributes": {"matmul_nbits_accuracy_level": None}},
        )
        for ep in ("qnn", "ov"):
            for config in configs:
                with self.subTest(ep=ep, config=config):
                    self.subprocess_run.reset_mock()
                    self.write_config(ep, config)
                    self.invoke(ep)
                    self.assert_pipeline(ep, None)

    def test_main_rejects_invalid_accuracy_before_quantization(self):
        for level in (-1, 5, True, False, 4.0, "4", [], {}):
            with self.subTest(level=level):
                self.subprocess_run.reset_mock()
                self.write_config(
                    "qnn", {"model_attributes": {"matmul_nbits_accuracy_level": level}}
                )
                with (
                    patch("sys.stderr", new=io.StringIO()) as stderr,
                    self.assertRaises(SystemExit) as error,
                ):
                    self.invoke("qnn")
                self.assertEqual(error.exception.code, 2)
                self.assertIn(
                    "qnn.json: input_model.config.model_attributes.matmul_nbits_accuracy_level",
                    stderr.getvalue(),
                )
                self.subprocess_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
