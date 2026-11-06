import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fuse_bf16_norm_cast import rewrite


class FuseBFloat16NormCastTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "original.onnx"
        self.output = self.source.with_name("fused.onnx")

    def model(self, scale=None, input_type=TensorProto.BFLOAT16):
        if scale is None:
            scale = [0.125, -0.75, 0.0, -0.0]
        shape = [2, 4]
        # Empty names are valid; removal must not depend on node names.
        nodes = [
            helper.make_node("Cast", ["X"], ["X_fp32"], to=TensorProto.FLOAT),
            helper.make_node("SimplifiedLayerNormalization", ["X_fp32", "scale"], ["Y_fp32"],
                             epsilon=1e-5, stash_type=TensorProto.FLOAT),
            helper.make_node("Cast", ["Y_fp32"], ["Y"], to=TensorProto.BFLOAT16),
        ]
        graph = helper.make_graph(
            nodes, "norm", [helper.make_tensor_value_info("X", input_type, shape)],
            [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, shape)],
            [numpy_helper.from_array(np.array(scale, dtype=np.float32), "scale")],
            value_info=[
                helper.make_tensor_value_info("X_fp32", TensorProto.FLOAT, shape),
                helper.make_tensor_value_info("Y_fp32", TensorProto.FLOAT, shape),
            ],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    def save(self, model):
        self.source.write_bytes(model.SerializeToString())

    def reject(self, model, **kwargs):
        self.save(model)
        with self.assertRaises(ValueError):
            rewrite(self.source, self.output, expected_casts=2, **kwargs)
        self.assertFalse(self.output.exists())

    def test_complete_chain_and_signed_zero_scale(self):
        self.save(self.model())
        before = self.source.read_bytes()
        counts = rewrite(self.source, self.output, expected_casts=2)
        self.assertEqual((counts["norms_rewritten"], counts["casts_removed"]), (1, 2))
        fused = onnx.load(self.output)
        self.assertEqual(len(fused.graph.node), 1)
        norm = fused.graph.node[0]
        self.assertEqual(norm.input[0], "X")
        self.assertEqual(norm.output[0], "Y")
        self.assertEqual(fused.graph.initializer[0].data_type, TensorProto.BFLOAT16)
        bits = np.frombuffer(fused.graph.initializer[0].raw_data, dtype="<u2").astype("<u4") << 16
        expected = np.array([0.125, -0.75, 0.0, -0.0], dtype="<f4").view("<u4")
        np.testing.assert_array_equal(bits, expected)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(len(fused.graph.value_info), 0)

    def test_inexact_and_nonfinite_scales(self):
        for value in (1.0001, float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                self.reject(self.model([value, 1, 1, 1]))

    def test_types_and_missing_type_information(self):
        self.reject(self.model(input_type=TensorProto.FLOAT16))
        model = self.model()
        del model.graph.value_info[:]
        self.reject(model)

    def test_shared_and_exposed_intermediates(self):
        for name in ("X_fp32", "Y_fp32"):
            for exposed in (False, True):
                with self.subTest(name=name, exposed=exposed):
                    model = self.model()
                    if exposed:
                        model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 4]))
                    else:
                        model.graph.node.append(helper.make_node("Identity", [name], ["extra"]))
                        model.graph.output.append(helper.make_tensor_value_info("extra", TensorProto.FLOAT, [2, 4]))
                    self.reject(model)

    def test_dynamic_overridable_and_session_supplied_scale(self):
        for dynamic in (False, True):
            model = self.model()
            model.graph.input.append(helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4]))
            if dynamic:
                del model.graph.initializer[:]
            self.reject(model)
        self.reject(self.model(), excluded_initializers=["scale"])

    def test_shared_scale_is_not_mutated(self):
        model = self.model()
        model.graph.node.append(helper.make_node("Identity", ["scale"], ["original_scale"]))
        model.graph.output.append(helper.make_tensor_value_info("original_scale", TensorProto.FLOAT, [4]))
        self.save(model)
        original = model.graph.initializer[0].SerializeToString()
        rewrite(self.source, self.output, expected_casts=2)
        fused = onnx.load(self.output)
        scale = next(t for t in fused.graph.initializer if t.name == "scale")
        self.assertEqual(scale.SerializeToString(), original)
        self.assertEqual(len(fused.graph.initializer), 2)

    def test_auxiliary_output_and_stash_type(self):
        model = self.model()
        model.graph.node[1].output.append("inverse_stddev")
        self.reject(model)
        model = self.model()
        for attribute in model.graph.node[1].attribute:
            if attribute.name == "stash_type":
                attribute.i = TensorProto.DOUBLE
        self.reject(model)

    def test_subgraphs_are_explicitly_rejected(self):
        model = self.model()
        model.graph.node.append(helper.make_node("If", ["condition"], ["capture"],
                                                 then_branch=model.graph, else_branch=model.graph))
        self.reject(model)

    def test_count_mismatch_and_overwrite_are_rejected(self):
        self.save(self.model())
        with self.assertRaisesRegex(ValueError, "Expected 4"):
            rewrite(self.source, self.output, expected_casts=4)
        self.assertFalse(self.output.exists())
        self.output.write_bytes(b"do not overwrite")
        with self.assertRaises(ValueError):
            rewrite(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"do not overwrite")
        with self.assertRaises(ValueError):
            rewrite(self.source, self.source)

    def test_existing_name_collision(self):
        model = self.model()
        model.graph.initializer.append(numpy_helper.from_array(np.ones(4, dtype=np.float32), "scale_norm_cast_bf16"))
        self.save(model)
        rewrite(self.source, self.output, expected_casts=2)
        fused = onnx.load(self.output)
        self.assertEqual(fused.graph.node[0].input[1], "scale_norm_cast_bf16_")
        self.assertEqual(len({t.name for t in fused.graph.initializer}), len(fused.graph.initializer))

    def test_cli(self):
        self.save(self.model())
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("fuse_bf16_norm_cast.py")),
             "--source", str(self.source), "--output", str(self.output), "--expected-casts", "2"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(result.stdout)["casts_removed"], 2)

    def test_external_weights_are_preserved(self):
        model = self.model()
        data = self.source.with_suffix(".data")
        onnx.save_model(model, self.source, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=data.name, size_threshold=0)
        before = (self.source.read_bytes(), data.read_bytes(), data.stat().st_mtime_ns)
        rewrite(self.source, self.output, expected_casts=2)
        self.assertEqual(before, (self.source.read_bytes(), data.read_bytes(), data.stat().st_mtime_ns))
        self.assertEqual(onnx.load(self.output).graph.initializer[0].data_type, TensorProto.BFLOAT16)


if __name__ == "__main__":
    unittest.main()
