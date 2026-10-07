# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""CPU-only tests for evidence arithmetic and failure conditions."""
from __future__ import annotations

import copy
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evidence  # noqa: E402


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = evidence.load_requests(evidence.EVIDENCE)
        cls.result = evidence.calculate()

    def test_accepted_matrix_has_135_records_and_45_independent_worker_identities(self):
        evidence.validate_requests(self.records)
        groups = self.result["benchmark_groups"]
        self.assertEqual(len(groups), 45)
        self.assertTrue(all(group["independent_workers"] == 3 for group in groups))
        self.assertEqual(len({r["worker_id"] for r in self.records}), 45)

    def test_accepted_matrix_rejects_missing_or_duplicate_combinations(self):
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            evidence.validate_requests(self.records[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            evidence.validate_requests(self.records[:-1] + [self.records[0]])

    def test_accepted_matrix_rejects_capacity_and_input_hash_changes(self):
        wrong = copy.deepcopy(self.records)
        wrong[0]["effective_sequence_capacity"] += 256
        with self.assertRaisesRegex(ValueError, "Capacity mismatch"):
            evidence.validate_requests(wrong)
        wrong = copy.deepcopy(self.records)
        wrong[0]["input_token_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Input IDs differ"):
            evidence.validate_requests(wrong)

    def test_benchmark_summary_is_order_independent_and_keeps_request_kinds_separate(self):
        forward = evidence.benchmark_groups(self.records)
        backward = evidence.benchmark_groups(list(reversed(self.records)))
        self.assertEqual(forward, backward)
        self.assertEqual({g["request_kind"] for g in forward}, set(evidence.KINDS))

    def test_every_checkpoint_uses_current_bytes_not_lifetime_peaks(self):
        for rows in self.result["allocator"].values():
            for row in rows:
                self.assertEqual(row["accounted"] + row["residual"], row["nvml"])
            cleanup = next(r for r in rows if r["checkpoint"] == "H")
            self.assertEqual(cleanup["kv_live"], 0)
            self.assertEqual(cleanup["logits_live"], 0)
            self.assertGreater(cleanup["genai_unused"], 0)
            self.assertGreater(cleanup["genai_high_water_not_added"], 0)
        data = evidence.read_json(evidence.EVIDENCE / "allocator_checkpoints.json")
        original = next(p for p in data["cases"]["7k"]["checkpoints"] if p["checkpoint"] == "G")
        altered = copy.deepcopy(original)
        altered["model_session"]["MaxInUse"] *= 2
        altered["genai_device"]["MaxInUse"] *= 2
        self.assertEqual(evidence.allocation_row(original)["accounted"], evidence.allocation_row(altered)["accounted"])
        for values in self.result["lifetime_controls"].values():
            self.assertAlmostEqual(sum(values.values()), 115.09764862060547)
            self.assertEqual(values["genai_global_lifetime_extra_mib"], 11)
            self.assertEqual(values["persistent_after_shutdown_extra_mib"], 48)

    def test_allocator_rejects_requested_live_exceeding_block_live(self):
        with self.assertRaisesRegex(ValueError, "current allocator ordering"):
            evidence.allocator_stats({"RequestedInUse": 5, "InUse": 4, "TotalAllocated": 10, "MaxInUse": 8})

    def test_half_metric_handles_negative_values_zero_and_exponent_boundary(self):
        self.assertEqual(evidence.ordered_half(0x8000), evidence.ordered_half(0))
        self.assertEqual(abs(evidence.ordered_half(0x8001) - evidence.ordered_half(1)), 2)
        self.assertEqual(abs(evidence.ordered_half(0xA022) - evidence.ordered_half(0x9F66)), 188)
        self.assertEqual(abs(evidence.ordered_half(0x9E45) - evidence.ordered_half(0xA089)), 580)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            evidence.half_value(0x7C00)

    def test_histogram_preserves_mean_counts_and_signed_zero_byte_difference(self):
        metrics = evidence.pair_metrics([(0x3C00, 0x3C01, 2)], 2, 4)
        self.assertEqual(metrics["max_absolute_difference"], 2**-10)
        self.assertEqual(metrics["mean_absolute_difference"], 2**-11)
        self.assertEqual(metrics["max_ordered_fp16_steps"], 1)
        signed_zero = evidence.pair_metrics([(0, 0x8000, 1)], 0, 1)
        self.assertEqual(signed_zero["max_ordered_fp16_steps"], 0)
        self.assertFalse(signed_zero["byte_identical"])
        with self.assertRaisesRegex(ValueError, "every logit"):
            evidence.pair_metrics([(0x3C00, 0x3C01, 1)], 0, 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evidence.pair_metrics([(0x3C00, 0x3C01, 1), (0x3C00, 0x3C01, 1)], 0, 2)

    def test_candidate_numerical_results_and_exact_eight_byte_allocator_difference(self):
        for case, steps, changed in (("7k", 188, 22160), ("28k", 580, 22670)):
            result = self.result["logits"][case]
            self.assertEqual(result["max_absolute_difference"], 0.015625)
            self.assertEqual(result["max_ordered_fp16_steps"], steps)
            self.assertEqual(result["different_numeric_fp16_values"], changed)
            self.assertEqual(result["model_session_reservation_delta_bytes"], 8)
            self.assertTrue(result["tokens_match"])
            self.assertFalse(result["exact_preserving_on_tested_logits"])
            self.assertTrue(all(abs(v) < 0.01 for v in result["worst_ordered_step_values"]))

    def test_checksum_verification_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture.json").write_text("{}\n")
            (root / "SHA256SUMS").write_text("0" * 64 + "  fixture.json\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                evidence.verify_checksums(root)

    def test_initializer_bytes_are_recomputed_from_architecture_and_quantization(self):
        storage = self.result["initializer_storage"]
        self.assertEqual(storage["total_bytes"], 16752996536)
        self.assertEqual(storage["category_bytes"]["expert_int4_packed_bytes"], 14495514624)
        self.assertEqual(storage["category_bytes"]["embeddings_fp16"], 622329856)

    def test_qmoe_source_subtotal_and_layer_reuse(self):
        for case in ("7k", "28k"):
            result = self.result["qmoe"][case]
            self.assertEqual(result["additional_growth_layers_1_to_47_bytes"], 0)
            self.assertLess(result["unmodeled_metadata_and_alignment_mib"], 0.04)

    def test_sequential_retention_is_outside_stable_allocator_capacity(self):
        series = self.result["sequential"]
        self.assertEqual(series["growth_mib"], 86)
        self.assertAlmostEqual(series["slope_request_2_to_15_mib"], 6.210989010989011)
        self.assertEqual(series["matched_token_ids"], 960)
        self.assertTrue(series["model_live_constant"])
        self.assertTrue(series["model_held_constant"])
        self.assertTrue(series["genai_held_constant"])
        self.assertFalse(series["plateau_observed"])

    def test_portable_recipe_matches_executed_pass_parameters(self):
        recipe = evidence.read_json(ROOT / "cuda/kquant_fp16/config.json")
        provenance = evidence.read_json(evidence.EVIDENCE / "provenance.json")
        self.assertEqual(recipe["input_model"], provenance["build"]["recorded_recipe"]["input_model"])
        executed = provenance["build"]["executed_pass_lineage"]
        configured = recipe["passes"]["kquant"]
        for key, value in configured.items():
            if key != "type":
                self.assertEqual(value, executed[0]["pass_run_config"][key])
        self.assertEqual(executed[1]["parent_model_id"], executed[0]["model_id"])
        self.assertEqual(recipe["passes"]["mobius"]["precision"], executed[1]["pass_run_config"]["precision"])

    def test_corrected_runner_capacity_prompt_and_cli_are_importable_without_cuda(self):
        spec = importlib.util.spec_from_file_location("qwen_benchmark_copy", ROOT / "scripts/run_ort_llama_benchmark.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for label, (actual, capacity) in evidence.CONTEXTS.items():
            self.assertEqual(module.requested_sequence_capacity(label, 64), capacity)
            self.assertGreater(len(module.build_prompt(label)), actual)
            module.validate_input_capacity(range(actual), capacity, 64)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            module.validate_input_capacity(range(10), 12, 3)
        self.assertTrue(math.isfinite(module.token_metrics(1, [1, 2], [2, 3])["decode_tokens_per_second"]))

    def test_all_published_tables_match_the_committed_evidence(self):
        evidence.check_documents(ROOT, evidence.tables(self.result))


if __name__ == "__main__":
    unittest.main()
