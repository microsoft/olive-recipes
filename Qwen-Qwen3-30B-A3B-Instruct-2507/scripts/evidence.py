# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Verify and regenerate the committed evidence using only the standard library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
MIB = 1024**2
GIB = 1024**3
CONTEXTS = {2048: (1801, 2304), 4096: (3596, 4352), 8192: (7189, 8448),
            16384: (14376, 16640), 32768: (28747, 33024)}
VARIANTS = (("ort", "baseline", "ORT baseline"), ("ort", "device", "ORT Reserve"), ("llama_cpp", "n/a", "llama.cpp"))
KINDS = ("cold", "warm_1", "warm_2")
INT_FIELDS = {
    "context_label", "repetition", "request_index", "prompt_tokens", "completion_tokens",
    "max_output_tokens", "requested_sequence_capacity", "effective_sequence_capacity",
    "sample_count", "kv_cache_tensor_bytes",
}
BOOL_FIELDS = {"eos_included_in_completion_tokens", "original_config_integrity", "no_prefix_reuse_verified"}
FLOAT_FIELDS = {
    "ttft_ms", "request_ttft_ms", "prompt_processing_ms", "prefill_to_first_token_ms", "request_setup_ms",
    "decode_tokens_per_second", "model_load_ms", "tokenization_ms", "pre_load_vram_mib",
    "pre_request_vram_mib", "request_peak_vram_mib", "load_peak_vram_mib", "peak_vram_mib",
    "post_exit_vram_mib", "prefill_peak_vram_mib", "decode_peak_vram_mib", "inference_peak_vram_mib",
    "max_sample_gap_ms", "median_sample_gap_ms",
}
STATS = ("InUse", "RequestedInUse", "TotalAllocated", "MaxInUse")
FRESH_EXPORT = "fresh_export_mmlu.json"
Z95 = 1.96


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)

def verify_checksums(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    require(manifest.is_file(), "Evidence checksum manifest is missing")
    listed = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(Path(name).name == name and name != manifest.name, "Invalid checksum entry")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "Evidence file missing or symlinked")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Evidence checksum mismatch: {name}")
        require(name not in listed, "Duplicate checksum entry")
        listed.add(name)
    actual = {p.name for p in directory.iterdir() if p.is_file() and p != manifest}
    require(listed == actual, "Evidence checksum inventory is incomplete")


def half_value(bits: int) -> float:
    require(type(bits) is int and 0 <= bits <= 0xFFFF, "Invalid binary16 bit pattern")
    value = struct.unpack("<e", struct.pack("<H", bits))[0]
    require(math.isfinite(value), "Non-finite logits are not supported by this evidence")
    return value


def ordered_half(bits: int) -> int:
    half_value(bits)
    magnitude = bits & 0x7FFF
    return 0x8000 - magnitude if bits & 0x8000 else 0x8000 + magnitude


def pair_metrics(pairs: list[tuple[int, int, int]], equal_count: int, total: int) -> dict[str, Any]:
    require(type(equal_count) is int and equal_count >= 0, "Invalid equal-pair count")
    require(type(total) is int and total > 0, "Invalid logit count")
    require(len({(a, b) for a, b, _ in pairs}) == len(pairs), "Duplicate pair histogram rows")
    absolute_sum, step_sum = [], 0
    maximum_abs, maximum_steps, different_values = 0.0, 0, 0
    worst_step_values = None
    unequal_count = 0
    for a, b, count in pairs:
        require(type(count) is int and count > 0 and a != b, "Invalid unequal histogram row")
        difference = abs(half_value(a) - half_value(b))
        steps = abs(ordered_half(a) - ordered_half(b))
        if steps > maximum_steps:
            worst_step_values = [half_value(a), half_value(b)]
        unequal_count += count
        maximum_abs = max(maximum_abs, difference)
        maximum_steps = max(maximum_steps, steps)
        different_values += count if steps else 0
        absolute_sum.append(difference * count)
        step_sum += steps * count
    require(equal_count + unequal_count == total, "Histogram does not represent every logit")
    return {
        "logit_count": total,
        "different_bit_pairs": unequal_count,
        "different_numeric_fp16_values": different_values,
        "max_absolute_difference": maximum_abs,
        "mean_absolute_difference": math.fsum(absolute_sum) / total,
        "max_ordered_fp16_steps": maximum_steps,
        "mean_ordered_fp16_steps": step_sum / total,
        "byte_identical": unequal_count == 0,
        "worst_ordered_step_values": worst_step_values,
    }


def load_requests(directory: Path) -> list[dict[str, Any]]:
    with (directory / "accepted_requests.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        for field in INT_FIELDS:
            record[field] = int(record[field]) if record[field] else None
        for field in FLOAT_FIELDS:
            record[field] = float(record[field]) if record[field] else None
        for field in BOOL_FIELDS:
            require(record[field] in ("true", "false"), f"Invalid boolean {field}")
            record[field] = record[field] == "true"
    return records


def validate_requests(records: list[dict[str, Any]]) -> None:
    keys = [
        (r["context_label"], (r["runtime"], r["ort_allocator_mode"]), r["repetition"], r["request_kind"])
        for r in records
    ]
    expected = set(itertools.product(CONTEXTS, [(r, m) for r, m, _ in VARIANTS], range(1, 4), KINDS))
    require(len(keys) == len(set(keys)) == 135 and set(keys) == expected, "Accepted matrix is incomplete or duplicated")
    require(len({r["worker_id"] for r in records}) == 45, "Independent-worker identities differ")
    prompt_hashes: dict[int, set[str]] = {}
    for r in records:
        label = r["context_label"]
        prompt, capacity = CONTEXTS[label]
        identity = f"{label}/{r['runtime']}/{r['ort_allocator_mode']}/{r['repetition']}"
        require(r["worker_id"] == identity, "Worker identity mismatch")
        require(r["status"] == "complete" and r["request_index"] == KINDS.index(r["request_kind"]), "Invalid status or request ordinal")
        require(r["prompt_tokens"] == prompt, "Actual prompt-token count changed")
        require(r["requested_sequence_capacity"] == r["effective_sequence_capacity"] == capacity, "Capacity mismatch")
        require(prompt + r["max_output_tokens"] <= capacity, "Input and output do not fit")
        require(0 < r["completion_tokens"] <= r["max_output_tokens"] == 64, "Completion count mismatch")
        require(r["stop_reason"] in ("length", "eos"), "Unknown stop reason")
        require(r["eos_included_in_completion_tokens"] == (r["stop_reason"] == "eos"), "EOS count mismatch")
        if r["stop_reason"] == "length":
            require(r["completion_tokens"] == 64, "Length stop before requested limit")
        require(r["original_config_integrity"] and r["no_prefix_reuse_verified"], "Integrity/fresh-state check failed")
        require(r["pre_load_vram_mib"] == r["post_exit_vram_mib"] == 0, "Worker GPU boundary was not idle")
        for name in FLOAT_FIELDS:
            value = r[name]
            require(value is not None and math.isfinite(value) and value >= 0, f"Invalid accepted {name}")
        require(math.isclose(r["ttft_ms"], r["prompt_processing_ms"] + r["prefill_to_first_token_ms"], abs_tol=1e-6), "TTFT boundary mismatch")
        require(math.isclose(r["request_ttft_ms"], r["ttft_ms"] + r["request_setup_ms"], abs_tol=1e-6), "Request TTFT boundary mismatch")
        require(r["kv_cache_dtype"] == "float16", "Unexpected KV dtype")
        if r["runtime"] == "ort":
            require(r["kv_cache_tensor_bytes"] == 48 * 2 * 4 * capacity * 128 * 2, "ORT KV inspection differs from formula")
        prompt_hashes.setdefault(label, set()).add(r["input_token_ids_sha256"])
    require(all(len(hashes) == 1 for hashes in prompt_hashes.values()), "Input IDs differ within a tier")

def validate_accepted_provenance(provenance: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Check declarations against records; this does not re-observe historical hardware."""
    validate_requests(records)
    require(isinstance(provenance, dict), "Accepted provenance root is invalid")
    accepted = provenance.get("accepted")
    require(isinstance(accepted, dict), "Accepted provenance is missing")
    matrix = accepted.get("matrix")
    require(isinstance(matrix, dict), "Accepted provenance matrix is missing")

    def integer(mapping: dict[str, Any], key: str, expected: int) -> None:
        require(type(mapping.get(key)) is int and mapping[key] == expected,
                f"Accepted provenance {key} contradicts the records or documented conditions")

    require(accepted.get("profiled") is False, "Accepted provenance must declare unprofiled measurements")
    integer(accepted, "gpu_index", 0)
    integer(accepted, "record_count", len(records))
    workers = {r["worker_id"] for r in records}
    integer(accepted, "independent_workers", len(workers))

    exits = accepted.get("benchmark_exit_status")
    require(isinstance(exits, dict), "Accepted provenance exit statuses are missing")
    for key in ("benchmark_exit_code", "tee_exit_code", "launcher_exit_code"):
        integer(exits, key, 0)

    contexts = matrix.get("context_labels")
    actual_contexts = {r["context_label"] for r in records}
    require(isinstance(contexts, list) and all(type(v) is int for v in contexts)
            and len(contexts) == len(set(contexts)) and set(contexts) == actual_contexts,
            "Accepted provenance context_labels contradict the records")
    variants = matrix.get("variants")
    require(isinstance(variants, list) and all(
        isinstance(v, list) and len(v) == 2 and all(isinstance(part, str) for part in v)
        for v in variants), "Accepted provenance variants are missing or invalid")
    declared_variants = {tuple(v) for v in variants}
    actual_variants = {(r["runtime"], r["ort_allocator_mode"]) for r in records}
    require(len(variants) == len(declared_variants) and declared_variants == actual_variants,
            "Accepted provenance variants contradict the records")

    actual_repetitions = {r["repetition"] for r in records}
    integer(matrix, "worker_repetitions", len(actual_repetitions))
    expected_repetitions = set(range(1, matrix["worker_repetitions"] + 1))
    require(actual_repetitions == expected_repetitions, "Accepted provenance repetition numbering differs")
    for context, variant in itertools.product(actual_contexts, actual_variants):
        group = [r for r in records if r["context_label"] == context
                 and (r["runtime"], r["ort_allocator_mode"]) == variant]
        require({r["repetition"] for r in group} == expected_repetitions,
                "Accepted provenance repetition structure contradicts a record group")

    integer(matrix, "requests_per_worker", len(KINDS))
    for worker in workers:
        group = [r for r in records if r["worker_id"] == worker]
        require(len(group) == matrix["requests_per_worker"]
                and {(r["request_kind"], r["request_index"]) for r in group}
                == set(zip(KINDS, range(len(KINDS)))),
                "Accepted provenance cold/warm structure contradicts a logical worker")
    integer(matrix, "expected_workers", len(workers))
    integer(matrix, "expected_request_records", len(records))
    output_caps = {r["max_output_tokens"] for r in records}
    require(len(output_caps) == 1, "Record output limits differ")
    integer(matrix, "max_generated_ids", next(iter(output_caps)))

    capacities = matrix.get("requested_capacities")
    require(isinstance(capacities, dict) and set(capacities) == {str(v) for v in actual_contexts},
            "Accepted provenance requested_capacities are missing or incomplete")
    for context in actual_contexts:
        observed = {
            (r["requested_sequence_capacity"], r["effective_sequence_capacity"])
            for r in records if r["context_label"] == context
        }
        require(len(observed) == 1, "Record context capacities differ")
        requested, effective = next(iter(observed))
        require(type(capacities[str(context)]) is int
                and capacities[str(context)] == requested == effective,
                "Accepted provenance requested_capacities contradict the records")


def benchmark_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_requests(records)
    output = []
    for kind in KINDS:
        for label, (prompt, capacity) in CONTEXTS.items():
            for runtime, mode, display in VARIANTS:
                group = [r for r in records if r["request_kind"] == kind and r["context_label"] == label
                         and r["runtime"] == runtime and r["ort_allocator_mode"] == mode]
                require(len(group) == 3, "Summary does not contain three independent repetitions")
                summary = {
                    "request_kind": kind, "context_label": label, "prompt_tokens": prompt,
                    "capacity": capacity, "variant": display, "independent_workers": 3,
                    "output_count_stop": sorted({(r["completion_tokens"], r["stop_reason"]) for r in group}),
                }
                for field in ("peak_vram_mib", "request_peak_vram_mib", "ttft_ms", "decode_tokens_per_second"):
                    values = [r[field] for r in group]
                    summary[field] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
                output.append(summary)
    return output


def allocator_stats(stats: dict[str, int] | None) -> dict[str, int]:
    if stats is None:
        return dict.fromkeys(STATS, 0)
    require(all(type(stats.get(field)) is int for field in STATS), "Missing allocator statistic")
    require(0 <= stats["RequestedInUse"] <= stats["InUse"] <= stats["TotalAllocated"], "Invalid current allocator ordering")
    require(stats["MaxInUse"] >= stats["InUse"], "High-water below current live")
    return stats

def initializer_storage(provenance: dict[str, Any]) -> dict[str, Any]:
    storage = provenance["artifacts"]["initializer_storage"]
    a = storage["architecture"]
    layers, hidden, inter = a["num_hidden_layers"], a["hidden_size"], a["moe_intermediate_size"]
    experts, group = a["num_local_experts"], a["group_size"]
    q_width, kv_width = a["num_attention_heads"] * a["head_dim"], a["num_key_value_heads"] * a["head_dim"]
    attention_elements = layers * (2 * hidden * q_width + 2 * hidden * kv_width)
    formulas = {
        "expert_int4_packed_bytes": layers * experts * 3 * hidden * inter // 2,
        "expert_scales_fp16": layers * experts * (2 * inter * (hidden // group) + hidden * (inter // group)) * 2,
        "attention_int4_packed_bytes": attention_elements // 2,
        "attention_scales_fp16": attention_elements // group * 2,
        "embeddings_fp16": a["vocab_size"] * hidden * 2,
        "lm_head_fp16": a["vocab_size"] * hidden * 2,
        "router_fp16": layers * hidden * experts * 2,
        "norms_fp16": (layers * (2 * hidden + 2 * a["head_dim"]) + hidden) * 2,
        "rotary_cache_fp16": 2 * a["max_position_embeddings"] * (a["head_dim"] // 2) * 2,
        "graph_constants_int64": storage["categories"]["graph_constants_int64"]["bytes"],
    }
    require(set(formulas) == set(storage["categories"]), "Unexpected initializer category")
    require(sum(v["tensors"] for v in storage["categories"].values()) == storage["initializers"] == 828,
            "Initializer count changed")
    for name, expected in formulas.items():
        require(expected == storage["categories"][name]["bytes"], f"Initializer byte formula differs: {name}")
    require(sum(formulas.values()) == storage["total_bytes"], "Initializer storage sum mismatch")
    return {"category_bytes": formulas, "total_bytes": sum(formulas.values()), "total_mib": sum(formulas.values()) / MIB}


def allocation_row(point: dict[str, Any]) -> dict[str, Any]:
    model = allocator_stats(point["model_session"])
    genai = allocator_stats(point["genai_device"])
    other = genai["RequestedInUse"] - point["kv_logical_live_bytes"] - point["logits_logical_live_bytes"]
    require(other >= 0, "Logical payload exceeds GenAI requested bytes")
    components = {
        "model_live": model["InUse"],
        "model_unused": model["TotalAllocated"] - model["InUse"],
        "kv_live": point["kv_logical_live_bytes"],
        "logits_live": point["logits_logical_live_bytes"],
        "genai_other_requested": other,
        "genai_live_padding": genai["InUse"] - genai["RequestedInUse"],
        "genai_unused": genai["TotalAllocated"] - genai["InUse"],
        "auxiliary_held": point["auxiliary_allocator_held_bytes"],
        "cuda_baseline": point["cuda_initialization_process_baseline_bytes"],
    }
    require(all(type(v) is int and v >= 0 for v in components.values()), "Invalid memory component")
    accounted = sum(components.values())
    require(accounted == model["TotalAllocated"] + genai["TotalAllocated"] + components["auxiliary_held"] + components["cuda_baseline"], "Allocator sum mismatch")
    nvml = point["nvml_process_bytes"]
    residual = nvml - accounted
    return {
        "checkpoint": point["checkpoint"], **components, "accounted": accounted,
        "nvml": nvml, "residual": residual,
        "residual_percent": 100 * residual / nvml if nvml else None,
        "model_high_water_not_added": model["MaxInUse"],
        "genai_high_water_not_added": genai["MaxInUse"],
    }


def allocation_cases(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for case, item in data["cases"].items():
        require(item["kv_dtype"] == item["logits_dtype"] == "float16", "Unexpected tensor dtype")
        kv = item["layers"] * 2 * item["kv_heads"] * item["sequence_capacity"] * item["head_dim"] * 2
        require(len(item["kv_tensors"]) == 96 and sum(t["bytes"] for t in item["kv_tensors"]) == kv, "KV tensor metadata mismatch")
        for tensor in item["kv_tensors"]:
            require(tensor["shape"] == [1, 4, item["sequence_capacity"], 128] and tensor["dtype"] == 10, "KV shape/type changed")
            require(tensor["bytes"] == math.prod(tensor["shape"]) * 2, "KV tensor byte count mismatch")
        require(item["raw_logits"]["shape"] == [1, item["prompt_tokens"], 151936], "Raw logits shape changed")
        require(item["raw_logits"]["bytes"] == item["prompt_tokens"] * 151936 * 2, "Raw logits byte count mismatch")
        rows = [allocation_row(p) for p in item["checkpoints"]]
        require(len({r["checkpoint"] for r in rows}) == len(rows) == 12, "Lifecycle coverage changed")
        output[case] = rows
    return output

def lifetime_controls(rows: list[dict[str, Any]]) -> dict[str, float]:
    points = {r["checkpoint"]: r for r in rows}
    h, model_gone, shutdown = (points[name] for name in ("H", "I_model_destroyed", "J_genai_shutdown"))
    return {
        "model_lifetime_extra_mib": (h["nvml"] - model_gone["nvml"] - h["model_live"] - h["model_unused"]) / MIB,
        "genai_global_lifetime_extra_mib": (model_gone["nvml"] - shutdown["nvml"] - model_gone["genai_unused"]) / MIB,
        "persistent_after_shutdown_extra_mib": (shutdown["nvml"] - shutdown["cuda_baseline"]) / MIB,
    }


def logits_cases(data: dict[str, Any], directory: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for case, item in data["cases"].items():
        file = Path(item["pairs_file"])
        require(file.name == str(file), "Pair file must be a relative filename")
        with (directory / file).open(newline="") as stream:
            pairs = [(int(r["original_fp16_bits"]), int(r["candidate_fp16_bits"]), int(r["count"])) for r in csv.DictReader(stream)]
        metrics = pair_metrics(pairs, item["equal_bit_pair_count"], item["logit_count"])
        require(metrics["different_bit_pairs"] == item["unequal_pair_count"], "Unequal-pair count changed")
        a, b, expected = item["original_output_ids"], item["candidate_output_ids"], item["accepted_output_ids"]
        require(len(a) == len(b) == len(expected) == 64, "Candidate completion length changed")
        metrics["tokens_match"] = a == b == expected
        metrics["original_peak"] = item["original_sampled_device_peak_bytes"] / MIB
        metrics["candidate_peak"] = item["candidate_sampled_device_peak_bytes"] / MIB
        metrics["saving_bytes"] = item["original_sampled_device_peak_bytes"] - item["candidate_sampled_device_peak_bytes"]
        metrics["saving_mib"] = metrics["saving_bytes"] / MIB
        metrics["exact_preserving_on_tested_logits"] = metrics["byte_identical"] and metrics["tokens_match"]
        metrics["model_session_reservation_delta_bytes"] = (
            item["candidate_model_session_held_bytes"] - item["original_model_session_held_bytes"]
        )
        require(metrics["model_session_reservation_delta_bytes"] == 8,
                "Recorded candidate's eight-byte model-session difference changed")
        output[case] = metrics
    return output


def qmoe_cases(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for case, item in data["cases"].items():
        layers = item["layers"]
        require(len(layers) == 48 and [r["layer"] for r in layers] == list(range(48)), "Missing QMoE layers")
        require(item["memory_annotated_nodes"] == 873 and item["model_arena_nodes"] == 871, "Profile event coverage differs")
        first = layers[0]
        for layer in layers:
            require(layer["peak_after_bytes"] == first["peak_after_bytes"], "QMoE high-water accumulated across layers")
            require(layer["held_after_bytes"] == first["held_after_bytes"], "QMoE capacity accumulated across layers")
            require(layer["profile_live_delta_bytes"] == 0, "Net QMoE live growth observed")
        require(all(r["profile_held_delta_bytes"] == 0 for r in layers[1:]), "Later QMoE layers grew reservation")
        n = item["prompt_tokens"] * item["top_k"]
        components = {
            "overlapped_inputs": n * item["hidden"] * 2,
            "overlapped_outputs": n * item["hidden"] * 2,
            "dedicated_fc1": n * item["intermediate"] * 2,
            "split_k2_swiglu_partials": 2 * (n * item["intermediate"] * 2) * 4,
            "routing_lower_bound": n * 4 * 2 + item["experts"] * item["prompt_tokens"] * 4 + (item["experts"] + 1) * 8,
            "outer_metadata": n * 12,
            "alpha_pointer_arrays": 2 * item["experts"] * 8,
        }
        require(components == item["component_bytes_lower_bound"], "Workspace formula inputs changed")
        calculated = sum(components.values())
        observed = first["peak_after_bytes"] - first["peak_before_bytes"]
        require(0 <= observed - calculated < 0.04 * MIB, "Source subtotal and measured peak disagree")
        output[case] = {
            "calculated_lower_bound_mib": calculated / MIB,
            "observed_peak_jump_mib": observed / MIB,
            "unmodeled_metadata_and_alignment_mib": (observed - calculated) / MIB,
            "reservation_jump_mib": (first["held_after_bytes"] - first["held_before_bytes"]) / MIB,
            "additional_growth_layers_1_to_47_bytes": sum(r["profile_held_delta_bytes"] for r in layers[1:]),
        }
    return output


def sequential_summary(data: dict[str, Any], expected_ids: list[int]) -> dict[str, Any]:
    requests = data["requests"]
    require(data["request_count"] == len(requests) == 15 and data["model_loads"] == 1, "Wrong sequential experiment size")
    require(data["prompt_tokens"] == 7189 and data["sequence_capacity"] == 8448, "Wrong sequential shape")
    require(data["do_sample"] is False and data["profiling_enabled"] is False, "Sequential configuration changed")
    require(len({r["input_token_ids_sha256"] for r in requests}) == 1, "Sequential prompts differ")
    rows = []
    for index, request in enumerate(requests, 1):
        require(request["request_index"] == index and request["state_tokens_before"] == 0, "Fresh-state identity changed")
        require(request["completion_tokens"] == 64 and request["state_tokens_after"] == 7189 + 64, "Sequential token count mismatch")
        require(request["output_ids"] == expected_ids, "Sequential output IDs differ from accepted baseline")
        require(len(request["checkpoints"]) == 5, "Before/after checkpoint is missing")
        for point in request["checkpoints"].values():
            allocator_stats(point["model_session"])
            allocator_stats(point["genai_device"])
        point = request["checkpoints"]["after_cleanup"]
        model, genai = point["model_session"], point["genai_device"]
        require(genai["InUse"] == genai["RequestedInUse"] == 0, "GenAI live allocations survive request cleanup")
        rows.append({
            "request_index": index, "process_mib": point["nvml_process_bytes"] / MIB,
            "device_mib": point["nvml_device_used_bytes"] / MIB,
            "model_live_mib": model["InUse"] / MIB,
            "model_requested_mib": model["RequestedInUse"] / MIB,
            "model_held_mib": model["TotalAllocated"] / MIB,
            "genai_live_bytes": genai["InUse"], "genai_requested_bytes": genai["RequestedInUse"],
            "genai_held_mib": genai["TotalAllocated"] / MIB,
            "tokens_match": True,
        })
    require(data["post_exit_device_used_bytes"] == data["post_exit_gpu_process_count"] == 0, "GPU cleanup failed")
    x, y = list(range(2, 16)), [r["process_mib"] for r in rows[1:]]
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / sum((a - mean_x)**2 for a in x)
    return {
        "rows": rows, "growth_mib": rows[-1]["process_mib"] - rows[0]["process_mib"],
        "slope_request_2_to_15_mib": slope,
        "model_live_constant": len({r["model_live_mib"] for r in rows}) == 1,
        "model_held_constant": len({r["model_held_mib"] for r in rows}) == 1,
        "genai_held_constant": len({r["genai_held_mib"] for r in rows}) == 1,
        "plateau_observed": len({r["process_mib"] for r in rows[1:]}) == 1,
        "matched_token_ids": sum(len(r["output_ids"]) for r in requests),
    }


def varint_size(value: int) -> int:
    require(type(value) is int and value >= 0, "Invalid protobuf length")
    size = 1
    while value >= 0x80:
        value >>= 7
        size += 1
    return size


def exact_mcnemar_p(only_first: int, only_second: int) -> float:
    """Two-sided exact binomial (p = 0.5) p-value for the discordant pairs, in exact integer arithmetic."""
    discordant = only_first + only_second
    smaller = min(only_first, only_second)
    if discordant == 0 or 2 * smaller == discordant:
        return 1.0
    tail = Fraction(sum(math.comb(discordant, k) for k in range(smaller + 1)), 2**discordant)
    return min(1.0, float(2 * tail))


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def paired_statistics(entry: dict[str, Any], questions: int) -> dict[str, Any]:
    keys = ("both_right", "only_first_right", "only_second_right", "both_wrong")
    require(all(type(entry[key]) is int and entry[key] >= 0 for key in keys), "Paired counts must be non-negative integers")
    require(sum(entry[key] for key in keys) == questions, "Paired counts do not sum to the question total")
    both, first, second, _ = (entry[key] for key in keys)
    mean = (first - second) / questions
    variance = ((first + second) - questions * mean * mean) / (questions - 1)
    error = math.sqrt(max(variance, 0.0) / questions)
    return {
        "first_correct": both + first, "second_correct": both + second,
        "delta_points": 100 * mean, "ci95_points": [100 * (mean - Z95 * error), 100 * (mean + Z95 * error)],
        "mcnemar_exact_p": exact_mcnemar_p(first, second),
    }


def artifact_relationships(record: dict[str, Any], provenance: dict[str, Any]) -> dict[str, int]:
    locked = provenance["artifacts"]
    artifacts = record["artifacts"]
    data, graph, config = (artifacts[name] for name in ("model.onnx.data", "model.onnx", "genai_config.json"))
    for entry in (data, graph, config):
        for key in ("fresh_sha256", "archived_sha256"):
            require(len(entry[key]) == 64 and set(entry[key]) <= set("0123456789abcdef"), "Invalid SHA-256 value")
    require(data["archived_bytes"] == locked["original_external_data"]["bytes"]
            and data["archived_sha256"] == locked["original_external_data"]["sha256"], "Archived external data differs from the provenance record")
    require(graph["archived_bytes"] == locked["original_graph"]["bytes"]
            and graph["archived_sha256"] == locked["original_graph"]["sha256"], "Archived graph differs from the provenance record")
    require(config["archived_sha256"] == locked["original_genai_config_sha256"], "Archived genai_config differs from the provenance record")
    require(data["fresh_bytes"] == data["archived_bytes"] and data["fresh_sha256"] == data["archived_sha256"],
            "Fresh external data is not byte-identical to the archived file")
    compare = record["graph_comparison"]
    require(compare["differing_text_format_lines"] == 1 and compare["differing_field"] == "graph.name",
            "The documented graph difference is not limited to graph.name")
    require(compare["op_type_histogram_identical"] is True and compare["initializer_names_identical"] is True,
            "Graph structure differs beyond graph.name")
    require(graph["fresh_sha256"] != graph["archived_sha256"], "The documented model.onnx exception requires differing hashes")
    fresh_chars, archived_chars = compare["graph_name_chars_fresh"], compare["graph_name_chars_archived"]
    prefix_delta = varint_size(fresh_chars) - varint_size(archived_chars)
    size_delta = graph["fresh_bytes"] - graph["archived_bytes"]
    require(size_delta == (fresh_chars - archived_chars) + prefix_delta,
            "model.onnx size difference is not explained by the graph.name length difference")
    require(config["json_content_identical"] is True, "genai_config.json content differs")
    return {"graph_size_delta": size_delta, "graph_name_char_delta": fresh_chars - archived_chars, "length_prefix_delta": prefix_delta}


def fresh_export_summary(record: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    require(record["schema"] == "fresh-export-mmlu/2", "Unsupported fresh-export schema")
    added = record["olive_run"]["packages_added_before_attempt_2"]
    require(added.get("requests") == record["environment"]["requests"] and len(added) > 1,
            "The recorded requests version or its added transitive dependencies are inconsistent")
    mmlu = record["mmlu"]
    questions = mmlu["questions"]
    require(type(questions) is int and questions > 1, "Invalid MMLU question total")
    require("test split" in mmlu["protocol"] and "limit 200 per subject" in mmlu["protocol"], "MMLU protocol changed")
    captured = mmlu["captured_statistics"]
    require(captured["ci95_seed"] is None, "The paired CI is deterministic; no seed is expected")
    correct: dict[str, int] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for name, entry in mmlu["paired_counts"].items():
        stats = paired_statistics(entry, questions)
        claimed = captured["comparisons"][name]
        require(close(stats["delta_points"], claimed["delta_points"]), f"Captured delta does not match the paired counts: {name}")
        require(all(close(a, b) for a, b in zip(stats["ci95_points"], claimed["ci95_points"])),
                f"Captured 95% CI does not match the paired counts: {name}")
        require(close(stats["mcnemar_exact_p"], claimed["mcnemar_exact_p"]), f"Captured exact McNemar p does not match the paired counts: {name}")
        for side, count in ((entry["first"], stats["first_correct"]), (entry["second"], stats["second_correct"])):
            require(correct.setdefault(side, count) == count, f"Correct counts differ between comparisons: {side}")
        comparisons[name] = {**entry, **stats}
    require(set(correct) == set(captured["accuracy"]) == set(captured["stderr"]), "Captured accuracy sides do not match the paired counts")
    accuracy = {side: count / questions for side, count in correct.items()}
    for side, value in accuracy.items():
        require(close(captured["accuracy"][side], value, 1e-12), f"Captured accuracy does not match the paired counts: {side}")
        require(close(captured["stderr"][side], math.sqrt(value * (1 - value) / (questions - 1)), 1e-12),
                f"Captured stderr does not match the paired counts: {side}")
    return {"record": record, "questions": questions, "correct": correct, "accuracy": accuracy, "comparisons": comparisons,
            "artifacts": artifact_relationships(record, provenance)}


def calculate(directory: Path = EVIDENCE) -> dict[str, Any]:
    verify_checksums(directory)
    provenance = read_json(directory / "provenance.json")
    requests = load_requests(directory)
    validate_accepted_provenance(provenance, requests)
    weights = initializer_storage(provenance)
    groups = benchmark_groups(requests)
    allocation = allocation_cases(read_json(directory / "allocator_checkpoints.json"))
    logits_source = read_json(directory / "logits_pairs.json")
    logits = logits_cases(logits_source, directory)
    qmoe = qmoe_cases(read_json(directory / "qmoe_workspace.json"))
    sequential = sequential_summary(read_json(directory / "sequential_requests.json"), logits_source["cases"]["7k"]["accepted_output_ids"])
    comparison = []
    for case, tokens in (("7k", 7189), ("28k", 28747)):
        row = next(g for g in groups if g["request_kind"] == "cold" and g["prompt_tokens"] == tokens and g["variant"] == "llama.cpp")
        candidate = logits[case]["candidate_peak"]
        llama = row["peak_vram_mib"]["median"]
        checkpoints = {r["checkpoint"]: r for r in allocation[case]}
        source = read_json(directory / "allocator_checkpoints.json")["cases"][case]
        points = {r["checkpoint"]: r for r in source["checkpoints"]}
        delta = points["E"]["genai_device"]["TotalAllocated"] - points["C"]["genai_device"]["TotalAllocated"]
        require(delta == logits[case]["saving_bytes"], "GenAI reservation growth does not match candidate delta")
        accepted_peak = next(
            g for g in groups if g["request_kind"] == "cold" and g["prompt_tokens"] == tokens and g["variant"] == "ORT Reserve"
        )["peak_vram_mib"]["median"]
        require(points["G"]["nvml_device_used_bytes"] - checkpoints["G"]["nvml"] == int(8.875 * MIB),
                "Measured device/process counter offset changed")
        require(round(points["G"]["nvml_device_used_bytes"] / MIB, 2) == accepted_peak,
                "Fresh diagnostic does not reproduce the accepted rounded device peak")
        comparison.append({"prompt_tokens": tokens, "candidate_mib": candidate, "llama_median_mib": llama,
                           "provisional_gap_mib": candidate - llama})
    return {"benchmark_groups": groups, "initializer_storage": weights, "allocator": allocation,
            "lifetime_controls": {case: lifetime_controls(rows) for case, rows in allocation.items()}, "logits": logits,
            "qmoe": qmoe, "sequential": sequential, "comparison": comparison,
            "fresh_export": fresh_export_summary(read_json(directory / FRESH_EXPORT), provenance)}


def interval(metric: dict[str, float]) -> str:
    return f"{metric['median']:,.2f} [{metric['min']:,.2f}, {metric['max']:,.2f}]"


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def fresh_export_tables(summary: dict[str, Any]) -> dict[str, str]:
    record = summary["record"]
    install, run, smoke = record["documented_install"], record["olive_run"], record["genai_smoke"]
    first, second = run["attempt_1_documented"], run["attempt_2"]
    relations, compare = summary["artifacts"], record["graph_comparison"]
    steps = [
        "| Step | Result |",
        "|---|---|",
        f"| `pip install -r cuda/requirements.txt` in a clean environment | exit {install['exit']}, {install['wall_s']} s |",
        f"| `olive run --config cuda/kquant_fp16/config.json` as documented | exit {first['exit']} after {first['wall_s']} s: {first['error']} |",
        f"| The same command after adding `requests=={record['environment']['requests']}` and its transitive dependencies | exit {second['exit']}, {second['wall_s']} s "
        f"(KQuant pass {second['kquant_pass_s']:.1f} s, MobiusBuilder pass {second['mobius_pass_s']:.1f} s); "
        f"peak GPU 0 memory {second['peak_gpu_memory_mib']:,} MiB |",
        f"| ORT GenAI smoke test on the fresh export | `{smoke['output_text']}`; the 7,189-token greedy run matched "
        f"{smoke['tokens_7189_ids_equal_of_64']}/64 accepted IDs |",
    ]
    data, graph, config = (record["artifacts"][name] for name in ("model.onnx.data", "model.onnx", "genai_config.json"))
    digest = f"{data['fresh_sha256'][:8]}...{data['fresh_sha256'][-4:]}"
    files = [
        "| File | Fresh bytes | Archived bytes | SHA-256 |",
        "|---|---:|---:|---|",
        f"| `model.onnx.data` | {data['fresh_bytes']:,} | {data['archived_bytes']:,} | identical (`{digest}`) |",
        f"| `model.onnx` | {graph['fresh_bytes']:,} | {graph['archived_bytes']:,} | differs only in `graph.name` "
        f"({compare['graph_name_chars_fresh']} versus {compare['graph_name_chars_archived']} characters: "
        f"{relations['graph_name_char_delta']:+d} characters and {relations['length_prefix_delta']:+d} length-prefix byte "
        f"= {relations['graph_size_delta']:+d} bytes) |",
        f"| `genai_config.json` | {config['fresh_bytes']:,} | {config['archived_bytes']:,} | differs in raw bytes; JSON content identical |",
    ]
    structure = (f"The op-type histogram, node count ({compare['nodes']:,}), initializer count ({compare['initializers']:,}), "
                 f"initializer names and input/output name counts ({compare['input_names']} / {compare['output_names']}) are identical.")
    sections = {"fresh-export": "\n".join(steps + [""] + files + ["", structure])}
    mmlu, questions = record["mmlu"], summary["questions"]
    stats = mmlu["captured_statistics"]
    names = {"onnx_fresh": "ONNX fresh export", "torch_bf16": "Torch bf16 (previously saved baseline, not rerun)",
             "onnx_archived": "ONNX archived artifact"}
    lines = [f"| Side | Correct / {questions:,} | Accuracy (stderr, points) |", "|---|---:|---:|"]
    for side in ("onnx_fresh", "torch_bf16", "onnx_archived"):
        lines.append(f"| {names[side]} | {summary['correct'][side]:,} | {percent(summary['accuracy'][side])} ({100 * stats['stderr'][side]:.2f}) |")
    lines.extend([
        "",
        "| Paired comparison | Both right | Only first right | Only second right | Both wrong | Delta (points) | 95% CI (points) | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    labels = {"onnx_fresh_vs_torch_bf16": "ONNX fresh minus Torch bf16", "onnx_fresh_vs_onnx_archived": "ONNX fresh versus archived ONNX"}
    for name, row in summary["comparisons"].items():
        low, high = row["ci95_points"]
        lines.append(f"| {labels[name]} | {row['both_right']:,} | {row['only_first_right']:,} | {row['only_second_right']:,} | "
                     f"{row['both_wrong']:,} | {row['delta_points']:+.2f} | {low:.2f} to {high:.2f} | {row['mcnemar_exact_p']:.2g} |")
    extra = mmlu["captured_not_recomputable"]
    over = extra["prompts_over_512_tokens"]
    lines.extend([
        "",
        f"- Protocol: {mmlu['protocol']}.",
        f"- CI method: {stats['ci95_method']}",
        f"- Exact McNemar: {stats['mcnemar_method']}.",
        "- Captured, not recomputable from the compact evidence: both sides chose the same option for "
        f"{percent(extra['same_option_fraction_onnx_fresh_vs_torch_bf16'])} of questions, and {over['count']} of {over['of']:,} prompts "
        f"exceed 512 tokens ({over['note']}).",
    ])
    sections["mmlu"] = "\n".join(lines)
    main_row = summary["comparisons"]["onnx_fresh_vs_torch_bf16"]
    low, high = main_row["ci95_points"]
    sections["mmlu-summary"] = "\n".join([
        f"| MMLU sanity check (test split, up to 200 per subject, {questions:,} questions) | Accuracy |",
        "|---|---:|",
        f"| ONNX fresh export | {percent(summary['accuracy']['onnx_fresh'])} |",
        f"| Torch bf16 (previously saved baseline, not rerun) | {percent(summary['accuracy']['torch_bf16'])} |",
        f"| Paired difference (ONNX minus Torch) | {main_row['delta_points']:+.2f} points (95% CI {low:.2f} to {high:.2f}) |",
    ])
    return sections


def tables(result: dict[str, Any]) -> dict[str, str]:
    sections = {}
    weights = [
        "| Initializer storage category | MiB |",
        "|---|---:|",
    ]
    for name, value in result["initializer_storage"]["category_bytes"].items():
        weights.append(f"| {name} | {value/MIB:,.6f} |")
    weights.append(f"| **Total** | **{result['initializer_storage']['total_mib']:,.6f}** |")
    sections["weights"] = "\n".join(weights)
    accepted = []
    for kind in KINDS:
        accepted.extend([
            f"### {kind}", "",
            "| Prompt tokens (label) | Variant | Outputs / stop | Peak device MiB | TTFT ms | Decode IDs/s |",
            "|---:|---|---|---:|---:|---:|",
        ])
        for group in result["benchmark_groups"]:
            if group["request_kind"] == kind:
                accepted.append(
                    f"| {group['prompt_tokens']:,} ({group['context_label']:,}) | {group['variant']} | "
                    + ", ".join(f"{count}/{stop}" for count, stop in group["output_count_stop"])
                    + " | " + " | ".join(interval(group[key]) for key in ("peak_vram_mib", "ttft_ms", "decode_tokens_per_second")) + " |"
                )
        accepted.append("")
    sections["accepted"] = "\n".join(accepted).strip()
    headline = [
        "| Sampled device-used peak | 7,189 prompt tokens | 28,747 prompt tokens |",
        "|---|---:|---:|",
    ]
    for variant in ("ORT baseline", "ORT Reserve", "llama.cpp"):
        values = [
            next(g for g in result["benchmark_groups"] if g["request_kind"] == "cold" and g["prompt_tokens"] == n and g["variant"] == variant)["peak_vram_mib"]["median"]
            for n in (7189, 28747)
        ]
        headline.append(f"| {variant}, accepted median | {values[0]:,.2f} MiB | {values[1]:,.2f} MiB |")
    headline.append(
        f"| Last-row candidate, diagnostic n=1 | {result['logits']['7k']['candidate_peak']:,.2f} MiB | "
        f"{result['logits']['28k']['candidate_peak']:,.2f} MiB |"
    )
    sections["headline"] = "\n".join(headline)
    for case, rows in result["allocator"].items():
        lines = [
            "| Point | Main live | Main unused | KV | Logits | GenAI other + live padding | GenAI unused | Accounted incl. CUDA baseline | NVML process | Residual (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            if r["checkpoint"] not in tuple("ABCDEFGH"):
                continue
            values = [r[key] / MIB for key in ("model_live", "model_unused", "kv_live", "logits_live")]
            values.append((r["genai_other_requested"] + r["genai_live_padding"]) / MIB)
            values.extend(r[key] / MIB for key in ("genai_unused", "accounted", "nvml"))
            percent = "n/a" if r["residual_percent"] is None else f"{r['residual_percent']:.4f}%"
            lines.append(f"| {r['checkpoint']} | " + " | ".join(f"{v:,.2f}" for v in values)
                         + f" | {r['residual']/MIB:.2f} ({percent}) |")
        sections[f"allocator-{case}"] = "\n".join(lines)
    lines = [
        "| Case | Calculated lower bound MiB | Observed QMoE peak jump MiB | Unmodeled small metadata MiB | Reservation growth MiB | Layers 1-47 extra reservation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case, r in result["qmoe"].items():
        lines.append(f"| {case} | {r['calculated_lower_bound_mib']:.2f} | {r['observed_peak_jump_mib']:.2f} | "
                     f"{r['unmodeled_metadata_and_alignment_mib']:.6f} | {r['reservation_jump_mib']:.2f} | "
                     f"{r['additional_growth_layers_1_to_47_bytes']} bytes |")
    sections["qmoe"] = "\n".join(lines)
    lines = [
        "| Case | Original MiB | Candidate MiB | Saved MiB | Max absolute diff | Mean absolute diff | Max ordered FP16 steps | Differing FP16 values | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case, r in result["logits"].items():
        lines.append(f"| {case} | {r['original_peak']:,.2f} | {r['candidate_peak']:,.2f} | {r['saving_mib']:,.2f} | "
                     f"{r['max_absolute_difference']} | {r['mean_absolute_difference']:.10f} | "
                     f"{r['max_ordered_fp16_steps']} | {r['different_numeric_fp16_values']:,} | "
                     f"{'64/64 match' if r['tokens_match'] else 'DIFFER'} |")
    sections["logits"] = "\n".join(lines)
    lines = [
        "| Request | Post-cleanup process MiB | Increase MiB | Model live MiB | Model held MiB | GenAI held MiB | GenAI live/requested bytes | Tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    previous = None
    for r in result["sequential"]["rows"]:
        delta = 0 if previous is None else r["process_mib"] - previous
        lines.append(f"| {r['request_index']} | {r['process_mib']:,.2f} | {delta:.2f} | {r['model_live_mib']:,.2f} | "
                     f"{r['model_held_mib']:,.2f} | {r['genai_held_mib']:,.2f} | "
                     f"{r['genai_live_bytes']}/{r['genai_requested_bytes']} | 64/64 match |")
        previous = r["process_mib"]
    sections["sequential"] = "\n".join(lines)
    sections.update(fresh_export_tables(result["fresh_export"]))
    return sections


def check_documents(root: Path, sections: dict[str, str]) -> None:
    combined = "\n".join((root / name).read_text() for name in ("README.md", "FINDINGS.md"))
    for name, expected in sections.items():
        begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
        require(combined.count(begin) == combined.count(end) == 1, f"Missing/duplicate published table {name}")
        actual = combined.split(begin, 1)[1].split(end, 1)[0].strip()
        require(actual == expected, f"Published {name} table differs from committed evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "tables", "summary"))
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    parser.add_argument("--section")
    parser.add_argument("--check-docs", action="store_true")
    args = parser.parse_args()
    result = calculate(args.evidence_dir)
    sections = tables(result)
    if args.check_docs:
        check_documents(ROOT, sections)
    if args.command == "verify":
        print("PASS: evidence checksum inventory and initializer shape/quantization byte formulas.")
        print("PASS: accepted provenance declarations and CSV-derived counts, matrix and request structure agree.")
        print("PASS: 135 unique accepted records; 45 worker identities; three repetitions per group.")
        print("PASS: 24 same-checkpoint byte budgets; exact KV/logits shape arithmetic.")
        print("PASS: all 48 QMoE layer counters; exact FP16 difference metrics; all candidate output IDs.")
        print("PASS: fifteen sequential requests, 960 output IDs, allocator ordering and external GPU cleanup.")
        print("PASS: fresh-export record: MMLU paired counts, accuracies, exact McNemar p, deterministic paired CI, "
              "artifact hash/size relations to the provenance record and the documented graph.name exception.")
        if args.check_docs:
            print("PASS: every published Markdown table regenerated from committed evidence.")
    elif args.command == "summary":
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.section:
        require(args.section in sections, "Unknown table section")
        print(sections[args.section])
    else:
        for name, section in sections.items():
            print(f"## {name}\n\n{section}\n")


if __name__ == "__main__":
    main()
