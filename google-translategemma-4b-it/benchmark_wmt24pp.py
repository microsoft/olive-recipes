# -------------------------------------------------------------------------
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
"""
WMT24++ translation benchmark for TranslateGemma ONNX model.

Evaluates translation quality on the WMT24++ dataset (google/wmt24pp) using COMET.
Compares against the model card reported scores:
  - MetricX (lower=better): 5.32  (4B)
  - COMET   (higher=better): 81.6 (4B)

Usage:
    python benchmark_wmt24pp.py                                          # 5 popular lang pairs, 100 segs each
    python benchmark_wmt24pp.py --lang-pairs en-de_DE en-fr_FR en-ja_JP  # specific pairs
    python benchmark_wmt24pp.py --max-segments 0                         # all segments (slow)
    python benchmark_wmt24pp.py --lang-pairs all                         # all 55 language pairs
    python benchmark_wmt24pp.py --lang-pairs all --max-segments 150      # all 55, stratified 150/pair (~8.5h CPU)
    python benchmark_wmt24pp.py --comet-model Unbabel/XCOMET-XL          # different COMET model

Prerequisites:
    pip install unbabel-comet datasets
"""

import argparse
import json
import random
import sys
import io
import time
from pathlib import Path
from collections import defaultdict

import onnxruntime_genai as og
from transformers import AutoTokenizer
from datasets import load_dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent

ALL_LANG_PAIRS = [
    "en-ar_EG", "en-ar_SA", "en-bg_BG", "en-bn_IN", "en-ca_ES",
    "en-cs_CZ", "en-da_DK", "en-de_DE", "en-el_GR", "en-es_MX",
    "en-et_EE", "en-fa_IR", "en-fi_FI", "en-fil_PH", "en-fr_CA",
    "en-fr_FR", "en-gu_IN", "en-he_IL", "en-hi_IN", "en-hr_HR",
    "en-hu_HU", "en-id_ID", "en-is_IS", "en-it_IT", "en-ja_JP",
    "en-kn_IN", "en-ko_KR", "en-lt_LT", "en-lv_LV", "en-ml_IN",
    "en-mr_IN", "en-nl_NL", "en-no_NO", "en-pa_IN", "en-pl_PL",
    "en-pt_BR", "en-pt_PT", "en-ro_RO", "en-ru_RU", "en-sk_SK",
    "en-sl_SI", "en-sr_RS", "en-sv_SE", "en-sw_KE", "en-sw_TZ",
    "en-ta_IN", "en-te_IN", "en-th_TH", "en-tr_TR", "en-uk_UA",
    "en-ur_PK", "en-vi_VN", "en-zh_CN", "en-zh_TW", "en-zu_ZA",
]

DEFAULT_LANG_PAIRS = ["en-de_DE", "en-fr_FR", "en-es_MX", "en-ja_JP", "en-zh_CN"]


def stratified_sample(segments: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n segments with proportional domain stratification."""
    if n <= 0 or n >= len(segments):
        return segments

    by_domain = defaultdict(list)
    for seg in segments:
        by_domain[seg["domain"]].append(seg)

    total = len(segments)
    sampled = []
    remaining = n

    domains = sorted(by_domain.keys())
    for i, domain in enumerate(domains):
        pool = by_domain[domain]
        if i == len(domains) - 1:
            k = remaining
        else:
            k = max(1, round(n * len(pool) / total))
            k = min(k, len(pool), remaining)
        sampled.extend(rng.sample(pool, k))
        remaining -= k

    rng.shuffle(sampled)
    return sampled


def lp_to_lang_code(lp: str, hf_tok=None) -> str:
    """Convert WMT24++ lang pair 'en-de_DE' -> target lang code for TranslateGemma.

    TranslateGemma's chat template only accepts codes in its languages dict.
    WMT24++ uses locale codes like 'is_IS' which maps to 'is-IS', but the template
    may only know 'is'. We try the full locale first, then fall back to the base code.
    """
    target = lp.split("-", 1)[1]
    full_code = target.replace("_", "-")

    if hf_tok is not None:
        # Probe the template to see which code works
        for code in [full_code, full_code.split("-")[0]]:
            try:
                hf_tok.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": code, "text": "test"}]}],
                    tokenize=False, add_generation_prompt=True,
                )
                return code
            except Exception:
                continue
        # Last resort: return the base language code
        return full_code.split("-")[0]

    return full_code


def translate_batch(model, processor, stream, hf_tok, sources: list[str], target_lang: str, max_length: int) -> list[str]:
    """Translate a list of source texts one at a time."""
    translations = []
    for src in sources:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": target_lang,
                        "text": src,
                    }
                ],
            }
        ]
        prompt = hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=max_length, do_sample=False, top_k=1)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            tokens.append(generator.get_next_tokens()[0])

        translation = processor.decode(tokens)
        translations.append(translation.strip())
        del generator

    return translations


def score_comet(sources, hypotheses, references, comet_model_name):
    """Score translations using COMET. Returns per-segment scores and system score."""
    from comet import download_model, load_from_checkpoint

    print(f"\n  Loading COMET model: {comet_model_name}")
    model_path = download_model(comet_model_name)
    model = load_from_checkpoint(model_path)

    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    output = model.predict(data, batch_size=32, gpus=0)

    return output


def main():
    parser = argparse.ArgumentParser(description="WMT24++ benchmark for TranslateGemma ONNX model")
    parser.add_argument("--model-dir", default=str(SCRIPT_DIR / "builtin" / "cpu_and_mobile" / "models"),
                        help="Path to the ONNX model directory")
    parser.add_argument("--hf-model-dir", default=str(SCRIPT_DIR / "model"),
                        help="HF model path for tokenizer/chat template")
    parser.add_argument("--lang-pairs", nargs="+", default=DEFAULT_LANG_PAIRS,
                        help="Language pairs to evaluate (e.g. en-de_DE en-fr_FR), or 'all' for all 55")
    parser.add_argument("--max-segments", type=int, default=100,
                        help="Max segments per language pair (0 = all)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum generation length per translation")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da",
                        help="COMET model to use for scoring")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "wmt24pp_results.json"),
                        help="Output JSON file for results")
    parser.add_argument("--skip-comet", action="store_true",
                        help="Skip COMET scoring (just generate translations)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stratified sampling (default: 42)")
    parser.add_argument("--resume-from", default=None,
                        help="Resume from a previous results JSON (skip already-completed pairs)")
    args = parser.parse_args()

    if args.lang_pairs == ["all"]:
        args.lang_pairs = ALL_LANG_PAIRS

    print("=" * 70)
    print("WMT24++ Benchmark for TranslateGemma ONNX")
    print("=" * 70)
    print(f"Model:          {args.model_dir}")
    print(f"Lang pairs:     {len(args.lang_pairs)} pairs")
    print(f"Max segments:   {args.max_segments or 'all'}")
    print(f"COMET model:    {args.comet_model}")
    print(f"Sampling:       domain-stratified, seed={args.seed}")
    est_segs = args.max_segments * len(args.lang_pairs) if args.max_segments > 0 else "all"
    est_hours = (args.max_segments * len(args.lang_pairs) * 3.75 / 3600) if args.max_segments > 0 else "?"
    print(f"Est. segments:  {est_segs}  (~{est_hours:.1f}h on CPU)")
    print()

    print("Loading ONNX model...")
    model = og.Model(args.model_dir)
    processor = model.create_multimodal_processor()
    stream = processor.create_stream()
    hf_tok = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)
    print("Model loaded.\n")

    rng = random.Random(args.seed)

    all_sources = []
    all_hypotheses = []
    all_references = []
    results_per_lp = {}
    completed_lps = set()

    if args.resume_from and Path(args.resume_from).exists():
        with open(args.resume_from) as f:
            prev = json.load(f)
        results_per_lp = prev.get("per_language_pair", {})
        completed_lps = set(results_per_lp.keys())
        print(f"Resuming: {len(completed_lps)} pairs already completed, skipping them.\n")

    t_start = time.time()

    for lp_idx, lp in enumerate(args.lang_pairs):
        target_lang = lp_to_lang_code(lp, hf_tok)
        elapsed_total = time.time() - t_start

        ds = list(load_dataset("google/wmt24pp", lp, split="train"))
        segments = [x for x in ds if not x["is_bad_source"]]

        if args.max_segments > 0:
            # Always sample so RNG stays in sync even when resuming
            segments = stratified_sample(segments, args.max_segments, rng)

        if lp in completed_lps:
            print(f"--- [{lp_idx+1}/{len(args.lang_pairs)}] {lp} (target: {target_lang})  [SKIPPED - resumed] ---")
            continue

        print(f"--- [{lp_idx+1}/{len(args.lang_pairs)}] {lp} (target: {target_lang})  [elapsed: {elapsed_total/60:.0f}m] ---")

        sources = [seg["source"] for seg in segments]
        references = [seg["target"] for seg in segments]

        print(f"  Translating {len(sources)} segments...", flush=True)
        t0 = time.time()
        hypotheses = translate_batch(model, processor, stream, hf_tok, sources, target_lang, args.max_length)
        elapsed = time.time() - t0
        seg_per_sec = len(sources) / elapsed if elapsed > 0 else 0

        print(f"  Done in {elapsed:.1f}s ({seg_per_sec:.2f} seg/s)")

        # Show a few examples
        for i in range(min(3, len(sources))):
            print(f"    src: {sources[i][:80]}")
            print(f"    hyp: {hypotheses[i][:80]}")
            print(f"    ref: {references[i][:80]}")
            print()

        results_per_lp[lp] = {
            "num_segments": len(sources),
            "translate_time_s": round(elapsed, 1),
            "seg_per_sec": round(seg_per_sec, 2),
        }

        all_sources.extend(sources)
        all_hypotheses.extend(hypotheses)
        all_references.extend(references)

    new_pairs = len(all_sources) // (args.max_segments or 1) if all_sources else 0
    total_pairs_done = len(results_per_lp)
    print(f"\nNew translations: {len(all_sources)} segments across {total_pairs_done - len(completed_lps)} new pairs")
    print(f"Total pairs:      {total_pairs_done} (including {len(completed_lps)} resumed)\n")

    # COMET scoring (only for newly translated pairs)
    if not args.skip_comet and all_sources:
        output = score_comet(all_sources, all_hypotheses, all_references, args.comet_model)
        scores = output.scores

        # Assign per-LP COMET scores for new pairs
        offset = 0
        for lp in args.lang_pairs:
            if lp in completed_lps:
                continue
            if lp not in results_per_lp:
                continue
            n = results_per_lp[lp]["num_segments"]
            lp_scores = scores[offset:offset + n]
            lp_comet = sum(lp_scores) / len(lp_scores) if lp_scores else 0
            results_per_lp[lp]["comet"] = round(lp_comet * 100, 2)
            offset += n

    # Compute system average from all per-LP COMET scores (resumed + new)
    lp_comets = [(r["num_segments"], r["comet"]) for r in results_per_lp.values() if "comet" in r]
    if lp_comets:
        total_segs = sum(n for n, _ in lp_comets)
        system_comet = sum(n * c for n, c in lp_comets) / total_segs
    else:
        system_comet = None

    print(f"\n{'=' * 70}")
    print(f"COMET Results (model: {args.comet_model})")
    print(f"{'=' * 70}")
    print(f"{'Lang Pair':<15} {'Segments':>8} {'COMET':>8}  {'Note'}")
    print("-" * 50)
    for lp in args.lang_pairs:
        if lp not in results_per_lp:
            continue
        r = results_per_lp[lp]
        note = "(resumed)" if lp in completed_lps else ""
        print(f"{lp:<15} {r['num_segments']:>8} {r.get('comet', 'N/A'):>8}  {note}")
    print("-" * 50)
    if system_comet is not None:
        print(f"{'SYSTEM':>15} {total_segs:>8} {round(system_comet, 2):>8}")
        print()
        print(f"Model card (4B WMT24++ COMET):  81.6")
        print(f"This run COMET:                 {round(system_comet, 2)}")

    # Save results
    total_elapsed = time.time() - t_start
    final_results = {
        "model_dir": args.model_dir,
        "comet_model": args.comet_model,
        "max_segments_per_lp": args.max_segments,
        "seed": args.seed,
        "sampling": "domain-stratified",
        "total_segments": total_segs if lp_comets else len(all_sources),
        "total_lang_pairs": len(results_per_lp),
        "total_time_hours": round(total_elapsed / 3600, 2),
        "system_comet": round(system_comet, 2) if system_comet is not None else None,
        "model_card_comet": 81.6,
        "per_language_pair": results_per_lp,
    }

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
