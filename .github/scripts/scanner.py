# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import itertools
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def _scan(dirpath: Path) -> Tuple[
    Dict[str, Set[Tuple[str, Path]]], Dict[str, Set[Tuple[str, Path]]], Dict[str, Set[Tuple[str, Path]]]
]:
    grouped_by_ep = defaultdict(set)
    grouped_by_device = defaultdict(set)
    grouped_by_arch = defaultdict(set)

    for filepath in dirpath.rglob("info.yml"):
        print(f"Scanning {filepath.relative_to(dirpath)} ...")
        try:
            with filepath.open() as strm:
                data = yaml.safe_load(strm)

            relpath = filepath.relative_to(dirpath)
            model_name = relpath.parts[0]

            arch = data.get('arch')
            if arch:
                grouped_by_arch[arch].add((model_name, relpath.parent))

            recipes = data.get('recipes') or []
            for recipe in recipes:
                name = recipe.get("name") or model_name
                filename = recipe.get("file")
                if filename:
                    eps = recipe.get("eps") or recipe.get("ep") or ["CPUExecutionProvider"]
                    devices = recipe.get("devices") or recipe.get("device") or ["cpu"]

                    if isinstance(eps, str):
                        eps = [eps]

                    for ep in eps:
                        ep = ep[:-len("ExecutionProvider")]
                        grouped_by_ep[ep].add((name, relpath.parent / filename))

                    if isinstance(devices, str):
                        devices = [devices]

                    for device in devices:
                        grouped_by_device[device].add((name, relpath.parent / filename))
        except Exception as e:
            print(f"Failed to load/parse {filepath}.", e)

    return grouped_by_arch, grouped_by_device, grouped_by_ep


def _tabulate(grouped: Dict[str, Set[Tuple[str, Path]]]) -> List[str]:
    keys = sorted(grouped.keys())

    columns = [
        sorted([f'[{name}]({relpath.as_posix()})' for name, relpath in grouped[key]]) for key in keys
    ]
    columns = list(map(list, itertools.zip_longest(*columns, fillvalue="")))

    content = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join([":---:"] * len(grouped)) + " |"
    ] + [
        "| " + " | ".join(column) + " |" for column in columns
    ]

    return content


def _merge_lines(lines: List[str], begin_pattern: str, end_pattern: str, replacement: List[str]) -> List[str]:
    begin_index = lines.index(begin_pattern)
    end_index = lines.index(end_pattern)

    if begin_index >= 0 and end_index >= 0:
        # Make sure to leave the comment lines intact so next update works correctly.
        lines = lines[:begin_index + 1] + replacement + lines[end_index:]

    return lines


def _rewrite(dirpath: Path, arch_table: List[str], device_table: List[str], ep_table: List[str]):
    template_filepath = dirpath / "README.md"
    content = template_filepath.read_text(encoding="utf-8")
    lines = content.splitlines()

    lines = _merge_lines(lines, "<!-- begin_arch_models -->", "<!-- end_arch_models -->", arch_table)
    lines = _merge_lines(lines, "<!-- begin_device_models -->", "<!-- end_device_models -->", device_table)
    lines = _merge_lines(lines, "<!-- begin_ep_models -->", "<!-- end_ep_models -->", ep_table)

    content = "\n".join(lines) + "\n"
    readme_filepath = dirpath / "README.md"
    readme_filepath.write_text(content, encoding="utf-8")


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath', dest='dirpath', required=True, type=str, help='Root directory path')
    parser.add_argument('--verbose', action='store_true', help='Print trace information')
    args = parser.parse_args()
    args.dirpath = Path(args.dirpath).resolve()

    grouped_by_arch, grouped_by_device, grouped_by_ep = _scan(args.dirpath)

    if args.verbose:
        print('=' * 120)
        import pprint
        pprint.pprint({
            "grouped_by_arch": grouped_by_arch,
            "grouped_by_device": grouped_by_device,
            "grouped_by_ep": grouped_by_ep,
        })
        print('=' * 120)

    arch_table = _tabulate(grouped_by_arch)
    ep_table = _tabulate(grouped_by_ep)
    device_table = _tabulate(grouped_by_device)

    if args.verbose:
        print("Grouped by arch: ", arch_table, "\n", '=' * 120)
        print("Grouped by ep: ", ep_table, "\n", '=' * 120)
        print("Grouped by device: ", device_table, "\n", '=' * 120)

    _rewrite(args.dirpath, arch_table, device_table, ep_table)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(_main())
