# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Script to scan the input directory for files with name "olive_ci.json"
and generate output that can be set as strategy matrix for github job.

When --changed-files is provided (one file path per line), only recipes
whose directory contains at least one changed file are included.
This avoids running all recipes when a PR only touches one recipe folder.

Example:
    python generate_matrix.py <input directory> <ubuntu|windows> <cpu|cuda>
    python generate_matrix.py <input directory> <ubuntu|windows> <cpu|cuda> --changed-files changed.txt
"""
import argparse
import json
import sys
from pathlib import Path

_defaults = {
    "requirements_file": "",
}

parser = argparse.ArgumentParser()
parser.add_argument("dirpath", type=Path)
parser.add_argument("os", choices=["ubuntu", "windows"])
parser.add_argument("device", choices=["cpu", "cuda"])
parser.add_argument("--changed-files", type=Path, default=None,
                    help="File containing list of changed file paths (one per line)")
parser.add_argument("--recipe-filter", type=str, default=None,
                    help="Only include recipes whose directory name contains this substring")
args = parser.parse_args()

dirpath = args.dirpath
os = args.os
device = args.device

# If changed-files is provided, compute the set of recipe directories that
# contain at least one changed file. Shared files (e.g. .github/scripts/*)
# trigger all recipes.
changed_recipe_dirs = None
if args.changed_files and args.changed_files.exists():
    changed_paths = [Path(line.strip()) for line in args.changed_files.read_text().splitlines() if line.strip()]
    changed_recipe_dirs = set()
    run_all = False
    for p in changed_paths:
        # Changes to shared CI scripts or workflows trigger all recipes
        if str(p).startswith(".github/"):
            run_all = True
            break
    if run_all:
        changed_recipe_dirs = None  # None means "run all"
    else:
        for filepath in dirpath.rglob("olive_ci.json"):
            recipe_dir = filepath.parent.relative_to(dirpath)
            for p in changed_paths:
                try:
                    # Check if the changed file is inside this recipe's directory
                    Path(p).relative_to(recipe_dir)
                    changed_recipe_dirs.add(str(recipe_dir))
                    break
                except ValueError:
                    continue

recipes = []
for filepath in dirpath.rglob("olive_ci.json"):
    recipe_dir = str(filepath.parent.relative_to(dirpath))

    # Skip recipes that weren't touched in this PR
    if changed_recipe_dirs is not None and recipe_dir not in changed_recipe_dirs:
        continue

    # Skip recipes that don't match the filter (comma-separated list)
    if args.recipe_filter:
        filters = [f.strip() for f in args.recipe_filter.split(",")]
        if not any(f in recipe_dir for f in filters):
            continue

    with filepath.open() as strm:
        for config in json.load(strm):
            if config["os"] == os and config["device"] == device:
                config["name"] = f"{filepath.parent.name} | {config['name']} | {os} | {device}"
                config["path"] = str(filepath)
                config["cwd"] = recipe_dir

                for key, value in _defaults.items():
                    if key not in config:
                        config[key] = value

                recipes.append(config)

matrix = {"include": recipes}
output = json.dumps(matrix)
print(output)
