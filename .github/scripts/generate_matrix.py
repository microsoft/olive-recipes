# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Script to scan the input directory for files with name "olive_ci.json"
and generate output that can be set as strategy matrix for github job.

Example:
    python generate_matrix.py <input directory> <ubuntu|windows> <cpu|cuda>
"""
import json
import sys
from pathlib import Path

_defaults = {
    "requirements_file": "",
}

dirpath = Path(sys.argv[1])
os = sys.argv[2]
device = sys.argv[3]

recipes = []
for filepath in dirpath.rglob("olive_ci.json"):
    with filepath.open() as strm:
        for config in json.load(strm):
            if config["os"] == os and config["device"] == device:
                config["name"] = f"{filepath.parent.name} | {config['name']} | {os} | {device}"
                config["path"] = str(filepath)
                config["cwd"] = str(filepath.parent.relative_to(dirpath))

                for key, value in _defaults.items():
                    if key not in config:
                        config[key] = value

                recipes.append(config)

matrix = {"include": recipes}
output = json.dumps(matrix)
print(output)
