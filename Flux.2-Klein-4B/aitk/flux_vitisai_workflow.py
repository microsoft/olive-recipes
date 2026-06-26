# -------------------------------------------------------------------------
# AITK AitkPython wrapper for FLUX.2-klein-4B (AMD NPU / VitisAI).
#
# AITK invokes this script from the project (aitk) folder with:
#   python flux_vitisai_workflow.py --config <wrapper.json> --runtime <ep>
#                                  [--model_config <model.json>]
#
# `--config` lives in the run/history folder (where the staged per-component
# configs and the assembled pipeline are written). The project scripts
# (export_models.py, user_script.py) stay in this folder and are NOT copied
# into the history folder, so export_models.py is referenced at its real path.
# -------------------------------------------------------------------------

import argparse
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local AITK copies are prefixed "vitisai_" (so sanitize can diff them against
# RyzenAI/config_*.json); export_models.py consumes the prefix-less names.
SUBMODEL_NAMES = ["transformer", "text_encoder", "vae_decoder", "vae_encoder"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to input config file")
    parser.add_argument("--model_config", help="path to input model config file")
    parser.add_argument("--runtime", required=True, help="runtime")
    return parser.parse_args()


def load_update_config(config_path: str, cache_dir: str, output_dir: str) -> dict:
    """Load a component olive config and re-root its cache/output into the run area."""
    with open(config_path, "r", encoding="utf-8") as file:
        oliveJson = json.load(file)

    oliveJson["cache_dir"] = cache_dir
    oliveJson["output_dir"] = os.path.join(os.path.dirname(output_dir), oliveJson["output_dir"])

    return oliveJson


def main():
    args = parse_arguments()

    # Evaluation entrypoint is not wired up for this recipe.
    if args.model_config:
        logger.info("model_config provided but evaluation is not implemented; skipping.")
        return

    with open(args.config, "r", encoding="utf-8") as file:
        oliveJson = json.load(file)

    cache_dir: str = oliveJson["cache_dir"]
    output_dir: str = oliveJson["output_dir"]

    history_folder = os.path.dirname(os.path.abspath(args.config))
    logger.info(f"history dir: {history_folder}")

    # Stage each component config into the run folder under the prefix-less name
    # export_models.py expects, with cache/output re-rooted into the run area.
    for name in SUBMODEL_NAMES:
        src = os.path.join(SCRIPT_DIR, f"vitisai_config_{name}.json")
        if not os.path.exists(src):
            logger.warning(f"missing component config: {src}; skipping {name}")
            continue
        updated = load_update_config(src, cache_dir, output_dir)
        dst = os.path.join(history_folder, f"config_{name}.json")
        with open(dst, "w", encoding="utf-8") as file:
            json.dump(updated, file, indent=4)
        logger.info(f"staged config_{name}.json")

    # Run export/compile/assemble. export_models.py stays in this folder, so it
    # is invoked by its real path; cwd is the run folder so the relative cache /
    # output / footprint paths in the staged configs resolve there.
    subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPT_DIR, "export_models.py"),
            "--output_dir",
            output_dir,
        ],
        cwd=history_folder,
        check=True,
    )

    logger.info(f"FLUX.2-klein-4B AMD NPU pipeline assembled under: {os.path.join(history_folder, output_dir)}")


if __name__ == "__main__":
    main()
