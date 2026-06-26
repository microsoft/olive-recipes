# -------------------------------------------------------------------------
# AITK AitkPython wrapper for FLUX.2-klein-4B (AMD NPU / VitisAI).
#
# AITK invokes this script with:
#   python flux_vitisai_workflow.py --config <wrapper.json> --runtime <ep>
#                                  [--model_config <model.json>]
#
# It prepares the per-component Olive configs and delegates the real work
# (ONNX export + VitisGenerateModelSD NPU compilation + pipeline assembly)
# to export_models.py, which is copied into the same run folder.
# -------------------------------------------------------------------------

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)

# Component configs the export driver consumes. The local AITK copies are
# prefixed "vitisai_" (so sanitize can diff them against RyzenAI/config_*.json);
# export_models.py expects the prefix-less "config_<name>.json" names.
SUBMODEL_NAMES = ["transformer", "text_encoder", "vae_decoder", "vae_encoder"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to input config file")
    parser.add_argument("--model_config", help="path to input model config file")
    parser.add_argument("--runtime", required=True, help="runtime")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Evaluation entrypoint is not wired up for this recipe.
    if args.model_config:
        logger.info("model_config provided but evaluation is not implemented; skipping.")
        return

    with open(args.config, "r", encoding="utf-8") as file:
        olive_json = json.load(file)

    history_folder = os.path.dirname(os.path.abspath(args.config))
    output_dir = os.path.join(history_folder, olive_json.get("output_dir", "model/flux_vitisai"))

    # Stage the component configs under the names export_models.py reads.
    for name in SUBMODEL_NAMES:
        src = os.path.join(history_folder, f"vitisai_config_{name}.json")
        dst = os.path.join(history_folder, f"config_{name}.json")
        if not os.path.exists(src):
            logger.warning(f"missing component config: {src}; skipping {name}")
            continue
        logger.info(f"staging {os.path.basename(src)} -> {os.path.basename(dst)}")
        shutil.copyfile(src, dst)

    # Run the export/compile/assemble pipeline. export_models.py resolves its
    # config and footprint paths relative to its own location (this folder).
    subprocess.run(
        [
            sys.executable,
            os.path.join(history_folder, "export_models.py"),
            "--output_dir",
            output_dir,
        ],
        cwd=history_folder,
        check=True,
    )

    logger.info(f"FLUX.2-klein-4B AMD NPU pipeline assembled at: {output_dir}")


if __name__ == "__main__":
    main()
