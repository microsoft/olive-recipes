import argparse
import json
import logging
import os
import subprocess
import sys

import olive.workflows

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to input config file")
    parser.add_argument("--runtime", help="runtime")
    return parser.parse_args()


def load_update_config(config_path: str, cache_dir: str, output_dir: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        oliveJson = json.load(file)

    oliveJson["cache_dir"] = cache_dir
    oliveJson["output_dir"] = output_dir

    return oliveJson


def copy_to_history_folder(
    history_folder: str | None,
    config_path: str,
    cache_dir: str,
    output_dir: str,
) -> dict:
    oliveJson = load_update_config(config_path, cache_dir, output_dir)
    if history_folder is not None:
        config_name = os.path.basename(config_path)
        os.makedirs(history_folder, exist_ok=True)
        logger.info(f"Copying {config_path} to {history_folder}...")
        with open(os.path.join(history_folder, config_name), "w", encoding="utf-8") as file:
            json.dump(oliveJson, file, indent=4)
    return oliveJson


def main():
    args = parse_arguments()

    # define script and config file paths
    data_prep_script = "prep_ov_quant_data.py"
    vision_encoder_config = "sam21_vision_encoder_ov.json"
    mask_decoder_config = "sam21_mask_decoder_ov.json"

    if args.config:
        with open(args.config, "r", encoding="utf-8") as file:
            aitkJson = json.load(file)
        output_dir: str = aitkJson["output_dir"]
        cache_dir: str = aitkJson["cache_dir"]
        history_folder: str | None = os.path.dirname(args.config)
        logger.info(f"history dir: {history_folder}")
    else:
        output_dir = "model"
        cache_dir = "cache"
        history_folder = None

    # prepare the calibration data in blocking mode
    # data is required for running the olive workflows for
    # both SAM2.1 Vision Encoder and Mask Decoder models.
    subprocess.run([sys.executable, data_prep_script], check=True)

    # run SAM 2.1 Vision Encoder workflow to generate
    # Intel® OpenVINO Execution Provider ONNX Encapsulated OVIR model
    encoder_json = copy_to_history_folder(
        history_folder, vision_encoder_config, cache_dir, os.path.join(output_dir, "encoder")
    )
    output_ve = olive.workflows.run(encoder_json)
    if output_ve is None or not output_ve.has_output_model():
        error = f"Execution of {vision_encoder_config} was unsuccessful. SAM 2.1 ONNX Intel® OpenVINO IR Encapsulated Vision Encoder model file was not generated."
        raise RuntimeError(error)

    # run SAM 2.1 Mask Decoder workflow to generate
    # Intel® OpenVINO Execution Provider ONNX Encapsulated OVIR model
    decoder_json = copy_to_history_folder(
        history_folder, mask_decoder_config, cache_dir, os.path.join(output_dir, "decoder")
    )
    output_md = olive.workflows.run(decoder_json)
    if output_md is None or not output_md.has_output_model():
        error = f"Execution of {mask_decoder_config} was unsuccessful. SAM 2.1 ONNX Intel® OpenVINO IR Encapsulated Mask Decoder model file was not generated."
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
