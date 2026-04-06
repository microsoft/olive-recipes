import argparse
import json
import os
import subprocess
import sys
import logging

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to input config file")
    parser.add_argument("--model_config", help="path to input model config file")
    parser.add_argument("--runtime", required=True, help="runtime")
    return parser.parse_args()

def load_update_config(
        config_path: str,
        cache_dir: str,
        output_dir: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as file:
        oliveJson = json.load(file)

    oliveJson["cache_dir"] = cache_dir
    oliveJson["output_dir"] = os.path.join(os.path.dirname(output_dir), oliveJson["output_dir"])

    return oliveJson

def copy_olive_config(
        history_folder: str,
        config_path: str,
        cache_dir: str,
        output_dir: str):
    logger.info(f"Copying {config_path} to {history_folder}...")
    oliveJson = load_update_config(config_path, cache_dir, output_dir)
    # save updated config for record
    config_name = os.path.basename(config_path)
    os.makedirs(history_folder, exist_ok=True)
    with open(os.path.join(history_folder, config_name), 'w', encoding='utf-8') as file:
        json.dump(oliveJson, file, indent=4)

def main():
    args = parse_arguments()

    with open(args.config, 'r', encoding='utf-8') as file:
        oliveJson = json.load(file)

    # For static quantization, the QDQ data should match the target scenario.
    guidance_scale=str(7.5)
    num_inference_steps=str(25)

    if args.model_config:
        model_path: str = os.path.dirname(args.model_config)
        execution_provider: str = oliveJson["systems"]["target_system"]["accelerators"][0]["execution_providers"][0]
        device_str: str = oliveJson["systems"]["target_system"]["accelerators"][0]["device"]
        output_file = os.path.join(os.path.dirname(args.config), "metrics.json")

        # Run evaluator
        subprocess.run([sys.executable, "sd_ov_evaluation.py",
                        "--script_dir", os.path.dirname(model_path),
                        "--model_dir", "optimized",
                        "--model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5",
                        "--guidance_scale", guidance_scale,
                        "--num_inference_steps", num_inference_steps,
                        "--execution_provider", execution_provider,
                        "--device_str", device_str,
                        "--output_file", output_file],
                        check=True)
        return


    # Get arguments
    output_dir: str = oliveJson["output_dir"]
    cache_dir: str = oliveJson["cache_dir"]

    history_folder = os.path.dirname(args.config)

    logger.info(f"history dir: {history_folder}")
    os.makedirs(os.path.join(history_folder, "model"), exist_ok=True)

    submodel_names = ["vae_encoder", "vae_decoder", "unet", "text_encoder", "safety_checker"]

    for submodel_name in submodel_names:
        config_name = f"config_{submodel_name}.json"
        copy_olive_config(history_folder, config_name, cache_dir, output_dir)

    # run stable_diffusion.py to generate onnx model
    subprocess.run([sys.executable, "stable_diffusion.py",
                    "--script_dir", history_folder,
                    "--model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "--provider", "openvino",
                    "--optimize"],
                   check=True)

if __name__ == "__main__":
    main()
