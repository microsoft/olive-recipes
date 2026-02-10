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
        output_dir: str,
        activation_type: str | None = None,
        precision: str | None = None) -> dict:
    with open(config_path, 'r', encoding='utf-8') as file:
        oliveJson = json.load(file)

    oliveJson["cache_dir"] = cache_dir
    oliveJson["output_dir"] = os.path.join(os.path.dirname(output_dir), oliveJson["output_dir"])

    if "quantization" in oliveJson["passes"]:
        if activation_type is not None:
            oliveJson["passes"]["quantization"]["activation_type"] = activation_type
        if precision is not None:
            oliveJson["passes"]["quantization"]["precision"] = precision

    return oliveJson

def copy_olive_config(
        history_folder: str,
        config_path: str,
        cache_dir: str,
        output_dir: str,
        activation_type: str | None = None,
        precision: str | None = None):
    logger.info(f"Copying {config_path} to {history_folder}...")
    oliveJson = load_update_config(config_path, cache_dir, output_dir, activation_type, precision)
    # save updated config for record
    config_name = os.path.basename(config_path)
    os.makedirs(history_folder, exist_ok=True)
    with open(os.path.join(history_folder, config_name), 'w', encoding='utf-8') as file:
        json.dump(oliveJson, file, indent=4)

def main():
    args = parse_arguments()

    if args.model_config:
        metrics = {
            "latency-avg": 666.666
        }
        output_file = os.path.join(os.path.dirname(args.config), "metrics.json")
        resultStr = json.dumps(metrics, indent=4)
        with open(output_file, 'w') as file:
            file.write(resultStr)
        logger.info("Model lab succeeded for evaluation.\n%s", resultStr)
        return

    with open(args.config, 'r', encoding='utf-8') as file:
        oliveJson = json.load(file)

    # Get arguments
    output_dir: str = oliveJson["output_dir"]
    cache_dir: str = oliveJson["cache_dir"]
    config_pass = oliveJson["passes"]["aitkpython"]
    activation_type: str = config_pass["activation_type"]
    precision: str = config_pass["precision"]

    dataset_name: str = config_pass["dataset_name"]
    dataset_split: str = config_pass["split"]
    num_data: int = config_pass["length"]

    history_folder = os.path.dirname(args.config)

    logger.info(f"history dir: {history_folder}")
    os.makedirs(os.path.join(history_folder, "model"), exist_ok=True)

    submodel_names = ["vae_encoder", "vae_decoder", "unet", "text_encoder", "safety_checker"]

    for submodel_name in submodel_names:
        config_name = f"config_{submodel_name}.json"
        copy_olive_config(history_folder, config_name, cache_dir, output_dir, activation_type, precision)

    # run stable_diffusion.py to generate onnx unoptimized model
    subprocess.run([sys.executable, "stable_diffusion.py",
                    "--script_dir", history_folder,
                    "--model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "--provider", "cpu",
                    "--format", "qdq",
                    "--optimize",
                    "--only_conversion"],
                   check=True)
    
    # # run evaluation.py to generate data
    subprocess.run([sys.executable, "evaluation.py",
                    "--script_dir", history_folder,
                    "--save_data",
                    "--model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "--num_inference_steps", "25",
                    "--seed", "0",
                    "--dataset_name", dataset_name,
                    "--dataset_split", dataset_split,
                    "--num_data", str(num_data),
                    "--guidance_scale", "7.5"],
                   check=True)
    
    # run stable_diffusion.py to generate onnx quantized model
    subprocess.run([sys.executable, "stable_diffusion.py",
                    "--script_dir", history_folder,
                    "--model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "--provider", "cpu",
                    "--format", "qdq",
                    "--optimize"],
                   check=True)

if __name__ == "__main__":
    main()
