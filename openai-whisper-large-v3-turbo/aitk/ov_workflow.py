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


def main():
    args = parse_arguments()

    with open(args.config, 'r', encoding='utf-8') as file:
        oliveJson = json.load(file)

    # Get arguments
    output_dir: str = oliveJson["output_dir"]
    cache_dir: str = oliveJson["cache_dir"]
    config_pass = oliveJson["passes"]["aitkpython"]
    weight_format = config_pass["weight_format"]
    enable_npu_ws = config_pass["enable_npu_ws"]
    
    cache_dir += "_ov_npu" if enable_npu_ws else "_ov"

    # When we have model_config, we are in evaluation
    if args.model_config:
        model_path: str = os.path.dirname(args.model_config)
        execution_provider: str = oliveJson["systems"]["target_system"]["accelerators"][0]["execution_providers"][0]
        device_str: str = oliveJson["systems"]["target_system"]["accelerators"][0]["device"]
        output_file = os.path.join(os.path.dirname(args.config), "metrics.json")

        # Run evaluator
        subprocess.run([sys.executable, "ov_evaluate.py",
                        "--execution_provider", execution_provider,
                        "--device_str", device_str,
                        "--output_file", output_file,
                        "--model_path", model_path],
                       check=True)
        return

    # Generate model
    subprocess.run([sys.executable, "convert_whisper_to_ovir.py",
                    "--output_dir", output_dir,
                    "--cache_dir", cache_dir,
                    "--model", "openai/whisper-large-v3-turbo",
                    "--weight-format", weight_format,
                    "--enable_npu_ws", str(enable_npu_ws)],
                   check=True)


if __name__ == "__main__":
    main()
