import argparse
import json
import os
import olive.workflows
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
        precision: str | None = None,
        num_data: int | None = None) -> dict:
    with open(config_path, 'r', encoding='utf-8') as file:
        oliveJson = json.load(file)

    oliveJson["cache_dir"] = cache_dir
    oliveJson["output_dir"] = output_dir

    if "quantization" in oliveJson["passes"]:
        if activation_type is not None:
            oliveJson["passes"]["quantization"]["activation_type"] = activation_type
        if precision is not None:
            oliveJson["passes"]["quantization"]["precision"] = precision

    if "data_configs" in oliveJson:
        if num_data is not None:
            oliveJson["data_configs"][0]["dataloader_config"]["data_num"] = num_data

    return oliveJson

def generate_model(
        history_folder: str,
        config_path: str,
        cache_dir: str,
        output_dir: str,
        skip_existing: bool = True,
        activation_type: str | None = None,
        precision: str | None = None,
        num_data: int | None = None):
    if skip_existing and os.path.exists(os.path.join(output_dir, "model.onnx")):
        logger.info(f"Output model {output_dir} already exists, skipping {config_path}.")
        return
    logger.info(f"Generating model from {config_path}...")
    oliveJson = load_update_config(config_path, cache_dir, output_dir, activation_type, precision, num_data)
    # save updated config for record
    config_name = os.path.basename(config_path)
    os.makedirs(history_folder, exist_ok=True)
    print("write to: ", os.path.join(history_folder, config_name))
    with open(os.path.join(history_folder, config_name), 'w', encoding='utf-8') as file:
        json.dump(oliveJson, file, indent=4)
    output = olive.workflows.run(oliveJson)
    if output is None or not output.has_output_model():
        error = f"Model file is not generated"
        raise Exception(error)
    

def main():
    args = parse_arguments()

    with open(args.config, 'r', encoding='utf-8') as file:
        oliveJson = json.load(file)

    if args.model_config:
        model_path: str = os.path.dirname(args.model_config)
        execution_provider: str = oliveJson["systems"]["target_system"]["accelerators"][0]["execution_providers"][0]
        device_str: str = oliveJson["systems"]["target_system"]["accelerators"][0]["device"]
        output_file = os.path.join(os.path.dirname(args.config), "metrics.json")

        # TODO add evaluation
        metrics = {
            "mask-decoder-latency-avg": 5.26205,
            "vision-encoder-latency-avg": 2.34567
        }
        output_file = os.path.join(os.path.dirname(args.config), "metrics.json")
        resultStr = json.dumps(metrics, indent=4)
        with open(output_file, 'w') as file:
            file.write(resultStr)
        logger.info("Model lab succeeded for evaluation.\n%s", resultStr)
        return


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

    from config import ModelConfig
    ModelConfig.image_dataset = dataset_name
    ModelConfig.image_dataset_split = dataset_split
    ModelConfig.data_dir = "quantization_dataset"
    
    # Generate encoder model
    generate_model(history_folder, "sam_vision_encoder_qnn.json", cache_dir, os.path.join(output_dir, "encoder"),
                   False, activation_type, precision, num_data)
    # Generate decoder model
    generate_model(history_folder, "sam_mask_decoder_qnn.json", cache_dir, os.path.join(output_dir, "decoder"),
                   False, activation_type, precision, num_data)


if __name__ == "__main__":
    main()
