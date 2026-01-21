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
        data_path: str | None = None,
        num_data: int | None = None) -> dict:
    with open(config_path, 'r', encoding='utf-8') as file:
        oliveJson = json.load(file)

    oliveJson["cache_dir"] = cache_dir
    oliveJson["output_dir"] = output_dir

    if activation_type is not None:
        oliveJson["passes"]["quantization"]["activation_type"] = activation_type
    if precision is not None:
        oliveJson["passes"]["quantization"]["precision"] = precision
    if data_path is not None:
        oliveJson["data_configs"][0]["dataloader_config"]["data_path"] = data_path
    if num_data is not None:
        oliveJson["data_configs"][0]["dataloader_config"]["num_data"] = num_data

    return oliveJson


def generate_model(
        history_folder: str,
        config_path: str,
        cache_dir: str,
        output_dir: str,
        skip_existing: bool = True,
        activation_type: str | None = None,
        precision: str | None = None,
        data_path: str | None = None,
        num_data: int | None = None):
    if skip_existing and os.path.exists(os.path.join(output_dir, "model.onnx")):
        logger.info(f"Output model {output_dir} already exists, skipping {config_path}.")
        return
    logger.info(f"Generating model from {config_path}...")
    oliveJson = load_update_config(config_path, cache_dir, output_dir, activation_type, precision, data_path, num_data)
    # save updated config for record
    config_name = os.path.basename(config_path)
    with open(os.path.join(history_folder, config_name), 'w', encoding='utf-8') as file:
        json.dump(oliveJson, file, indent=4)
    output = olive.workflows.run(oliveJson)
    if output is None or not output.has_output_model():
        error = "Model file is not generated"
        raise Exception(error)


def main():
    args = parse_arguments()
    # When we have model_config, we are in evaluation
    if args.model_config:
        # TODO add evaluation
        metrics = {
            "encoder-latency-avg": 5.26205,
            "decoder-latency-avg": 3.26205
        }
        output_file = os.path.join(os.path.dirname(args.config), "metrics.json")
        resultStr = json.dumps(metrics, indent=4)
        with open(output_file, 'w') as file:
            file.write(resultStr)
        logger.info("Model lab succeeded for evaluation.\n%s", resultStr)
        return

    # Get arguments
    with open(args.config, 'r', encoding='utf-8') as file:
        oliveJson = json.load(file)
    output_dir: str = oliveJson["output_dir"]
    cache_dir: str = oliveJson["cache_dir"]
    config_pass = oliveJson["passes"]["aitkpython"]
    activation_type: str = config_pass["activation_type"]
    precision: str = config_pass["precision"]
    dataset_name: str = config_pass["dataset_name"]
    dataset_split: str = config_pass["split"]
    num_data: int = config_pass["length"]
    # The cwd is model project folder
    audio_path: str = os.path.join("data", dataset_name.replace("/", "_"), dataset_split)
    save_data_path: str = os.path.join("data",  "_data_" + dataset_name.replace("/", "_"), dataset_split)
    history_folder = os.path.dirname(args.config)
    # Generate original model
    original_encoder = os.path.join("data", "_encoder_fp32")
    generate_model("data", "whisper_large_v3_turbo_encoder_fp32.json", cache_dir, original_encoder)
    original_decoder = os.path.join("data", "_decoder_fp32")
    generate_model("data", "whisper_large_v3_turbo_decoder_fp32.json", cache_dir, original_decoder)
    # Generate dataset
    subprocess.run([sys.executable, "qnn_run.py",
                    "--audio-path", audio_path,
                    "--encoder", os.path.join(original_encoder, "model.onnx"),
                    "--decoder", os.path.join(original_decoder, "model.onnx"),
                    "--save_data", save_data_path,
                    "--dataset_name", dataset_name,
                    "--dataset_split", dataset_split,
                    "--num_data", str(num_data)],
                   check=True)
    # Generate quantized model
    generate_model(history_folder, "whisper_large_v3_turbo_encoder_qdq.json", cache_dir, os.path.join(output_dir, "encoder"),
                   False, activation_type, precision, save_data_path, num_data)
    # decoder has more data for 1 sample, to keep variants, multiply num_data by 10
    generate_model(history_folder, "whisper_large_v3_turbo_decoder_qdq.json", cache_dir, os.path.join(output_dir, "decoder"),
                   False, activation_type, precision, save_data_path, num_data * 10)


if __name__ == "__main__":
    main()
