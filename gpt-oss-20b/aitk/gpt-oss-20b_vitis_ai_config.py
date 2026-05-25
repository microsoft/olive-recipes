import argparse
import json
import logging
import os

import olive.workflows

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)

INNER_CONFIG = "gpt-oss-20b_quark_vitisai_llm.json"
HF_REPO = "onnxruntime/gpt-oss-20b-onnx"
HF_SUBDIR = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
LOCAL_DIR = "./models/gpt-oss-20b-onnx"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to input config file")
    parser.add_argument("--model_config", help="path to input model config file")
    parser.add_argument("--runtime", required=True, help="runtime")
    return parser.parse_args()


def download_onnx_model() -> str:
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s (%s/*) to %s", HF_REPO, HF_SUBDIR, LOCAL_DIR)
    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"{HF_SUBDIR}/*"],
        local_dir=LOCAL_DIR,
    )
    return os.path.join(LOCAL_DIR, HF_SUBDIR)


def main():
    args = parse_arguments()

    with open(args.config, "r", encoding="utf-8") as f:
        outerJson = json.load(f)

    if args.model_config:
        # Evaluation entrypoint — no custom evaluator wired up for this recipe.
        return

    output_dir = outerJson["output_dir"]
    cache_dir = outerJson["cache_dir"]

    model_dir = download_onnx_model()

    inner_path = os.path.join(os.path.dirname(args.config), INNER_CONFIG)
    with open(inner_path, "r", encoding="utf-8") as f:
        innerJson = json.load(f)

    innerJson["input_model"]["model_path"] = model_dir
    innerJson["cache_dir"] = cache_dir
    innerJson["output_dir"] = output_dir

    history_folder = os.path.dirname(args.config)
    os.makedirs(history_folder, exist_ok=True)
    with open(os.path.join(history_folder, INNER_CONFIG), "w", encoding="utf-8") as f:
        json.dump(innerJson, f, indent=4)

    output = olive.workflows.run(innerJson)
    if output is None or not output.has_output_model():
        raise Exception("Model file is not generated")


if __name__ == "__main__":
    main()
