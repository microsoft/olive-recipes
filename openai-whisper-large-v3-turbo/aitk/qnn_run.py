# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import os
from qnn_app import HfWhisperAppWithSave, infer_audio, get_audio_name
import logging

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)

def register_execution_providers():
    import json
    import subprocess
    import sys
    import onnxruntime as ort

    worker_script = os.path.abspath('winml.py')
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    for item in paths.items():
        try:
            ort.register_execution_provider_library(item[0], item[1])
        except Exception as e:
            logger.warning(f"Failed to register execution provider {item[0]}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--audio-path",
        type=str,
        help="Path to folder containing audio files or a single audio file path",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        help="Path to encoder onnx file",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to decoder onnx file",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="HuggingFace Whisper model id",
    )
    parser.add_argument(
        "--execution_provider",
        type=str,
        default="CPUExecutionProvider",
        help="ORT Execution provider",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--save_data",
        type=str,
        default=None,
        help="(Optional) Path to save quantization data",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="librispeech_asr",
        help="(Optional) dataset to download",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="(Optional) dataset split to download",
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=100,
        help="Number of data samples to use for quantization. Only applicable if --save_data is enabled",
    )

    args = parser.parse_args()

    encoder_path = args.encoder
    decoder_path = args.decoder

    register_execution_providers()
    app = HfWhisperAppWithSave(encoder_path, decoder_path, args.model_id, args.execution_provider, args.device_type)

    if not os.path.exists(args.audio_path) or os.path.isdir(args.audio_path):
        from datasets import load_dataset
        import numpy as np

        os.makedirs(args.audio_path, exist_ok=True)
        streamed_dataset = load_dataset(args.dataset_name, "clean", split=args.dataset_split, streaming=True)
        i = 0
        for batch in streamed_dataset:
            i += 1
            file_path = os.path.join(args.audio_path, f"{batch['id']}.npy")
            if not os.path.exists(file_path):
                np.save(file_path, batch)

            audio_name = get_audio_name(file_path)
            if args.save_data and os.path.exists(os.path.join(args.save_data, audio_name)):
                #print(f"Skipping {file_path} as data already exists.")
                pass
            else:
                logger.info(f"Processing data {i} in {file_path} ...")
                infer_audio(app, args.model_id, file_path, args.save_data, audio_name)

            if i >= args.num_data:
                break

    else:
        infer_audio(app, args.model_id, args.audio_path, args.save_data)


if __name__ == "__main__":
    main()
