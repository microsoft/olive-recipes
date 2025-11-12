# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import os

from app import HfWhisperAppWithSave, infer_audio


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
        "--save_data",
        type=str,
        default=None,
        help="(Optional) Path to save quantization data",
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

    provider_options = {}
    if args.execution_provider == "QNNExectionProvider":
        provider_options = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "sustained_high_performance",
            "htp_graph_finalization_optimization_mode": "3",
            "offload_graph_io_quantization": "0",
        }

    app = HfWhisperAppWithSave(encoder_path, decoder_path, args.model_id, args.execution_provider, provider_options)

    if os.path.isdir(args.audio_path):
        for i, item in enumerate(os.listdir(args.audio_path)):
            if args.save_data and i == args.num_data:
                break

            full_path = os.path.join(args.audio_path, item)
            infer_audio(app, args.model_id, full_path, args.save_data)

    else:
        infer_audio(app, args.model_id, args.audio_path, args.save_data)


if __name__ == "__main__":
    main()
