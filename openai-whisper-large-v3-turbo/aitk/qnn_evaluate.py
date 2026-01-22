import argparse
import json
import os
import numpy as np

from qnn_app import HfWhisperAppWithSave, get_device_type
from transformers import WhisperProcessor
import logging

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper")
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
        "--device_str",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    audio_path = args.audio_path
    encoder_path = args.encoder
    decoder_path = args.decoder

    app = HfWhisperAppWithSave(encoder_path, decoder_path, args.model_id, args.execution_provider, get_device_type(args.device_str))

    encoder_latencies = []
    decoder_latencies = []

    # only pick one from dataset
    audio_file = next((os.path.join(d, f) for d, _, fs in os.walk(audio_path) for f in fs if f.endswith(".npy")), None)
    audio_dict = np.load(audio_file, allow_pickle=True).item()

    audio = audio_dict["audio"]["array"]
    audio_sample_rate = audio_dict["audio"]["sampling_rate"]

    for _ in range(20):
        app.transcribe_tokens(audio, audio_sample_rate, None, None)
        encoder_latencies.extend(app.encoder_latencies)
        decoder_latencies.extend(app.decoder_latencies)

    encoder_latency_avg = round(sum(encoder_latencies) / len(encoder_latencies) * 1000, 5)
    decoder_latency_avg = round(sum(decoder_latencies) / len(decoder_latencies) * 1000, 5)

    metrics = {
        "encoder-latency-avg": encoder_latency_avg,
        "decoder-latency-avg": decoder_latency_avg
    }
    resultStr = json.dumps(metrics, indent=4)
    with open(args.output_file, 'w') as file:
        file.write(resultStr)
    logger.info("Model lab succeeded for evaluation.\n%s", resultStr)


if __name__ == "__main__":
    main()
