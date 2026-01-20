import argparse
import json
import os
import time

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

    encoder_path = args.encoder
    decoder_path = args.decoder

    processor = WhisperProcessor.from_pretrained(args.model_id)
    app = HfWhisperAppWithSave(encoder_path, decoder_path, args.model_id, args.execution_provider, get_device_type(args.device_str))

    references = []
    predictions = []

    latencies = []

    if os.path.isdir(args.audio_path):
        for _, item in enumerate(os.listdir(args.audio_path)):
            import numpy as np
            
            audio_file = os.path.join(args.audio_path, item)
            audio_dict = np.load(audio_file, allow_pickle=True).item()

            audio = audio_dict["audio"]["array"]
            audio_sample_rate = audio_dict["audio"]["sampling_rate"]
            
            start_time = time.perf_counter()
            transcription = app.transcribe(audio, audio_sample_rate, None, None)
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

            prediction = processor.tokenizer._normalize(transcription)
            reference = processor.tokenizer._normalize(audio_dict["text"])
            references.append(reference)
            predictions.append(prediction)
            logger.info(f"Reference: {reference}")
            logger.info(f"prediction: {prediction}")

    latency_avg = round(sum(latencies) / len(latencies) * 1000, 5)
    metrics = {
        "latency-avg": latency_avg
    }
    resultStr = json.dumps(metrics, indent=4)
    with open(args.output_file, 'w') as file:
        file.write(resultStr)
    logger.info("Model lab succeeded for evaluation.\n%s", resultStr)


if __name__ == "__main__":
    main()
