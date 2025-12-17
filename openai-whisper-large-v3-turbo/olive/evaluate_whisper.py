import argparse

from app import HfWhisperAppWithSave
from datasets import load_dataset
from evaluate import load
from transformers import WhisperProcessor


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper")
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

    processor = WhisperProcessor.from_pretrained(args.model_id)
    app = HfWhisperAppWithSave(encoder_path, decoder_path, args.model_id, args.execution_provider, provider_options)

    streamed_dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

    # Prepare WER metric
    wer = load("wer")

    references = []
    predictions = []

    for item in streamed_dataset:
        audio = item["audio"]["array"]
        audio_sample_rate = item["audio"]["sampling_rate"]
        transcription = app.transcribe(audio, audio_sample_rate, None, None)
        prediction = processor.tokenizer._normalize(transcription)

        reference = processor.tokenizer._normalize(item["text"])
        references.append(reference)
        predictions.append(prediction)
        print("Reference: ", reference)
        print("prediction: ", prediction)

    # Compute WER
    print("WER:", 100 * wer.compute(references=references, predictions=predictions))


if __name__ == "__main__":
    main()
