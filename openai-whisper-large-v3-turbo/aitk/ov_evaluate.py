from pathlib import Path
import argparse
import json
import time
import onnxruntime_genai as og


def test_transcript(model_path, audio_path, num_beams=0, execution_provider="OpenVINO", device_type="NPU"):
    print("Loading model...")
    print(f"Model path: {model_path}")
    config = og.Config(model_path)
    config.set_provider_option(execution_provider, "device_type", device_type)
    model = og.Model(config)
    processor = model.create_multimodal_processor()

    print("Loading audio...")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Processing audio: {audio_path}")
    batch_size = 1
    decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    prompts = ["".join(decoder_prompt_tokens)]

    print(f"Processing:")
    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_length=448,
        batch_size=batch_size,
    )

    latencies = []

    for _ in range(20):
        audios = og.Audios.open(audio_path)
        inputs = processor(prompts, audios=audios)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        while not generator.is_done():
            start_time = time.perf_counter()
            generator.generate_next_token()
            latencies.append(time.perf_counter() - start_time)

    return latencies

def main():
    parser = argparse.ArgumentParser(description="Test Whisper ONNX GenAI models.")
    parser.add_argument("--execution_provider", type=str, default="CPUExecutionProvider", help="ORT Execution provider")
    parser.add_argument("--device_str", type=str, default="cpu")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", required=True, help="Path to Whisper ONNX GenAI model")
    args = parser.parse_args()

    # model path
    model_path = args.model_path
    test_audio_url = "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav"
    test_audio_name = "how_are_you_doing_today.wav"

    import requests
    r = requests.get(test_audio_url)
    with open(test_audio_name, "wb") as audio_file:
         audio_file.write(r.content)

    from winml import register_execution_providers_to_onnxruntime_genai
    register_execution_providers_to_onnxruntime_genai()

    num_beams = 1
    latencies = test_transcript(model_path, test_audio_name, num_beams, "OpenVINO", args.device_str.upper())

    latency_avg = round(sum(latencies) / len(latencies) * 1000, 5)
    throughput_avg = round(1 / latency_avg * 1000, 5)

    metrics = {
        "latency-avg": latency_avg,
        "throughput-avg": throughput_avg,
    }
    resultStr = json.dumps(metrics, indent=4)
    with open(args.output_file, 'w') as file:
        file.write(resultStr)
    print("Model lab succeeded for evaluation.\n", resultStr)


if __name__ == "__main__":
    main()
